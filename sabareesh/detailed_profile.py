"""
Detailed per-hop energy profiling for consensus agents.

Runs two modes:
  1. SERIAL — agents run one at a time, Zeus measures exact energy per hop
  2. PARALLEL — agents run concurrently, energy attributed proportionally

Every request is logged with:
  - Agent persona, round number
  - Input/output token counts
  - Latency (TTFT proxy, generation time)
  - Energy (joules) — exact in serial, attributed in parallel
  - J/input_token, J/output_token, J/total_token
  - Context length (grows each round as positions accumulate)
"""

import json
import re
import time
import argparse
import concurrent.futures
from dataclasses import dataclass, field, asdict

from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Zeus
# ---------------------------------------------------------------------------

_zeus_available = False
_monitor = None

def init_zeus(gpu_index=0):
    global _zeus_available, _monitor
    try:
        import torch
        from zeus.monitor import ZeusMonitor
        gpu_idx = gpu_index if gpu_index >= 0 else torch.cuda.current_device()
        _monitor = ZeusMonitor(gpu_indices=[gpu_idx])
        _zeus_available = True
        print(f"Zeus energy monitoring active on GPU {gpu_idx}")
    except Exception as e:
        print(f"Zeus not available ({e})")
        _zeus_available = False

def zeus_begin(name):
    if _zeus_available and _monitor:
        _monitor.begin_window(name)

def zeus_end(name):
    if _zeus_available and _monitor:
        m = _monitor.end_window(name)
        return {"time_s": round(m.time, 4), "gpu_energy_j": round(m.total_energy, 4)}
    return {}


# ---------------------------------------------------------------------------
# Request tracking
# ---------------------------------------------------------------------------

@dataclass
class HopRecord:
    mode: str               # "serial" or "parallel"
    round: int
    agent: str
    input_tokens: int       # estimated
    output_tokens: int      # estimated
    total_tokens: int
    context_length: int     # cumulative prompt size (grows each round)
    latency_s: float
    energy_j: float         # exact (serial) or attributed (parallel)
    j_per_input_token: float
    j_per_output_token: float
    j_per_total_token: float
    watts_during_hop: float
    position: str
    confidence: float
    agreement: float
    key_point: str


# ---------------------------------------------------------------------------
# LLM + Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are participating in a structured consensus discussion among technical experts.
You must respond with ONLY a valid JSON object — no explanation, no thinking, no preamble. Just the JSON.

{"position": "your detailed technical position (3-6 sentences)", "confidence": 0.0-1.0, "agreement_with_others": 0.0-1.0, "key_point": "the single most important technical insight in your view"}

Rules:
- Be deeply technical and substantive
- Update your position based on others' arguments when they make good technical points
- Move toward consensus when possible, but don't abandon strong positions without reason
- OUTPUT ONLY JSON. No other text before or after.
"""

PERSONAS = [
    "Systems Architect — distributed systems and state management expert",
    "ML Researcher — continual learning and catastrophic forgetting expert",
    "Infrastructure Engineer — GPU scheduling and memory management expert",
    "Knowledge Graph Specialist — structured knowledge and RAG expert",
    "Security Lead — model auditing, versioning, and compliance expert",
]

def make_llm(base_url, model, temperature, max_tokens):
    return ChatOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def build_messages(persona, topic, rnd, all_positions):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rnd == 0:
        messages.append({"role": "user", "content":
            f"You are: {persona}\n\nTopic: {topic}\n\nInitial position? JSON only."})
    else:
        prev = [p for p in all_positions if p["round"] == rnd - 1]
        others = "\n\n".join(
            f"**{p['agent']}** (conf:{p['confidence']:.1f} agree:{p.get('agreement','?')}):\n"
            f"{p['position']}\nKey: {p.get('key_point','?')}"
            for p in prev
        )
        messages.append({"role": "user", "content":
            f"You are: {persona}\n\nTopic: {topic}\n\n"
            f"Round {rnd+1}. Others:\n\n{others}\n\nUpdate position. JSON only."})

    # Estimate context length
    context_chars = sum(len(m["content"]) for m in messages)
    context_tokens = context_chars // 4
    return messages, context_tokens


def parse_response(content):
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if "```" in content:
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if m: content = m.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        fb, lb = content.find('{'), content.rfind('}')
        if fb != -1 and lb > fb:
            try: parsed = json.loads(content[fb:lb+1])
            except: pass
    if not parsed:
        parsed = {"position": content[:200], "confidence": 0.5,
                  "agreement_with_others": 0.5, "key_point": "parse error"}

    output_tokens = len(content) // 4
    return parsed, output_tokens


def call_agent_measured(llm, persona, topic, rnd, all_positions, mode="serial"):
    """Call a single agent with full measurement. In serial mode, Zeus wraps each call."""
    messages, context_tokens = build_messages(persona, topic, rnd, all_positions)
    input_tokens = context_tokens  # prompt is the context

    window = f"hop_{mode}_{rnd}_{persona[:15]}_{int(time.time()*1000)}"

    # In serial mode, Zeus measures this exact hop
    if mode == "serial":
        zeus_begin(window)

    t0 = time.perf_counter()
    response = llm.invoke(messages)
    latency = time.perf_counter() - t0

    energy = {}
    if mode == "serial":
        energy = zeus_end(window)

    parsed, output_tokens = parse_response(response.content.strip())
    total_tokens = input_tokens + output_tokens
    energy_j = energy.get("gpu_energy_j", 0)

    return {
        "round": rnd,
        "agent": persona,
        "position": parsed.get("position", ""),
        "confidence": parsed.get("confidence", 0.5),
        "agreement": parsed.get("agreement_with_others", 0.5),
        "key_point": parsed.get("key_point", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "context_length": context_tokens,
        "latency_s": latency,
        "energy_j": energy_j,
    }


# ---------------------------------------------------------------------------
# Serial profiling — exact per-hop energy
# ---------------------------------------------------------------------------

def run_serial(llm, topic, personas, max_rounds):
    """Run agents one at a time. Zeus measures each hop individually."""
    print(f"\n{'='*70}")
    print(f"  SERIAL MODE — exact per-hop energy measurement")
    print(f"{'='*70}")

    positions = []
    hops = []

    zeus_begin("serial_full")
    t0 = time.perf_counter()

    for rnd in range(max_rounds):
        print(f"\n  Round {rnd+1}:")
        new_positions = []

        for persona in personas:
            result = call_agent_measured(llm, persona, topic, rnd, positions, mode="serial")
            new_positions.append(result)

            e = result["energy_j"]
            j_per_out = e / result["output_tokens"] if e and result["output_tokens"] else 0
            j_per_in = e / result["input_tokens"] if e and result["input_tokens"] else 0
            j_per_tot = e / result["total_tokens"] if e and result["total_tokens"] else 0
            watts = e / result["latency_s"] if e and result["latency_s"] else 0

            hop = HopRecord(
                mode="serial", round=rnd, agent=persona,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                total_tokens=result["total_tokens"],
                context_length=result["context_length"],
                latency_s=round(result["latency_s"], 3),
                energy_j=round(e, 2),
                j_per_input_token=round(j_per_in, 4),
                j_per_output_token=round(j_per_out, 4),
                j_per_total_token=round(j_per_tot, 4),
                watts_during_hop=round(watts, 0),
                position=result["position"][:100],
                confidence=result["confidence"],
                agreement=result["agreement"],
                key_point=result["key_point"][:80],
            )
            hops.append(hop)

            print(f"    [{persona[:30]}] "
                  f"ctx={result['context_length']:>5} "
                  f"in={result['input_tokens']:>5} "
                  f"out={result['output_tokens']:>4} "
                  f"lat={result['latency_s']:.2f}s "
                  f"E={e:.1f}J "
                  f"J/out={j_per_out:.3f} "
                  f"W={watts:.0f}")

        positions.extend(new_positions)

        # Check consensus
        avg_agree = sum(p["agreement"] for p in new_positions) / len(new_positions)
        avg_conf = sum(p["confidence"] for p in new_positions) / len(new_positions)
        print(f"    Consensus: agree={avg_agree:.2f} conf={avg_conf:.2f}")

        if avg_agree >= 0.92 and avg_conf >= 0.9:
            print(f"    >>> CONSENSUS at round {rnd+1} <<<")
            break

    total_elapsed = time.perf_counter() - t0
    total_energy = zeus_end("serial_full")

    return hops, total_elapsed, total_energy, positions


# ---------------------------------------------------------------------------
# Parallel profiling — attributed per-hop energy
# ---------------------------------------------------------------------------

def run_parallel(llm, topic, personas, max_rounds):
    """Run agents concurrently. Energy attributed proportionally by latency."""
    print(f"\n{'='*70}")
    print(f"  PARALLEL MODE — time-proportional energy attribution")
    print(f"{'='*70}")

    positions = []
    hops = []

    zeus_begin("parallel_full")
    t0 = time.perf_counter()

    for rnd in range(max_rounds):
        print(f"\n  Round {rnd+1}:")

        round_window = f"parallel_round_{rnd}_{int(time.time()*1000)}"
        zeus_begin(round_window)
        rnd_start = time.perf_counter()

        new_positions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(personas)) as pool:
            futs = {
                pool.submit(call_agent_measured, llm, p, topic, rnd, positions, "parallel"): p
                for p in personas
            }
            for f in concurrent.futures.as_completed(futs):
                new_positions.append(f.result())

        rnd_elapsed = time.perf_counter() - rnd_start
        round_energy = zeus_end(round_window)
        round_j = round_energy.get("gpu_energy_j", 0)

        # Attribute energy proportionally by latency
        total_latency = sum(r["latency_s"] for r in new_positions)

        for result in new_positions:
            frac = result["latency_s"] / total_latency if total_latency > 0 else 1.0 / len(new_positions)
            attributed_j = round_j * frac

            j_per_out = attributed_j / result["output_tokens"] if attributed_j and result["output_tokens"] else 0
            j_per_in = attributed_j / result["input_tokens"] if attributed_j and result["input_tokens"] else 0
            j_per_tot = attributed_j / result["total_tokens"] if attributed_j and result["total_tokens"] else 0
            watts = attributed_j / result["latency_s"] if attributed_j and result["latency_s"] else 0

            hop = HopRecord(
                mode="parallel", round=rnd, agent=result["agent"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                total_tokens=result["total_tokens"],
                context_length=result["context_length"],
                latency_s=round(result["latency_s"], 3),
                energy_j=round(attributed_j, 2),
                j_per_input_token=round(j_per_in, 4),
                j_per_output_token=round(j_per_out, 4),
                j_per_total_token=round(j_per_tot, 4),
                watts_during_hop=round(watts, 0),
                position=result["position"][:100],
                confidence=result["confidence"],
                agreement=result["agreement"],
                key_point=result["key_point"][:80],
            )
            hops.append(hop)

            print(f"    [{result['agent'][:30]}] "
                  f"ctx={result['context_length']:>5} "
                  f"in={result['input_tokens']:>5} "
                  f"out={result['output_tokens']:>4} "
                  f"lat={result['latency_s']:.2f}s "
                  f"E={attributed_j:.1f}J(attr) "
                  f"J/out={j_per_out:.3f} "
                  f"W={watts:.0f}")

        positions.extend(new_positions)

        avg_agree = sum(p["agreement"] for p in new_positions) / len(new_positions)
        avg_conf = sum(p["confidence"] for p in new_positions) / len(new_positions)
        print(f"    Round energy: {round_j:.1f}J | wall: {rnd_elapsed:.2f}s")
        print(f"    Consensus: agree={avg_agree:.2f} conf={avg_conf:.2f}")

        if avg_agree >= 0.92 and avg_conf >= 0.9:
            print(f"    >>> CONSENSUS at round {rnd+1} <<<")
            break

    total_elapsed = time.perf_counter() - t0
    total_energy = zeus_end("parallel_full")

    return hops, total_elapsed, total_energy, positions


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_hops(hops, mode):
    """Print detailed analysis of per-hop energy data."""
    print(f"\n{'='*70}")
    print(f"  DETAILED ANALYSIS — {mode.upper()} MODE")
    print(f"{'='*70}")

    # By round
    rounds = sorted(set(h.round for h in hops))
    print(f"\n  By Round (context grows each round):")
    print(f"  {'round':>5} {'avg_ctx':>8} {'avg_in':>7} {'avg_out':>7} "
          f"{'avg_lat':>8} {'avg_E(J)':>8} {'avg_J/out':>9} {'avg_W':>6}")
    print(f"  {'─'*65}")
    for rnd in rounds:
        rnd_hops = [h for h in hops if h.round == rnd]
        print(f"  {rnd+1:>5} "
              f"{sum(h.context_length for h in rnd_hops)/len(rnd_hops):>8.0f} "
              f"{sum(h.input_tokens for h in rnd_hops)/len(rnd_hops):>7.0f} "
              f"{sum(h.output_tokens for h in rnd_hops)/len(rnd_hops):>7.0f} "
              f"{sum(h.latency_s for h in rnd_hops)/len(rnd_hops):>8.2f} "
              f"{sum(h.energy_j for h in rnd_hops)/len(rnd_hops):>8.1f} "
              f"{sum(h.j_per_output_token for h in rnd_hops)/len(rnd_hops):>9.4f} "
              f"{sum(h.watts_during_hop for h in rnd_hops)/len(rnd_hops):>6.0f}")

    # By agent persona
    agents = sorted(set(h.agent for h in hops))
    print(f"\n  By Agent Persona:")
    print(f"  {'agent':<35} {'avg_lat':>8} {'avg_E(J)':>8} {'avg_J/out':>9} {'avg_W':>6}")
    print(f"  {'─'*70}")
    for agent in agents:
        a_hops = [h for h in hops if h.agent == agent]
        print(f"  {agent[:35]:<35} "
              f"{sum(h.latency_s for h in a_hops)/len(a_hops):>8.2f} "
              f"{sum(h.energy_j for h in a_hops)/len(a_hops):>8.1f} "
              f"{sum(h.j_per_output_token for h in a_hops)/len(a_hops):>9.4f} "
              f"{sum(h.watts_during_hop for h in a_hops)/len(a_hops):>6.0f}")

    # Energy vs context length correlation
    print(f"\n  Energy vs Context Length (does energy grow with context?):")
    for h in sorted(hops, key=lambda x: x.context_length):
        print(f"    ctx={h.context_length:>5} → E={h.energy_j:>7.1f}J "
              f"lat={h.latency_s:.2f}s J/out={h.j_per_output_token:.4f} "
              f"[{h.agent[:20]} R{h.round+1}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detailed per-hop energy profiling")
    parser.add_argument("--topic", default="What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights, outside, or in a hybrid? How do you handle catastrophic forgetting vs. stale knowledge?")
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--serial-only", action="store_true")
    parser.add_argument("--parallel-only", action="store_true")
    args = parser.parse_args()

    init_zeus()

    all_hops = []

    # Serial mode — exact per-hop energy
    if not args.parallel_only:
        serial_hops, serial_time, serial_energy, _ = run_serial(
            make_llm(args.base_url, args.model, args.temperature, args.max_tokens),
            args.topic, PERSONAS[:args.num_agents], args.max_rounds)
        all_hops.extend(serial_hops)
        analyze_hops(serial_hops, "serial")

        print(f"\n  Serial totals: {serial_time:.1f}s, "
              f"{serial_energy.get('gpu_energy_j', '?')}J, "
              f"{len(serial_hops)} hops")

    # Parallel mode — attributed energy
    if not args.serial_only:
        parallel_hops, parallel_time, parallel_energy, _ = run_parallel(
            make_llm(args.base_url, args.model, args.temperature, args.max_tokens),
            args.topic, PERSONAS[:args.num_agents], args.max_rounds)
        all_hops.extend(parallel_hops)
        analyze_hops(parallel_hops, "parallel")

        print(f"\n  Parallel totals: {parallel_time:.1f}s, "
              f"{parallel_energy.get('gpu_energy_j', '?')}J, "
              f"{len(parallel_hops)} hops")

    # Compare serial vs parallel
    if not args.serial_only and not args.parallel_only:
        s_hops = [h for h in all_hops if h.mode == "serial"]
        p_hops = [h for h in all_hops if h.mode == "parallel"]

        print(f"\n{'='*70}")
        print(f"  SERIAL vs PARALLEL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'metric':<25} {'serial':>12} {'parallel':>12} {'ratio':>8}")
        print(f"  {'─'*60}")

        s_j = sum(h.energy_j for h in s_hops)
        p_j = sum(h.energy_j for h in p_hops)
        s_out = sum(h.output_tokens for h in s_hops)
        p_out = sum(h.output_tokens for h in p_hops)

        def row(label, sv, pv, unit=""):
            ratio = sv / pv if pv else 0
            print(f"  {label:<25} {sv:>10.1f}{unit:>2} {pv:>10.1f}{unit:>2} {ratio:>7.2f}x")

        row("Total energy", s_j, p_j, "J")
        row("Total time", serial_time, parallel_time, "s")
        row("Avg J/output_token", s_j/s_out if s_out else 0, p_j/p_out if p_out else 0, "")
        row("Avg latency/hop", sum(h.latency_s for h in s_hops)/len(s_hops),
            sum(h.latency_s for h in p_hops)/len(p_hops), "s")

    # Save all hop records
    outfile = f"results_detailed_profile_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump([asdict(h) for h in all_hops], f, indent=2)
    print(f"\n  Detailed hop records saved to {outfile}")
    print(f"  Total hops recorded: {len(all_hops)}")


if __name__ == "__main__":
    main()
