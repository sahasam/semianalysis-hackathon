"""
Efficiency sweep: find peak J/token across KV cache sizes and concurrency levels.

Tracks both joules-per-request AND joules-per-token for true efficiency measurement.
Also captures server-side metrics (token throughput, KV cache usage) via SGLang API.

All data recorded via unified schema (HopLog → DecisionLog → RunLog).
All LLM calls are async — no thread pools, SGLang handles batching.
"""

import asyncio
import json
import time
import argparse

from testbench.energy import init_zeus, zeus_begin, zeus_end
from testbench.schema import HopLog, DecisionLog, RunLog
from testbench.runner import sglang_chat, execute_decision, close_client, parse_json_response
from testbench.outputs import make_run_dir, save_json, get_results_root


# ---------------------------------------------------------------------------
# Server metrics
# ---------------------------------------------------------------------------

def get_server_metrics(base_url="http://localhost:25000"):
    """Get KV cache usage and other stats from SGLang."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/get_server_info", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Shared config
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
    "Product Architect — user-facing AI and personalization expert",
    "Neuroscience Researcher — bio-inspired memory and consolidation expert",
]

TOPICS = [
    "What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights, outside, or in a hybrid?",
    "How should a living model handle conflicting memories — when new observations contradict stored knowledge?",
    "What is the right security and governance model for an AI that continuously learns?",
    "How do you scale a living model's memory system across distributed infrastructure?",
    "What biological memory mechanisms are most applicable to AI systems?",
    "How should a living model handle personalization vs. shared knowledge?",
    "What is the energy cost of maintaining a living model vs. periodic retraining?",
]


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(persona, topic, rnd, prior_hops):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rnd == 0:
        messages.append({"role": "user", "content":
            f"You are: {persona}\n\nTopic: {topic}\n\nInitial position? JSON only."})
    else:
        others = "\n\n".join(
            f"**{h.agent[:40]}** (conf:{h.confidence:.1f} agree:{h.agreement:.1f}):\n"
            f"{h.response_text[:300]}\nKey: {h.key_point[:100]}"
            for h in prior_hops
        )
        messages.append({"role": "user", "content":
            f"You are: {persona}\n\nTopic: {topic}\n\n"
            f"Round {rnd+1}. Others:\n\n{others}\n\nUpdate position. JSON only."})

    input_chars = sum(len(m["content"]) for m in messages)
    return messages, input_chars // 4


# ---------------------------------------------------------------------------
# Async panel runner
# ---------------------------------------------------------------------------

async def run_panel(
    panel_id: int,
    topic: str,
    base_url: str,
    model: str,
    personas: list[str],
    max_rounds: int,
    temperature: float,
    max_tokens: int,
) -> DecisionLog:
    """Run a single consensus panel fully async."""

    def round_agent_fn_factory(rnd, agent_name, all_hops):
        prior_hops = [h for h in all_hops if h.round == rnd - 1] if rnd > 0 else []
        messages, input_tokens_est = _build_messages(agent_name, topic, rnd, prior_hops)

        async def agent_call():
            content, out_tokens, in_tokens = await sglang_chat(
                base_url, model, messages,
                max_tokens=max_tokens, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            parsed = parse_json_response(content)
            if parsed is None:
                parsed = {"position": content[:200], "confidence": 0.5,
                          "agreement_with_others": 0.5, "key_point": "parse error"}
            return {
                "response_text": parsed.get("position", ""),
                "input_tokens": in_tokens or input_tokens_est,
                "output_tokens": out_tokens,
                "context_length": input_tokens_est,
                "confidence": parsed.get("confidence", 0.5),
                "agreement": parsed.get("agreement_with_others", 0.5),
                "key_point": parsed.get("key_point", ""),
            }
        return agent_call

    def convergence_check(round_hops):
        if not round_hops:
            return False
        avg_agree = sum(h.agreement for h in round_hops) / len(round_hops)
        avg_conf = sum(h.confidence for h in round_hops) / len(round_hops)
        return avg_agree >= 0.92 and avg_conf >= 0.9

    decision = await execute_decision(
        experiment="efficiency",
        pattern="freeform",
        model=model,
        prompt=topic,
        agent_names=personas,
        max_rounds=max_rounds,
        round_agent_fn_factory=round_agent_fn_factory,
        convergence_check=convergence_check,
        energy_mode="round",
    )

    print(f"  [Panel {panel_id}] Done: {decision.n_rounds} rounds, "
          f"{decision.total_output_tokens} out_tok, E={decision.total_energy_j:.1f}J "
          f"E/hop={decision.energy_per_hop_j:.1f}J")

    return decision


# ---------------------------------------------------------------------------
# Energy normalization
# ---------------------------------------------------------------------------

def _normalize_batch_energy(decisions, batch_total_j):
    """Scale overlapping per-decision Zeus readings so sum == batch total."""
    raw_sum = sum(d.raw_energy_j for d in decisions)
    for dec in decisions:
        if raw_sum > 0:
            dec.total_energy_j = round(dec.raw_energy_j / raw_sum * batch_total_j, 4)
        else:
            dec.total_energy_j = round(batch_total_j / len(decisions), 4)
        dec.compute_derived()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

async def run_sweep(num_panels, num_agents, max_rounds, base_url, model,
                    temperature, max_tokens, mem_fraction):
    personas = PERSONAS[:num_agents]
    topics = (TOPICS * ((num_panels // len(TOPICS)) + 1))[:num_panels]

    print(f"\n{'='*60}")
    print(f"  EFFICIENCY: {num_panels}p x {num_agents}a = {num_panels*num_agents} concurrent")
    print(f"  mem_fraction={mem_fraction}")
    print(f"{'='*60}")

    run = RunLog(experiment="efficiency", model=model, config={
        "mem_fraction": mem_fraction, "num_panels": num_panels,
        "num_agents": num_agents, "max_rounds": max_rounds})

    window = f"eff_{mem_fraction}_{num_panels}p_{int(time.time())}"
    zeus_begin(window)
    t0 = time.perf_counter()

    # Fire all panels concurrently — fully async, no thread cap
    coros = [
        run_panel(i, topic, base_url, model, personas, max_rounds, temperature, max_tokens)
        for i, topic in enumerate(topics)
    ]
    decisions = await asyncio.gather(*coros)

    run.total_time_s = round(time.perf_counter() - t0, 2)
    energy = zeus_end(window)
    batch_j = energy.get("gpu_energy_j", 0)
    run.total_energy_j = batch_j

    # Normalize overlapping per-decision Zeus readings against batch total
    _normalize_batch_energy(list(decisions), batch_j)
    for dec in decisions:
        run.decisions.append(dec)

    run.compute_derived()

    # Aggregate stats
    total_reqs = run.total_hops
    total_output = sum(d.total_output_tokens for d in run.decisions)
    total_tokens = sum(d.total_tokens for d in run.decisions)
    output_tokens_per_s = total_output / run.total_time_s if run.total_time_s > 0 else 0
    avg_watts = run.total_energy_j / run.total_time_s if run.total_energy_j and run.total_time_s else None

    print(f"\n  {'─'*50}")
    print(f"  Hops (requests):  {total_reqs}")
    print(f"  Decisions:        {run.total_decisions}")
    print(f"  Output tokens:    {total_output:,}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"  Wall time:        {run.total_time_s:.1f}s")
    print(f"  Throughput:       {output_tokens_per_s:.0f} output tok/s")
    if run.total_energy_j:
        print(f"  Energy:           {run.total_energy_j:.0f} J")
        print(f"  Energy/hop:       {run.energy_per_hop_j:.2f} J")
        print(f"  Energy/decision:  {run.energy_per_decision_j:.1f} J")
        print(f"  J/output_token:   {run.j_per_output_token:.4f}")
        if avg_watts:
            print(f"  Avg power:        {avg_watts:.0f} W")
    print(f"  {'─'*50}")

    # Save with mem_fraction in config
    mem_tag = str(mem_fraction).replace('.', '')
    run_dir = make_run_dir("efficiency", f"m{mem_tag}_{num_panels}panels_{num_agents}agents")
    save_json(run.to_dict(), run_dir)

    await close_client()
    return run


# ---------------------------------------------------------------------------
# Summary across all results
# ---------------------------------------------------------------------------

def print_summary():
    """Read all efficiency results and print a comparison table."""
    results_root = get_results_root() / "efficiency"
    if not results_root.exists():
        print("No efficiency results found.")
        return

    rows = []
    for result_dir in sorted(results_root.iterdir()):
        result_file = result_dir / "results.json"
        if result_file.exists():
            with open(result_file) as fh:
                d = json.load(fh)
            d["_source_dir"] = str(result_dir.name)
            rows.append(d)

    if not rows:
        print("No efficiency results found.")
        return

    print(f"\n{'='*100}")
    print(f"  EFFICIENCY SWEEP SUMMARY")
    print(f"{'='*100}")
    print(f"  {'run_dir':<35} {'mem':>5} {'panels':>6} {'hops':>5} {'decs':>5} "
          f"{'out_tok':>8} {'tok/s':>6} {'E(J)':>8} {'E/hop':>7} {'E/dec':>7} {'J/out':>8} {'W':>6}")
    print(f"  {'─'*110}")

    for r in sorted(rows, key=lambda x: (
            x.get("config", {}).get("mem_fraction", 0),
            x.get("config", {}).get("num_panels", 0))):
        cfg = r.get("config", {})
        total_out = sum(d.get("total_output_tokens", 0) for d in r.get("decisions", []))
        tok_s = total_out / r.get("total_time_s", 1) if r.get("total_time_s") else 0
        watts = r.get("total_energy_j", 0) / r.get("total_time_s", 1) if r.get("total_energy_j") and r.get("total_time_s") else 0
        print(f"  {r.get('_source_dir','?'):<35} "
              f"{cfg.get('mem_fraction','?'):>5} {cfg.get('num_panels','?'):>6} "
              f"{r.get('total_hops','?'):>5} {r.get('total_decisions','?'):>5} "
              f"{total_out:>8} {tok_s:>6.0f} "
              f"{r.get('total_energy_j','?'):>8} "
              f"{r.get('energy_per_hop_j','?'):>7} "
              f"{r.get('energy_per_decision_j','?'):>7} "
              f"{r.get('j_per_output_token','?'):>8} "
              f"{watts:>6.0f}")

    # Find best efficiency
    with_j = [r for r in rows if r.get("j_per_output_token")]
    if with_j:
        best = min(with_j, key=lambda x: x["j_per_output_token"])
        cfg = best.get("config", {})
        print(f"\n  BEST: mem={cfg.get('mem_fraction')}, panels={cfg.get('num_panels')}, "
              f"J/out_tok={best['j_per_output_token']:.4f}, "
              f"E/hop={best.get('energy_per_hop_j','?')}J, "
              f"E/dec={best.get('energy_per_decision_j','?')}J")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--mem-fraction", type=float, default=0.80)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        print_summary()
    else:
        init_zeus()
        asyncio.run(run_sweep(
            num_panels=args.num_panels,
            num_agents=args.num_agents,
            max_rounds=args.max_rounds,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            mem_fraction=args.mem_fraction,
        ))


if __name__ == "__main__":
    main()
