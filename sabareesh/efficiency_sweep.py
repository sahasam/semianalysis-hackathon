"""
Efficiency sweep: find peak J/token across KV cache sizes and concurrency levels.

Tracks both joules-per-request AND joules-per-token for true efficiency measurement.
Also captures server-side metrics (token throughput, KV cache usage) via SGLang API.
"""

import json
import re
import os
import glob
import time
import argparse
import threading
import concurrent.futures
from dataclasses import dataclass

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


def make_llm(base_url, model, temperature, max_tokens):
    return ChatOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def call_agent(llm, persona, topic, rnd, all_positions):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rnd == 0:
        messages.append({"role": "user", "content":
            f"You are: {persona}\n\nTopic: {topic}\n\nInitial position? JSON only."})
    else:
        prev = [p for p in all_positions if p["round"] == rnd - 1]
        others = "\n\n".join(
            f"**{p['agent']}** (conf:{p['confidence']:.1f} agree:{p.get('agreement_with_others','?')}):\n"
            f"{p['position']}\nKey: {p.get('key_point','?')}"
            for p in prev
        )
        messages.append({"role": "user", "content":
            f"You are: {persona}\n\nTopic: {topic}\n\n"
            f"Round {rnd+1}. Others:\n\n{others}\n\nUpdate position. JSON only."})

    # Count input tokens (rough estimate)
    input_chars = sum(len(m["content"]) for m in messages)
    input_tokens_est = input_chars // 4

    t0 = time.perf_counter()
    response = llm.invoke(messages)
    latency = time.perf_counter() - t0

    content = response.content.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if "```" in content:
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if m: content = m.group(1).strip()

    output_tokens_est = len(content) // 4

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

    return {
        "round": rnd, "agent": persona,
        "position": parsed.get("position", ""),
        "confidence": parsed.get("confidence", 0.5),
        "agreement_with_others": parsed.get("agreement_with_others", 0.5),
        "key_point": parsed.get("key_point", ""),
        "latency_s": latency,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
    }


# ---------------------------------------------------------------------------
# Panel runner
# ---------------------------------------------------------------------------

def run_panel(panel_id, llm, topic, personas, max_rounds):
    positions = []
    timings = []
    total_input_tokens = 0
    total_output_tokens = 0

    for rnd in range(max_rounds):
        rnd_start = time.perf_counter()
        new_positions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(personas)) as pool:
            futs = {pool.submit(call_agent, llm, p, topic, rnd, positions): p for p in personas}
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                new_positions.append(r)
                total_input_tokens += r["input_tokens_est"]
                total_output_tokens += r["output_tokens_est"]

        rnd_elapsed = time.perf_counter() - rnd_start
        positions.extend(new_positions)

        avg_agree = sum(p["agreement_with_others"] for p in new_positions) / len(new_positions)
        avg_conf = sum(p["confidence"] for p in new_positions) / len(new_positions)

        timings.append({
            "round": rnd, "wall_s": round(rnd_elapsed, 2),
            "avg_latency_s": round(sum(p["latency_s"] for p in new_positions) / len(new_positions), 2),
        })

        if avg_agree >= 0.92 and avg_conf >= 0.9:
            break

    return {
        "panel_id": panel_id,
        "rounds": len(timings),
        "total_requests": len(timings) * len(personas),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "consensus": True if timings and timings[-1].get("avg_latency_s") else False,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(num_panels, num_agents, max_rounds, base_url, model, temperature, max_tokens, mem_fraction):
    llm = make_llm(base_url, model, temperature, max_tokens)
    personas = PERSONAS[:num_agents]
    topics = (TOPICS * ((num_panels // len(TOPICS)) + 1))[:num_panels]

    print(f"\n{'='*60}")
    print(f"  EFFICIENCY: {num_panels}p x {num_agents}a = {num_panels*num_agents} concurrent")
    print(f"  mem_fraction={mem_fraction}")
    print(f"{'='*60}")

    # Capture server metrics before
    pre_metrics = get_server_metrics(base_url)

    window = f"eff_{mem_fraction}_{num_panels}p_{int(time.time())}"
    zeus_begin(window)
    t0 = time.perf_counter()

    results = {}
    threads = []
    lock = threading.Lock()

    def _run(pid, topic):
        r = run_panel(pid, llm, topic, personas, max_rounds)
        with lock:
            results[pid] = r
        print(f"  [Panel {pid}] Done: {r['rounds']} rounds, "
              f"{r['total_output_tokens']} output tokens")

    for i, topic in enumerate(topics):
        t = threading.Thread(target=_run, args=(i, topic))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t0
    energy = zeus_end(window)

    # Capture server metrics after
    post_metrics = get_server_metrics(base_url)

    # Aggregate
    total_reqs = sum(r["total_requests"] for r in results.values())
    total_input = sum(r["total_input_tokens"] for r in results.values())
    total_output = sum(r["total_output_tokens"] for r in results.values())
    total_tokens = total_input + total_output

    energy_j = energy.get("gpu_energy_j", 0)

    j_per_req = energy_j / total_reqs if total_reqs > 0 and energy_j else None
    j_per_token = energy_j / total_tokens if total_tokens > 0 and energy_j else None
    j_per_output_token = energy_j / total_output if total_output > 0 and energy_j else None
    tokens_per_s = total_tokens / elapsed if elapsed > 0 else 0
    output_tokens_per_s = total_output / elapsed if elapsed > 0 else 0
    avg_watts = energy_j / elapsed if energy_j and elapsed else None

    print(f"\n  {'─'*50}")
    print(f"  Requests:         {total_reqs}")
    print(f"  Input tokens:     {total_input:,}")
    print(f"  Output tokens:    {total_output:,}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"  Wall time:        {elapsed:.1f}s")
    print(f"  Throughput:       {output_tokens_per_s:.0f} output tok/s")
    if energy_j:
        print(f"  Energy:           {energy_j:.0f} J")
        print(f"  J/request:        {j_per_req:.1f}")
        print(f"  J/output_token:   {j_per_output_token:.3f}")
        print(f"  J/total_token:    {j_per_token:.3f}")
        print(f"  Avg power:        {avg_watts:.0f} W")
    print(f"  {'─'*50}")

    summary = {
        "mem_fraction": mem_fraction,
        "num_panels": num_panels,
        "num_agents": num_agents,
        "max_concurrent": num_panels * num_agents,
        "total_requests": total_reqs,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "total_time_s": round(elapsed, 2),
        "output_tokens_per_s": round(output_tokens_per_s, 1),
        "total_energy_j": energy_j if energy_j else None,
        "j_per_request": round(j_per_req, 1) if j_per_req else None,
        "j_per_output_token": round(j_per_output_token, 4) if j_per_output_token else None,
        "j_per_total_token": round(j_per_token, 4) if j_per_token else None,
        "avg_watts": round(avg_watts, 0) if avg_watts else None,
        "panels": {str(k): v for k, v in results.items()},
    }

    outfile = f"results_eff_m{str(mem_fraction).replace('.','')}_p{num_panels}_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Summary across all results
# ---------------------------------------------------------------------------

def print_summary():
    """Read all efficiency results and print a comparison table."""
    files = sorted(glob.glob("results_eff_*.json"))
    if not files:
        print("No efficiency results found.")
        return

    rows = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        rows.append(d)

    print(f"\n{'='*90}")
    print(f"  EFFICIENCY SWEEP SUMMARY — J/token across KV cache sizes and concurrency")
    print(f"{'='*90}")
    print(f"  {'mem_frac':>8} {'panels':>6} {'concur':>6} {'reqs':>5} {'out_tok':>8} "
          f"{'tok/s':>6} {'J':>8} {'J/req':>6} {'J/out_tok':>10} {'Watts':>6}")
    print(f"  {'─'*85}")

    for r in sorted(rows, key=lambda x: (x.get("mem_fraction", 0), x.get("num_panels", 0))):
        print(f"  {r.get('mem_fraction','?'):>8} {r.get('num_panels','?'):>6} "
              f"{r.get('max_concurrent','?'):>6} {r.get('total_requests','?'):>5} "
              f"{r.get('total_output_tokens','?'):>8} "
              f"{r.get('output_tokens_per_s','?'):>6} "
              f"{r.get('total_energy_j','?'):>8} "
              f"{r.get('j_per_request','?'):>6} "
              f"{r.get('j_per_output_token','?'):>10} "
              f"{r.get('avg_watts','?'):>6}")

    # Find best efficiency
    with_j = [r for r in rows if r.get("j_per_output_token")]
    if with_j:
        best = min(with_j, key=lambda x: x["j_per_output_token"])
        print(f"\n  BEST EFFICIENCY: mem_fraction={best['mem_fraction']}, "
              f"panels={best['num_panels']}, "
              f"J/output_token={best['j_per_output_token']:.4f}, "
              f"{best['avg_watts']:.0f}W")

    # Save summary
    with open("results_efficiency_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Summary saved to results_efficiency_summary.json")


if __name__ == "__main__":
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
        run_sweep(
            num_panels=args.num_panels,
            num_agents=args.num_agents,
            max_rounds=args.max_rounds,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            mem_fraction=args.mem_fraction,
        )
