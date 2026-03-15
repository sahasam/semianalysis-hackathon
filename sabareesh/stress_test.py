"""
Stress test: scale concurrent consensus panels to find GPU saturation point.

Measures energy + performance as we increase concurrent agent panels
from 1 to N, each running independent consensus discussions.
"""

import json
import re
import time
import argparse
import threading
import concurrent.futures

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated


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
        print(f"Zeus not available ({e}), skipping energy measurement")
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
# LLM
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
    "What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights, outside, or in a hybrid? How do you handle catastrophic forgetting vs. stale knowledge?",
    "How should a living model handle conflicting memories — when new observations contradict stored knowledge? Should it favor recency, confidence, source authority, or consensus?",
    "What is the right security and governance model for an AI that continuously learns? How do you audit what it knows, roll back unsafe updates, and maintain compliance?",
    "How do you scale a living model's memory system across distributed infrastructure? What are the failure modes when persistent state is spread across nodes?",
    "What biological memory mechanisms (hippocampal replay, memory consolidation, synaptic pruning) are most applicable to AI systems, and how should they be implemented?",
    "How should a living model handle personalization vs. shared knowledge? When one user teaches the model something, should other users benefit? What about privacy?",
    "What is the energy and compute cost of maintaining a living model vs. periodic retraining? At what scale does continuous learning become more efficient than batch updates?",
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


def call_agent(llm, persona, topic, current_round, all_positions):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if current_round == 0:
        messages.append({"role": "user", "content": (
            f"You are: {persona}\n\nTopic: {topic}\n\nWhat is your initial position? Respond in JSON."
        )})
    else:
        prev = [p for p in all_positions if p["round"] == current_round - 1]
        others = "\n\n".join(
            f"**{p['agent']}** (conf: {p['confidence']:.1f}, agree: {p.get('agreement_with_others', '?')}):\n"
            f"Position: {p['position']}\nKey point: {p.get('key_point', '?')}"
            for p in prev
        )
        messages.append({"role": "user", "content": (
            f"You are: {persona}\n\nTopic: {topic}\n\n"
            f"Round {current_round + 1}. Previous positions:\n\n{others}\n\n"
            "Update your position. Respond in JSON."
        )})

    t0 = time.perf_counter()
    response = llm.invoke(messages)
    latency = time.perf_counter() - t0

    content = response.content.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if "```" in content:
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if m:
            content = m.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        fb = content.find('{')
        lb = content.rfind('}')
        if fb != -1 and lb > fb:
            try:
                parsed = json.loads(content[fb:lb+1])
            except json.JSONDecodeError:
                pass

    if not parsed:
        parsed = {"position": content[:200], "confidence": 0.5, "agreement_with_others": 0.5, "key_point": "parse error"}

    return {
        "round": current_round, "agent": persona,
        "position": parsed.get("position", ""),
        "confidence": parsed.get("confidence", 0.5),
        "agreement_with_others": parsed.get("agreement_with_others", 0.5),
        "key_point": parsed.get("key_point", ""),
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# Single panel runner (no LangGraph overhead for tighter control)
# ---------------------------------------------------------------------------

def run_single_panel(panel_id, llm, topic, personas, max_rounds):
    """Run one consensus panel, return metrics."""
    positions = []
    timings = []

    for rnd in range(max_rounds):
        rnd_start = time.perf_counter()
        new_positions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(personas)) as pool:
            futs = {pool.submit(call_agent, llm, p, topic, rnd, positions): p for p in personas}
            for f in concurrent.futures.as_completed(futs):
                new_positions.append(f.result())

        rnd_elapsed = time.perf_counter() - rnd_start
        positions.extend(new_positions)

        avg_agree = sum(p["agreement_with_others"] for p in new_positions) / len(new_positions)
        avg_conf = sum(p["confidence"] for p in new_positions) / len(new_positions)

        timings.append({
            "round": rnd,
            "wall_s": round(rnd_elapsed, 2),
            "avg_latency_s": round(sum(p["latency_s"] for p in new_positions) / len(new_positions), 2),
            "max_latency_s": round(max(p["latency_s"] for p in new_positions), 2),
            "avg_agreement": round(avg_agree, 2),
            "avg_confidence": round(avg_conf, 2),
        })

        print(f"  [Panel {panel_id}] Round {rnd+1}: wall={rnd_elapsed:.1f}s "
              f"agree={avg_agree:.2f} conf={avg_conf:.2f}")

        if avg_agree >= 0.92 and avg_conf >= 0.9:
            print(f"  [Panel {panel_id}] CONSENSUS at round {rnd+1}")
            break

    return {
        "panel_id": panel_id,
        "topic": topic[:80],
        "rounds": len(timings),
        "total_requests": sum(t.get("num_agents", len(personas)) for t in timings) if timings else 0,
        "timings": timings,
        "consensus": timings[-1]["avg_agreement"] >= 0.92 if timings else False,
    }


def run_stress_test(num_panels, num_agents, max_rounds, base_url, model, temperature, max_tokens):
    """Run N concurrent panels and measure aggregate throughput + energy."""
    llm = make_llm(base_url, model, temperature, max_tokens)
    personas = PERSONAS[:num_agents]
    topics = TOPICS[:num_panels]

    # Pad topics if we need more panels than topics
    while len(topics) < num_panels:
        topics.append(topics[len(topics) % len(TOPICS)])

    print(f"\n{'='*60}")
    print(f"  STRESS TEST: {num_panels} panels x {num_agents} agents")
    print(f"{'='*60}")
    print(f"  Max concurrent requests: {num_panels * num_agents}")
    print(f"  Model: {model} | Temp: {temperature} | Max tokens: {max_tokens}")

    window = f"stress_{num_panels}p_{num_agents}a_{int(time.time())}"
    zeus_begin(window)
    t0 = time.perf_counter()

    results = {}
    threads = []

    def _run(pid, topic):
        results[pid] = run_single_panel(pid, llm, topic, personas, max_rounds)

    for i, topic in enumerate(topics):
        t = threading.Thread(target=_run, args=(i, topic))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - t0
    energy = zeus_end(window)

    # Aggregate metrics
    total_reqs = sum(r["rounds"] * num_agents for r in results.values())
    total_rounds = sum(r["rounds"] for r in results.values())
    all_latencies = []
    for r in results.values():
        for t in r["timings"]:
            all_latencies.append(t["avg_latency_s"])

    print(f"\n{'='*60}")
    print(f"  RESULTS: {num_panels} panels x {num_agents} agents")
    print(f"{'='*60}")
    for pid, r in sorted(results.items()):
        print(f"  Panel {pid}: {r['rounds']} rounds, consensus={r['consensus']}")
    print(f"\n  Total requests: {total_reqs}")
    print(f"  Total wall time: {elapsed:.1f}s")
    print(f"  Effective req/s: {total_reqs / elapsed:.2f}")
    print(f"  Avg round latency: {sum(all_latencies) / len(all_latencies):.2f}s" if all_latencies else "")
    if energy:
        print(f"  Total energy: {energy['gpu_energy_j']:.1f} J")
        print(f"  Joules per request: {energy['gpu_energy_j'] / total_reqs:.1f} J/req")
        print(f"  Watts (avg): {energy['gpu_energy_j'] / elapsed:.0f} W")

    summary = {
        "num_panels": num_panels,
        "num_agents": num_agents,
        "max_concurrent": num_panels * num_agents,
        "total_requests": total_reqs,
        "total_time_s": round(elapsed, 2),
        "effective_req_per_s": round(total_reqs / elapsed, 2),
        "total_energy_j": energy.get("gpu_energy_j") if energy else None,
        "j_per_request": round(energy["gpu_energy_j"] / total_reqs, 1) if energy and total_reqs > 0 else None,
        "avg_watts": round(energy["gpu_energy_j"] / elapsed, 0) if energy else None,
        "panels": {str(k): v for k, v in results.items()},
    }

    outfile = f"results_stress_{num_panels}p_{num_agents}a_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {outfile}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-panels", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    init_zeus()

    run_stress_test(
        num_panels=args.num_panels,
        num_agents=args.num_agents,
        max_rounds=args.max_rounds,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
