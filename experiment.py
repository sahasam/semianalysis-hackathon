"""
Energy-measured 2-node LangGraph experiment.
 
Two agents:
  1. Drafter  — writes a first-pass answer
  2. Reviewer — critiques and improves it
 
Zeus measures GPU energy (joules) for each hop and the full pipeline.
"""
 
import json
import torch
from zeus.monitor import ZeusMonitor
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
 
# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "http://localhost:25000/v1"
GPU_INDEX = torch.cuda.current_device()
 
TASK = (
    "Explain how KV cache reuse in LLM serving reduces latency "
    "and energy consumption. Be specific and technical."
)
 
# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="not-needed",
    model=MODEL,
    temperature=0.7,
    max_tokens=512,
)
 
monitor = ZeusMonitor(gpu_indices=[GPU_INDEX])
hop_results = []
 
 
# ──────────────────────────────────────────────
# Energy-measured wrapper
# ──────────────────────────────────────────────
def measured_node(name, fn):
    """Wrap a LangGraph node so Zeus records its energy."""
    def wrapper(state):
        monitor.begin_window(name)
        output = fn(state)
        m = monitor.end_window(name)
        hop_results.append({
            "node": name,
            "time_s": round(m.time, 4),
            "gpu_energy_j": round(m.total_energy, 4),
        })
        print(f"  [{name}] {m.time:.2f}s, {m.total_energy:.2f} J")
        return output
    return wrapper
 
 
# ──────────────────────────────────────────────
# Agent nodes
# ──────────────────────────────────────────────
class State(TypedDict):
    task: str
    draft: str
    final: str
 
 
def drafter(state: State) -> dict:
    resp = llm.invoke(f"You are a technical writer. Answer this:\n\n{state['task']}")
    return {"draft": resp.content}
 
 
def reviewer(state: State) -> dict:
    resp = llm.invoke(
        f"You are a senior reviewer. Improve this draft — fix errors, "
        f"add detail, tighten the writing:\n\n{state['draft']}"
    )
    return {"final": resp.content}
 
 
# ──────────────────────────────────────────────
# Build graph
# ──────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("drafter", measured_node("drafter", drafter))
graph.add_node("reviewer", measured_node("reviewer", reviewer))
graph.add_edge(START, "drafter")
graph.add_edge("drafter", "reviewer")
graph.add_edge("reviewer", END)
app = graph.compile()
 
 
# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GPU {GPU_INDEX}: {torch.cuda.get_device_name(GPU_INDEX)}")
    print(f"Task: {TASK[:80]}...")
    print()
 
    # Measure the full pipeline
    monitor.begin_window("full_pipeline")
    output = app.invoke({"task": TASK})
    total = monitor.end_window("full_pipeline")
 
    # ── Report ──
    print("\n" + "=" * 60)
    print("PER-HOP ENERGY BREAKDOWN")
    print("=" * 60)
    for r in hop_results:
        print(f"  {r['node']:10s}  {r['time_s']:>7.2f}s  {r['gpu_energy_j']:>8.2f} J")
 
    print(f"\n{'TOTAL':10s}  {total.time:>7.2f}s  {total.total_energy:>8.2f} J")
 
    overhead_j = total.total_energy - sum(r["gpu_energy_j"] for r in hop_results)
    print(f"{'OVERHEAD':10s}  {'':>7s}  {overhead_j:>8.2f} J  (graph/framework)")
 
    # ── Save structured results ──
    results = {
        "task": TASK,
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(GPU_INDEX),
        "hops": hop_results,
        "total_time_s": round(total.time, 4),
        "total_energy_j": round(total.total_energy, 4),
        "overhead_j": round(overhead_j, 4),
    }
 
    outfile = "results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")
 
    # ── Print the actual output ──
    print("\n" + "=" * 60)
    print("FINAL OUTPUT (first 500 chars)")
    print("=" * 60)
    print(output["final"][:500])
