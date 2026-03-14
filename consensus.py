"""
Consensus Agent Orchestration Layer — with Zeus Energy Monitoring

Multiple agents debate a topic in rounds, sharing positions and updating
until they converge on agreement. Designed to be communication-heavy to
stress-test SGLang's batching and KV cache reuse.

Energy measurement via Zeus (zeus-ml) tracks GPU joules per round
and for the full pipeline.
"""

import json
import re
import time
import argparse
import concurrent.futures
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Zeus energy monitor (optional — graceful fallback if not available)
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
        print(f"Zeus not available ({e}), running without energy measurement")
        _zeus_available = False


def zeus_begin(window_name: str):
    if _zeus_available and _monitor:
        _monitor.begin_window(window_name)


def zeus_end(window_name: str) -> dict:
    if _zeus_available and _monitor:
        m = _monitor.end_window(window_name)
        return {"time_s": round(m.time, 4), "gpu_energy_j": round(m.total_energy, 4)}
    return {}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def merge_rounds(old: list, new: list) -> list:
    return old + new


class ConsensusState(TypedDict):
    topic: str
    num_agents: int
    max_rounds: int
    current_round: int
    agent_personas: list[str]
    positions: Annotated[list[dict], merge_rounds]
    consensus_reached: bool
    final_summary: str
    round_timings: Annotated[list[dict], merge_rounds]


# ---------------------------------------------------------------------------
# Shared system prompt (maximizes KV cache reuse via RadixAttention)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are participating in a structured consensus discussion among technical experts.
You must respond with ONLY a valid JSON object — no explanation, no thinking, no preamble. Just the JSON.

{"position": "your detailed technical position (3-6 sentences)", "confidence": 0.0-1.0, "agreement_with_others": 0.0-1.0, "key_point": "the single most important technical insight in your view"}

Rules:
- Be deeply technical and substantive — cite specific architectures, patterns, and trade-offs
- Update your position based on others' arguments when they make good technical points
- Move toward consensus when possible, but don't abandon strong positions without reason
- Your confidence should reflect how settled your view is
- agreement_with_others should reflect how close the group is to alignment
- OUTPUT ONLY JSON. No other text before or after.
"""


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def make_llm(base_url: str, model: str, temperature: float, max_tokens: int):
    return ChatOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ---------------------------------------------------------------------------
# Agent call (single agent, single round)
# ---------------------------------------------------------------------------

def call_agent(
    llm: ChatOpenAI,
    persona: str,
    topic: str,
    current_round: int,
    all_positions: list[dict],
) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if current_round == 0:
        messages.append({
            "role": "user",
            "content": (
                f"You are: {persona}\n\n"
                f"Topic for discussion: {topic}\n\n"
                "What is your initial position? Respond in JSON."
            ),
        })
    else:
        prev_round = current_round - 1
        prev_positions = [p for p in all_positions if p["round"] == prev_round]

        others_text = "\n\n".join(
            f"**{p['agent']}** (confidence: {p['confidence']:.1f}, "
            f"agreement: {p.get('agreement_with_others', 'N/A')}):\n"
            f"Position: {p['position']}\n"
            f"Key point: {p.get('key_point', 'N/A')}"
            for p in prev_positions
        )

        messages.append({
            "role": "user",
            "content": (
                f"You are: {persona}\n\n"
                f"Topic: {topic}\n\n"
                f"Round {current_round + 1}. Here are everyone's positions from the previous round:\n\n"
                f"{others_text}\n\n"
                "Update your position considering the others' arguments. Respond in JSON."
            ),
        })

    t0 = time.perf_counter()
    response = llm.invoke(messages)
    latency = time.perf_counter() - t0

    # Parse JSON response — handle Qwen3.5 thinking blocks
    content = response.content.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    if "```" in content:
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            try:
                parsed = json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

    if parsed is None:
        parsed = {
            "position": content[:300] if content else "no response",
            "confidence": 0.5,
            "agreement_with_others": 0.5,
            "key_point": "parse error",
        }

    return {
        "round": current_round,
        "agent": persona,
        "position": parsed.get("position", ""),
        "confidence": parsed.get("confidence", 0.5),
        "agreement_with_others": parsed.get("agreement_with_others", 0.5),
        "key_point": parsed.get("key_point", ""),
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def initialize(state: ConsensusState) -> dict:
    return {
        "current_round": 0,
        "positions": [],
        "consensus_reached": False,
        "final_summary": "",
        "round_timings": [],
    }


def run_debate_round(state: ConsensusState, llm: ChatOpenAI) -> dict:
    current_round = state["current_round"]
    personas = state["agent_personas"]
    topic = state["topic"]
    all_positions = state["positions"]

    print(f"\n{'='*60}")
    print(f"  ROUND {current_round + 1}")
    print(f"{'='*60}")

    zeus_begin(f"round_{current_round}")
    round_start = time.perf_counter()

    new_positions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(personas)) as pool:
        futures = {
            pool.submit(call_agent, llm, p, topic, current_round, all_positions): p
            for p in personas
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            new_positions.append(result)
            print(f"  [{result['agent'][:40]}] confidence={result['confidence']:.2f} "
                  f"agreement={result['agreement_with_others']:.2f} "
                  f"latency={result['latency_s']:.2f}s")
            print(f"    Key point: {result['key_point'][:80]}")

    round_elapsed = time.perf_counter() - round_start
    energy = zeus_end(f"round_{current_round}")

    timing = {
        "round": current_round,
        "wall_time_s": round_elapsed,
        "num_agents": len(personas),
        "avg_latency_s": sum(p["latency_s"] for p in new_positions) / len(new_positions),
        "max_latency_s": max(p["latency_s"] for p in new_positions),
    }
    timing.update(energy)

    energy_str = f" | Energy: {energy.get('gpu_energy_j', 'N/A')} J" if energy else ""
    print(f"\n  Round wall time: {round_elapsed:.2f}s | "
          f"Avg latency: {timing['avg_latency_s']:.2f}s | "
          f"Max latency: {timing['max_latency_s']:.2f}s{energy_str}")

    return {
        "positions": new_positions,
        "round_timings": [timing],
        "current_round": current_round + 1,
    }


def check_consensus(state: ConsensusState) -> dict:
    current_round = state["current_round"]
    latest_round = current_round - 1

    latest_positions = [p for p in state["positions"] if p["round"] == latest_round]
    if not latest_positions:
        return {"consensus_reached": False}

    avg_agreement = sum(p["agreement_with_others"] for p in latest_positions) / len(latest_positions)
    avg_confidence = sum(p["confidence"] for p in latest_positions) / len(latest_positions)

    print(f"\n  Consensus check: avg_agreement={avg_agreement:.2f}, avg_confidence={avg_confidence:.2f}")

    reached = avg_agreement >= 0.92 and avg_confidence >= 0.9

    if reached:
        print("  >>> CONSENSUS REACHED <<<")
    elif current_round >= state["max_rounds"]:
        print(f"  Max rounds ({state['max_rounds']}) reached without consensus")

    return {"consensus_reached": reached}


def summarize(state: ConsensusState, llm: ChatOpenAI) -> dict:
    last_round = state["current_round"] - 1
    final_positions = [p for p in state["positions"] if p["round"] == last_round]

    positions_text = "\n".join(
        f"- {p['agent']}: {p['position']} (confidence: {p['confidence']:.1f})"
        for p in final_positions
    )

    messages = [
        {"role": "system", "content": "Summarize the consensus reached by the group. Be concise. Do not include any thinking."},
        {"role": "user", "content": (
            f"Topic: {state['topic']}\n\n"
            f"Final positions after {state['current_round']} rounds:\n{positions_text}\n\n"
            f"Consensus reached: {state['consensus_reached']}\n"
            "Write a 3-5 sentence technical summary of the group's conclusion."
        )},
    ]

    zeus_begin("summarize")
    response = llm.invoke(messages)
    energy = zeus_end("summarize")

    return {"final_summary": response.content}


def should_continue(state: ConsensusState) -> str:
    if state["consensus_reached"]:
        return "summarize"
    if state["current_round"] >= state["max_rounds"]:
        return "summarize"
    return "debate"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_consensus_graph(llm: ChatOpenAI):
    graph = StateGraph(ConsensusState)

    graph.add_node("initialize", initialize)
    graph.add_node("debate", lambda s: run_debate_round(s, llm))
    graph.add_node("check_consensus", check_consensus)
    graph.add_node("summarize", lambda s: summarize(s, llm))

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "debate")
    graph.add_edge("debate", "check_consensus")
    graph.add_conditional_edges("check_consensus", should_continue, {
        "debate": "debate",
        "summarize": "summarize",
    })
    graph.add_edge("summarize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Default personas — technical experts for stateful/living model discussion
# ---------------------------------------------------------------------------

DEFAULT_PERSONAS = [
    "Systems Architect — expert in distributed systems, state management, and fault tolerance. Focuses on how to persist and replicate model state across nodes.",
    "ML Researcher — deep knowledge of continual learning, catastrophic forgetting, and online adaptation. Focuses on how a model learns continuously without losing prior knowledge.",
    "Infrastructure Engineer — expert in serving infrastructure, memory management, and GPU scheduling. Focuses on the operational cost and feasibility of maintaining live model state.",
    "Knowledge Graph Specialist — expert in structured knowledge representation, retrieval-augmented generation, and external memory systems. Focuses on how to give models persistent, queryable memory.",
    "Security & Governance Lead — expert in model auditing, versioning, and compliance. Focuses on how to track what a living model knows, when it learned it, and how to roll back unsafe updates.",
    "Product Architect — expert in user-facing AI systems, personalization, and context management. Focuses on how end users interact with a model that remembers and evolves.",
    "Neuroscience-Inspired Researcher — draws parallels from biological memory systems (hippocampal replay, memory consolidation). Focuses on bio-inspired architectures for persistent learning.",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_consensus(
    topic: str,
    base_url: str = "http://localhost:25000",
    model: str = "Qwen/Qwen3.5-27B",
    num_agents: int = 7,
    max_rounds: int = 15,
    temperature: float = 0.7,
    max_tokens: int = 500,
):
    personas = DEFAULT_PERSONAS[:num_agents]

    print(f"Topic: {topic}")
    print(f"Agents: {num_agents} | Max rounds: {max_rounds}")
    print(f"Model: {model} | Temp: {temperature} | Max tokens: {max_tokens}")
    print(f"Server: {base_url}")

    llm = make_llm(base_url, model, temperature, max_tokens)
    app = build_consensus_graph(llm)

    zeus_begin("full_pipeline")
    total_start = time.perf_counter()

    result = app.invoke({
        "topic": topic,
        "num_agents": num_agents,
        "max_rounds": max_rounds,
        "agent_personas": personas,
        "positions": [],
        "consensus_reached": False,
        "final_summary": "",
        "current_round": 0,
        "round_timings": [],
    })

    total_elapsed = time.perf_counter() - total_start
    total_energy = zeus_end("full_pipeline")

    # Print results
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"Consensus reached: {result['consensus_reached']}")
    print(f"Rounds completed: {result['current_round']}")
    print(f"Total time: {total_elapsed:.2f}s")
    if total_energy:
        print(f"Total GPU energy: {total_energy['gpu_energy_j']:.2f} J")
    print(f"\nSummary: {result['final_summary']}")

    # Timing + energy breakdown
    print(f"\n{'='*60}")
    print("  TIMING & ENERGY BREAKDOWN")
    print(f"{'='*60}")
    total_llm = 0
    total_round_energy = 0
    for t in result["round_timings"]:
        energy_j = t.get("gpu_energy_j", 0)
        total_round_energy += energy_j
        energy_str = f"  energy={energy_j:.1f}J" if energy_j else ""
        print(f"  Round {t['round']+1}: wall={t['wall_time_s']:.2f}s "
              f"avg_latency={t['avg_latency_s']:.2f}s "
              f"max_latency={t['max_latency_s']:.2f}s{energy_str}")
        total_llm += t["wall_time_s"]

    total_requests = result["current_round"] * num_agents + 1
    print(f"\n  Total LLM requests: {total_requests}")
    print(f"  Total wall time: {total_elapsed:.2f}s")
    print(f"  Time in LLM rounds: {total_llm:.2f}s")
    print(f"  Overhead: {total_elapsed - total_llm:.2f}s")
    print(f"  Effective req/s: {total_requests / total_elapsed:.2f}")
    if total_energy:
        print(f"  Total pipeline energy: {total_energy['gpu_energy_j']:.2f} J")
        print(f"  Energy in rounds: {total_round_energy:.2f} J")
        print(f"  Joules per request: {total_energy['gpu_energy_j'] / total_requests:.2f} J/req")
        print(f"  Joules per round: {total_round_energy / result['current_round']:.2f} J/round")

    # Save structured results
    results_out = {
        "topic": topic,
        "model": model,
        "num_agents": num_agents,
        "max_rounds": max_rounds,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "consensus_reached": result["consensus_reached"],
        "rounds_completed": result["current_round"],
        "total_requests": total_requests,
        "total_time_s": round(total_elapsed, 4),
        "total_energy_j": total_energy.get("gpu_energy_j", None),
        "round_timings": result["round_timings"],
        "positions": result["positions"],
        "final_summary": result["final_summary"],
    }

    outfile = f"results_consensus_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\n  Results saved to {outfile}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consensus agent orchestration with energy measurement")
    parser.add_argument("--topic", default="What is the best way to implement a living model that is not stateless — one that maintains persistent memory, learns continuously, and evolves its knowledge over time while remaining safe and auditable?")
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--num-agents", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    init_zeus()

    run_consensus(
        topic=args.topic,
        base_url=args.base_url,
        model=args.model,
        num_agents=args.num_agents,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
