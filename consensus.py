"""
Consensus Agent Orchestration Layer

Multiple agents debate a topic in rounds, sharing positions and updating
until they converge on agreement. Designed to be communication-heavy to
stress-test SGLang's batching and KV cache reuse.

Architecture:
  - N agents, each with a persona, generate positions in parallel
  - Each round, every agent sees all other positions and updates
  - An aggregator checks for convergence (agreement score)
  - Repeats until consensus or max rounds

Tunable knobs (for benchmarking):
  - num_agents: more agents = more parallel requests per round
  - max_rounds: more rounds = more total requests
  - max_tokens: longer responses = more generation work
  - temperature: higher = harder to converge = more rounds
"""

import json
import time
import argparse
import concurrent.futures
from typing import TypedDict, Annotated
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def merge_rounds(old: list, new: list) -> list:
    """Reducer: append new round data to history."""
    return old + new


class ConsensusState(TypedDict):
    topic: str
    num_agents: int
    max_rounds: int
    current_round: int
    agent_personas: list[str]
    # Each entry: {"round": int, "agent": str, "position": str, "confidence": float}
    positions: Annotated[list[dict], merge_rounds]
    consensus_reached: bool
    final_summary: str
    # Timing metrics
    round_timings: Annotated[list[dict], merge_rounds]


# ---------------------------------------------------------------------------
# Shared system prompt (maximizes KV cache reuse via RadixAttention)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are participating in a structured consensus discussion.
You must respond with ONLY a valid JSON object — no explanation, no thinking, no preamble. Just the JSON.

{"position": "your current position on the topic (2-4 sentences)", "confidence": 0.0-1.0, "agreement_with_others": 0.0-1.0, "key_point": "the single most important point in your view"}

Rules:
- Be thoughtful and substantive
- Update your position based on others' arguments when they make good points
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
    """Make a single agent generate/update its position."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # First round: just the topic
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
        # Subsequent rounds: show all positions from previous round
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

    import re
    # Strip <think>...</think> reasoning block (Qwen3.5 thinking model)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Handle markdown code blocks
    if "```" in content:
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()

    # Try to extract JSON: first try the whole content, then search for it
    parsed = None
    for attempt_content in [content]:
        try:
            parsed = json.loads(attempt_content)
            break
        except json.JSONDecodeError:
            pass

    if parsed is None:
        # Find the first { and last } to extract JSON object
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            try:
                parsed = json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass

    if parsed is None:
        parsed = {
            "position": content[:200] if content else "no response",
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
    """Set up initial state."""
    return {
        "current_round": 0,
        "positions": [],
        "consensus_reached": False,
        "final_summary": "",
        "round_timings": [],
    }


def run_debate_round(state: ConsensusState, llm: ChatOpenAI) -> dict:
    """All agents generate/update positions in parallel."""
    current_round = state["current_round"]
    personas = state["agent_personas"]
    topic = state["topic"]
    all_positions = state["positions"]

    print(f"\n{'='*60}")
    print(f"  ROUND {current_round + 1}")
    print(f"{'='*60}")

    round_start = time.perf_counter()

    # Fan out: all agents in parallel via thread pool
    new_positions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(personas)) as pool:
        futures = {
            pool.submit(call_agent, llm, p, topic, current_round, all_positions): p
            for p in personas
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            new_positions.append(result)
            print(f"  [{result['agent']}] confidence={result['confidence']:.2f} "
                  f"agreement={result['agreement_with_others']:.2f} "
                  f"latency={result['latency_s']:.2f}s")
            print(f"    Key point: {result['key_point']}")

    round_elapsed = time.perf_counter() - round_start

    timing = {
        "round": current_round,
        "wall_time_s": round_elapsed,
        "num_agents": len(personas),
        "avg_latency_s": sum(p["latency_s"] for p in new_positions) / len(new_positions),
        "max_latency_s": max(p["latency_s"] for p in new_positions),
    }
    print(f"\n  Round wall time: {round_elapsed:.2f}s | "
          f"Avg latency: {timing['avg_latency_s']:.2f}s | "
          f"Max latency: {timing['max_latency_s']:.2f}s")

    return {
        "positions": new_positions,
        "round_timings": [timing],
        "current_round": current_round + 1,
    }


def check_consensus(state: ConsensusState) -> dict:
    """Check if agents have reached consensus."""
    current_round = state["current_round"]
    latest_round = current_round - 1  # run_debate_round already incremented

    latest_positions = [p for p in state["positions"] if p["round"] == latest_round]

    if not latest_positions:
        return {"consensus_reached": False}

    avg_agreement = sum(p["agreement_with_others"] for p in latest_positions) / len(latest_positions)
    avg_confidence = sum(p["confidence"] for p in latest_positions) / len(latest_positions)

    print(f"\n  Consensus check: avg_agreement={avg_agreement:.2f}, avg_confidence={avg_confidence:.2f}")

    # Consensus = high agreement AND high confidence
    reached = avg_agreement >= 0.8 and avg_confidence >= 0.7

    if reached:
        print("  >>> CONSENSUS REACHED <<<")
    elif current_round >= state["max_rounds"]:
        print(f"  Max rounds ({state['max_rounds']}) reached without consensus")

    return {"consensus_reached": reached}


def summarize(state: ConsensusState, llm: ChatOpenAI) -> dict:
    """Generate final consensus summary."""
    last_round = state["current_round"] - 1
    final_positions = [p for p in state["positions"] if p["round"] == last_round]

    positions_text = "\n".join(
        f"- {p['agent']}: {p['position']} (confidence: {p['confidence']:.1f})"
        for p in final_positions
    )

    messages = [
        {"role": "system", "content": "Summarize the consensus reached by the group. Be concise."},
        {"role": "user", "content": (
            f"Topic: {state['topic']}\n\n"
            f"Final positions after {state['current_round']} rounds:\n{positions_text}\n\n"
            f"Consensus reached: {state['consensus_reached']}\n"
            "Write a 2-3 sentence summary of the group's conclusion."
        )},
    ]

    response = llm.invoke(messages)
    return {"final_summary": response.content}


def should_continue(state: ConsensusState) -> str:
    """Route: continue debating or move to summary."""
    if state["consensus_reached"]:
        return "summarize"
    if state["current_round"] >= state["max_rounds"]:
        return "summarize"
    return "debate"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_consensus_graph(llm: ChatOpenAI) -> StateGraph:
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
# Default personas
# ---------------------------------------------------------------------------

DEFAULT_PERSONAS = [
    "Pragmatist — focuses on practical feasibility and real-world constraints",
    "Visionary — emphasizes long-term potential and innovative possibilities",
    "Skeptic — questions assumptions and looks for weaknesses in arguments",
    "Ethicist — considers moral implications, fairness, and societal impact",
    "Analyst — relies on data, evidence, and logical reasoning",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_consensus(
    topic: str,
    base_url: str = "http://localhost:25000",
    model: str = "Qwen/Qwen3.5-27B",
    num_agents: int = 5,
    max_rounds: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 300,
):
    personas = DEFAULT_PERSONAS[:num_agents]

    print(f"Topic: {topic}")
    print(f"Agents: {num_agents} | Max rounds: {max_rounds}")
    print(f"Model: {model} | Temp: {temperature} | Max tokens: {max_tokens}")
    print(f"Server: {base_url}")

    llm = make_llm(base_url, model, temperature, max_tokens)
    app = build_consensus_graph(llm)

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

    # Print results
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"Consensus reached: {result['consensus_reached']}")
    print(f"Rounds completed: {result['current_round']}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"\nSummary: {result['final_summary']}")

    # Print timing breakdown
    print(f"\n{'='*60}")
    print("  TIMING BREAKDOWN")
    print(f"{'='*60}")
    total_llm = 0
    for t in result["round_timings"]:
        print(f"  Round {t['round']+1}: wall={t['wall_time_s']:.2f}s "
              f"avg_latency={t['avg_latency_s']:.2f}s "
              f"max_latency={t['max_latency_s']:.2f}s")
        total_llm += t["wall_time_s"]

    total_requests = result["current_round"] * num_agents + 1  # +1 for summary
    print(f"\n  Total LLM requests: {total_requests}")
    print(f"  Total wall time: {total_elapsed:.2f}s")
    print(f"  Time in LLM rounds: {total_llm:.2f}s")
    print(f"  Overhead: {total_elapsed - total_llm:.2f}s")
    print(f"  Effective req/s: {total_requests / total_elapsed:.2f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consensus agent orchestration")
    parser.add_argument("--topic", default="What is the most important challenge in AI safety and how should it be addressed?")
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    run_consensus(
        topic=args.topic,
        base_url=args.base_url,
        model=args.model,
        num_agents=args.num_agents,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
