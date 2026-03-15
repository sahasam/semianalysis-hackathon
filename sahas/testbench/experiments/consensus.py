"""
Consensus Agent Orchestration Layer — with Zeus Energy Monitoring

Multiple agents debate a topic in rounds, sharing positions and updating
until they converge on agreement. Designed to be communication-heavy to
stress-test SGLang's batching and KV cache reuse.

All LLM calls are async — no thread pools, SGLang handles batching.
All data recorded via unified schema (HopLog → DecisionLog → RunLog).
"""

import asyncio
import json
import time
import argparse

from testbench.energy import init_zeus, zeus_begin, zeus_end
from testbench.schema import HopLog, DecisionLog, RunLog
from testbench.runner import sglang_chat, execute_decision, close_client, parse_json_response
from testbench.outputs import make_run_dir, save_json


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
# Personas
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
# Message builder
# ---------------------------------------------------------------------------

def _build_messages(persona: str, topic: str, rnd: int, prior_hops: list[HopLog]):
    """Build chat messages for one agent in one round."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if rnd == 0:
        messages.append({
            "role": "user",
            "content": (
                f"You are: {persona}\n\n"
                f"Topic for discussion: {topic}\n\n"
                "What is your initial position? Respond in JSON."
            ),
        })
    else:
        others_text = "\n\n".join(
            f"**{h.agent[:60]}** (confidence: {h.confidence:.1f}, "
            f"agreement: {h.agreement:.1f}):\n"
            f"Position: {h.response_text[:300]}\n"
            f"Key point: {h.key_point[:100]}"
            for h in prior_hops
        )
        messages.append({
            "role": "user",
            "content": (
                f"You are: {persona}\n\n"
                f"Topic: {topic}\n\n"
                f"Round {rnd + 1}. Here are everyone's positions from the previous round:\n\n"
                f"{others_text}\n\n"
                "Update your position considering the others' arguments. Respond in JSON."
            ),
        })

    input_chars = sum(len(m["content"]) for m in messages)
    return messages, input_chars // 4


# ---------------------------------------------------------------------------
# Async consensus task
# ---------------------------------------------------------------------------

async def run_consensus_task(
    topic: str,
    base_url: str,
    model: str,
    personas: list[str],
    max_rounds: int,
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> DecisionLog:
    """Run one consensus decision fully async."""

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
                parsed = {"position": content[:300], "confidence": 0.5,
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

    def convergence_check(round_hops: list[HopLog]) -> bool:
        if not round_hops:
            return False
        avg_agree = sum(h.agreement for h in round_hops) / len(round_hops)
        avg_conf = sum(h.confidence for h in round_hops) / len(round_hops)
        reached = avg_agree >= 0.92 and avg_conf >= 0.9

        print(f"    Consensus check: agree={avg_agree:.2f} conf={avg_conf:.2f}"
              f"{' >>> CONSENSUS <<<' if reached else ''}")
        return reached

    decision = await execute_decision(
        experiment="consensus",
        pattern="freeform",
        model=model,
        prompt=topic,
        agent_names=personas,
        max_rounds=max_rounds,
        round_agent_fn_factory=round_agent_fn_factory,
        convergence_check=convergence_check,
        energy_mode="round",
    )

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
# Multi-panel: run multiple consensus panels concurrently (fully async)
# ---------------------------------------------------------------------------

async def run_multi_panel(
    topics: list[str],
    base_url: str,
    model: str,
    personas: list[str],
    max_rounds: int,
    temperature: float,
    max_tokens: int,
) -> RunLog:
    """Run multiple consensus panels concurrently — fully async, no thread cap."""
    run = RunLog(experiment="consensus", model=model, config={
        "mode": "multi_panel", "n_panels": len(topics),
        "n_agents": len(personas), "max_rounds": max_rounds})

    zeus_begin("multi_panel_full")
    t0 = time.perf_counter()

    coros = [
        run_consensus_task(topic, base_url, model, personas, max_rounds, temperature, max_tokens)
        for topic in topics
    ]
    decisions = await asyncio.gather(*coros)

    run.total_time_s = round(time.perf_counter() - t0, 2)
    energy = zeus_end("multi_panel_full")
    batch_j = energy.get("gpu_energy_j", 0)
    run.total_energy_j = batch_j

    # Normalize overlapping per-decision Zeus readings against batch total
    _normalize_batch_energy(list(decisions), batch_j)
    for dec in decisions:
        run.decisions.append(dec)

    run.compute_derived()

    # Print summary
    print(f"\n{'='*60}")
    print("  MULTI-PANEL RESULTS")
    print(f"{'='*60}")
    for i, dec in enumerate(run.decisions):
        print(f"  Panel {i}: {dec.n_rounds} rounds, converged={dec.converged}, "
              f"{dec.total_latency_s:.1f}s, {len(dec.hops)} hops, "
              f"E={dec.total_energy_j:.1f}J E/hop={dec.energy_per_hop_j:.1f}J")

    print(f"\n  Total wall time:      {run.total_time_s:.1f}s")
    print(f"  Total hops:           {run.total_hops}")
    print(f"  Total energy:         {run.total_energy_j:.1f} J")
    print(f"  Energy per hop:       {run.energy_per_hop_j:.2f} J")
    print(f"  Energy per decision:  {run.energy_per_decision_j:.1f} J")
    print(f"  J per output token:   {run.j_per_output_token:.4f}")

    return run


# ---------------------------------------------------------------------------
# Single consensus run
# ---------------------------------------------------------------------------

async def run_single(
    topic: str,
    base_url: str,
    model: str,
    personas: list[str],
    max_rounds: int,
    temperature: float,
    max_tokens: int,
) -> RunLog:
    run = RunLog(experiment="consensus", model=model, config={
        "mode": "single", "n_agents": len(personas), "max_rounds": max_rounds})

    print(f"Topic: {topic[:100]}...")
    print(f"Agents: {len(personas)} | Max rounds: {max_rounds}")
    print(f"Model: {model}")

    dec = await run_consensus_task(topic, base_url, model, personas, max_rounds, temperature, max_tokens)
    run.decisions.append(dec)
    run.total_time_s = dec.total_latency_s
    run.compute_derived()

    # Print results
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  Converged:            {dec.converged}")
    print(f"  Rounds:               {dec.n_rounds}")
    print(f"  Total hops:           {len(dec.hops)}")
    print(f"  Total time:           {dec.total_latency_s:.2f}s")
    print(f"  Total energy:         {dec.total_energy_j:.2f} J")
    print(f"  Energy per hop:       {dec.energy_per_hop_j:.2f} J")
    print(f"  Energy per decision:  {dec.energy_per_decision_j:.2f} J")
    print(f"  J per output token:   {dec.j_per_output_token:.4f}")

    # Per-round breakdown
    rounds = sorted(set(h.round for h in dec.hops))
    print(f"\n  {'round':>5} {'hops':>5} {'avg_lat':>8} {'sum_E(J)':>9} {'avg_agree':>9} {'avg_conf':>9}")
    print(f"  {'─'*50}")
    for rnd in rounds:
        rnd_hops = [h for h in dec.hops if h.round == rnd]
        n = len(rnd_hops)
        print(f"  {rnd+1:>5} {n:>5} "
              f"{sum(h.latency_s for h in rnd_hops)/n:>8.2f} "
              f"{sum(h.energy_j for h in rnd_hops):>9.1f} "
              f"{sum(h.agreement for h in rnd_hops)/n:>9.2f} "
              f"{sum(h.confidence for h in rnd_hops)/n:>9.2f}")

    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    personas = DEFAULT_PERSONAS[:args.num_agents]

    if args.multi_panel:
        topics = [
            "What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights or outside? How do you handle catastrophic forgetting vs. stale knowledge?",
            "How should a living model handle conflicting memories — when new observations contradict stored knowledge? Should it favor recency, confidence, source authority, or consensus?",
            "What is the right security and governance model for an AI system that continuously learns? How do you audit what it knows, when it learned it, and roll back unsafe updates?",
        ]
        run = await run_multi_panel(
            topics, args.base_url, args.model, personas,
            args.max_rounds, args.temperature, args.max_tokens)
        tag = f"multipanel_{args.num_agents}agents"
    else:
        run = await run_single(
            args.topic, args.base_url, args.model, personas,
            args.max_rounds, args.temperature, args.max_tokens)
        tag = f"{args.num_agents}agents_{args.max_rounds}rounds"

    run_dir = make_run_dir("consensus", tag)
    save_json(run.to_dict(), run_dir)
    await close_client()


def main():
    parser = argparse.ArgumentParser(description="Consensus agent orchestration with energy measurement")
    parser.add_argument("--topic", default="What is the best architecture for a living, non-stateless AI model? Should persistent memory live inside the weights (continual learning / online fine-tuning), outside the weights (RAG / knowledge graphs / external memory), or in a hybrid? How do you handle catastrophic forgetting vs. stale knowledge? What are the right trade-offs between plasticity and stability, and how do you make the whole system auditable and reversible? Be specific about implementation — what concrete components, data flows, and failure modes matter most?")
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--num-agents", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--multi-panel", action="store_true",
                        help="Run multiple consensus panels concurrently for GPU saturation")
    args = parser.parse_args()

    init_zeus()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
