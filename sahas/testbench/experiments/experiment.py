"""
Energy-measured 2-node LangGraph experiment.

Two agents:
  1. Drafter  — writes a first-pass answer
  2. Reviewer — critiques and improves it

Zeus measures GPU energy (joules) for each hop and the full pipeline.

All data recorded via unified schema (HopLog → DecisionLog → RunLog).
All LLM calls are async — no thread pools.
"""

import asyncio
import argparse

from testbench.energy import init_zeus
from testbench.schema import HopLog, DecisionLog, RunLog
from testbench.runner import sglang_chat, execute_decision, close_client
from testbench.outputs import make_run_dir, save_json


# ──────────────────────────────────────────────
# Config defaults
# ──────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_URL = "http://localhost:25000"

DEFAULT_TASK = (
    "Explain how KV cache reuse in LLM serving reduces latency "
    "and energy consumption. Be specific and technical."
)


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

async def run_experiment(task: str, base_url: str, model: str,
                         temperature: float, max_tokens: int) -> RunLog:
    """Run the 2-node drafter/reviewer pipeline."""

    # The "agents" are drafter and reviewer — they run sequentially (serial energy mode)
    agent_names = ["drafter", "reviewer"]

    # State that gets passed between agents
    draft_text = ""

    def round_agent_fn_factory(rnd, agent_name, all_hops):
        nonlocal draft_text

        if agent_name == "drafter":
            messages = [
                {"role": "user", "content": f"You are a technical writer. Answer this:\n\n{task}"}
            ]
        else:
            # Reviewer uses the drafter's output
            drafter_hops = [h for h in all_hops if h.agent == "drafter"]
            draft = drafter_hops[-1].response_text if drafter_hops else task
            messages = [
                {"role": "user", "content":
                    f"You are a senior reviewer. Improve this draft — fix errors, "
                    f"add detail, tighten the writing:\n\n{draft}"}
            ]

        input_chars = sum(len(m["content"]) for m in messages)
        input_tokens_est = input_chars // 4

        async def agent_call():
            content, out_tokens, in_tokens = await sglang_chat(
                base_url, model, messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return {
                "response_text": content,
                "input_tokens": in_tokens or input_tokens_est,
                "output_tokens": out_tokens,
                "context_length": input_tokens_est,
            }

        return agent_call

    def convergence_check(round_hops):
        # Single round — drafter and reviewer each go once
        return True

    decision = await execute_decision(
        experiment="experiment",
        pattern="drafter_reviewer",
        model=model,
        prompt=task,
        agent_names=agent_names,
        max_rounds=1,
        round_agent_fn_factory=round_agent_fn_factory,
        convergence_check=convergence_check,
        energy_mode="exact",  # Serial: exact per-hop energy
    )

    run = RunLog(experiment="experiment", model=model, config={
        "task": task[:200], "temperature": temperature, "max_tokens": max_tokens})
    run.decisions.append(decision)
    run.total_time_s = decision.total_latency_s
    run.compute_derived()

    return run


async def async_main(args):
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"GPU: {gpu_name}")
    except Exception:
        gpu_name = "unknown"

    print(f"Task: {args.task[:80]}...")
    print()

    run = await run_experiment(args.task, args.base_url, args.model,
                               args.temperature, args.max_tokens)

    dec = run.decisions[0]

    # ── Report ──
    print("\n" + "=" * 60)
    print("PER-HOP ENERGY BREAKDOWN")
    print("=" * 60)
    for h in dec.hops:
        print(f"  {h.agent:10s}  {h.latency_s:>7.2f}s  {h.energy_j:>8.2f} J")

    overhead_j = dec.total_energy_j - sum(h.energy_j for h in dec.hops)
    print(f"\n{'TOTAL':10s}  {dec.total_latency_s:>7.2f}s  {dec.total_energy_j:>8.2f} J")
    print(f"{'OVERHEAD':10s}  {'':>7s}  {overhead_j:>8.2f} J  (framework)")

    print(f"\n  Energy per hop:       {dec.energy_per_hop_j:.2f} J")
    print(f"  Energy per decision:  {dec.energy_per_decision_j:.2f} J")
    print(f"  J per output token:   {dec.j_per_output_token:.4f}")

    # ── Save ──
    run_dir = make_run_dir("experiment", "2node")
    save_json(run.to_dict(), run_dir)

    # ── Print the actual output ──
    reviewer_hops = [h for h in dec.hops if h.agent == "reviewer"]
    if reviewer_hops:
        print("\n" + "=" * 60)
        print("FINAL OUTPUT (first 500 chars)")
        print("=" * 60)
        print(reviewer_hops[-1].response_text[:500])

    await close_client()


def main():
    parser = argparse.ArgumentParser(description="2-node energy-measured experiment")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    init_zeus()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
