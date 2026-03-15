"""
Detailed per-hop energy profiling for consensus agents.

Runs two modes:
  1. SERIAL — agents run one at a time, Zeus measures exact energy per hop
  2. PARALLEL — agents run concurrently, energy attributed proportionally

All data recorded via unified schema (HopLog → DecisionLog → RunLog).
All LLM calls are async — no thread pools, SGLang handles batching.
"""

import asyncio
import json
import time
import argparse

from testbench.energy import init_zeus
from testbench.schema import HopLog, DecisionLog, RunLog
from testbench.runner import sglang_chat, execute_decision, close_client, parse_json_response
from testbench.outputs import make_run_dir, save_json, save_csv_rows


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


def _make_agent_fn(base_url, model, persona, topic, rnd, prior_hops, max_tokens, temperature):
    """Create an async callable for one agent."""
    messages, input_tokens_est = _build_messages(persona, topic, rnd, prior_hops)

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


# ---------------------------------------------------------------------------
# Profile run: serial or parallel via execute_decision energy_mode
# ---------------------------------------------------------------------------

async def run_profile(
    mode: str,
    topic: str,
    base_url: str,
    model: str,
    personas: list[str],
    max_rounds: int,
    temperature: float,
    max_tokens: int,
) -> DecisionLog:
    """Run profiling in serial (exact per-hop energy) or parallel (attributed) mode."""
    energy_mode = "exact" if mode == "serial" else "round"

    print(f"\n{'='*70}")
    print(f"  {mode.upper()} MODE — {'exact per-hop' if mode == 'serial' else 'time-proportional'} energy")
    print(f"{'='*70}")

    def round_agent_fn_factory(rnd, agent_name, all_hops):
        prior_hops = [h for h in all_hops if h.round == rnd - 1] if rnd > 0 else []
        return _make_agent_fn(base_url, model, agent_name, topic, rnd, prior_hops, max_tokens, temperature)

    def convergence_check(round_hops):
        if not round_hops:
            return False
        avg_agree = sum(h.agreement for h in round_hops) / len(round_hops)
        avg_conf = sum(h.confidence for h in round_hops) / len(round_hops)
        reached = avg_agree >= 0.92 and avg_conf >= 0.9

        print(f"    Consensus: agree={avg_agree:.2f} conf={avg_conf:.2f}"
              f"{' >>> CONSENSUS <<<' if reached else ''}")
        return reached

    decision = await execute_decision(
        experiment="profile",
        pattern=f"freeform_{mode}",
        model=model,
        prompt=topic,
        agent_names=personas,
        max_rounds=max_rounds,
        round_agent_fn_factory=round_agent_fn_factory,
        convergence_check=convergence_check,
        energy_mode=energy_mode,
    )

    # Print per-hop detail
    for h in decision.hops:
        print(f"    [{h.agent[:30]}] R{h.round+1} "
              f"ctx={h.context_length:>5} "
              f"in={h.input_tokens:>5} out={h.output_tokens:>4} "
              f"lat={h.latency_s:.2f}s "
              f"E={h.energy_j:.1f}J "
              f"J/out={h.j_per_output_token:.4f} "
              f"W={h.avg_watts:.0f}")

    return decision


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_hops(decision: DecisionLog, mode: str):
    """Print detailed analysis of per-hop energy data."""
    hops = decision.hops
    if not hops:
        return

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
        rh = [h for h in hops if h.round == rnd]
        n = len(rh)
        print(f"  {rnd+1:>5} "
              f"{sum(h.context_length for h in rh)/n:>8.0f} "
              f"{sum(h.input_tokens for h in rh)/n:>7.0f} "
              f"{sum(h.output_tokens for h in rh)/n:>7.0f} "
              f"{sum(h.latency_s for h in rh)/n:>8.2f} "
              f"{sum(h.energy_j for h in rh)/n:>8.1f} "
              f"{sum(h.j_per_output_token for h in rh)/n:>9.4f} "
              f"{sum(h.avg_watts for h in rh)/n:>6.0f}")

    # By agent persona
    agents = sorted(set(h.agent for h in hops))
    print(f"\n  By Agent Persona:")
    print(f"  {'agent':<35} {'avg_lat':>8} {'avg_E(J)':>8} {'avg_J/out':>9} {'avg_W':>6}")
    print(f"  {'─'*70}")
    for agent in agents:
        ah = [h for h in hops if h.agent == agent]
        n = len(ah)
        print(f"  {agent[:35]:<35} "
              f"{sum(h.latency_s for h in ah)/n:>8.2f} "
              f"{sum(h.energy_j for h in ah)/n:>8.1f} "
              f"{sum(h.j_per_output_token for h in ah)/n:>9.4f} "
              f"{sum(h.avg_watts for h in ah)/n:>6.0f}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_hops_csv(run: RunLog, run_dir):
    """Export all hops to a flat CSV for analysis."""
    rows = []
    for d in run.decisions:
        for h in d.hops:
            rows.append({
                "decision_id": h.decision_id, "hop_id": h.hop_id,
                "experiment": h.experiment, "pattern": h.pattern,
                "energy_mode": h.energy_mode,
                "round": h.round, "agent": h.agent,
                "input_tokens": h.input_tokens, "output_tokens": h.output_tokens,
                "total_tokens": h.total_tokens, "context_length": h.context_length,
                "latency_s": h.latency_s, "energy_j": h.energy_j,
                "j_per_output_token": h.j_per_output_token,
                "j_per_total_token": h.j_per_total_token,
                "avg_watts": h.avg_watts,
                "confidence": h.confidence, "agreement": h.agreement,
            })
    if rows:
        save_csv_rows(rows, list(rows[0].keys()), run_dir, "hops.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    personas = PERSONAS[:args.num_agents]
    run = RunLog(experiment="profile", model=args.model, config={
        "topic": args.topic[:100], "num_agents": args.num_agents,
        "max_rounds": args.max_rounds, "modes": []})

    # Serial mode
    if not args.parallel_only:
        run.config["modes"].append("serial")
        dec = await run_profile("serial", args.topic, args.base_url, args.model,
                                personas, args.max_rounds, args.temperature, args.max_tokens)
        run.decisions.append(dec)
        analyze_hops(dec, "serial")
        print(f"\n  Serial totals: {dec.total_latency_s:.1f}s, "
              f"{dec.total_energy_j:.1f}J, {len(dec.hops)} hops")

    # Parallel mode
    if not args.serial_only:
        run.config["modes"].append("parallel")
        dec = await run_profile("parallel", args.topic, args.base_url, args.model,
                                personas, args.max_rounds, args.temperature, args.max_tokens)
        run.decisions.append(dec)
        analyze_hops(dec, "parallel")
        print(f"\n  Parallel totals: {dec.total_latency_s:.1f}s, "
              f"{dec.total_energy_j:.1f}J, {len(dec.hops)} hops")

    run.compute_derived()

    # Serial vs parallel comparison
    serial_decs = [d for d in run.decisions if "serial" in d.pattern]
    parallel_decs = [d for d in run.decisions if "parallel" in d.pattern]
    if serial_decs and parallel_decs:
        sd, pd = serial_decs[0], parallel_decs[0]
        print(f"\n{'='*70}")
        print(f"  SERIAL vs PARALLEL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'metric':<25} {'serial':>12} {'parallel':>12} {'ratio':>8}")
        print(f"  {'─'*60}")

        def row(label, sv, pv, unit=""):
            ratio = sv / pv if pv else 0
            print(f"  {label:<25} {sv:>10.1f}{unit:>2} {pv:>10.1f}{unit:>2} {ratio:>7.2f}x")

        row("Total energy", sd.total_energy_j, pd.total_energy_j, "J")
        row("Total time", sd.total_latency_s, pd.total_latency_s, "s")
        row("Energy per hop", sd.energy_per_hop_j, pd.energy_per_hop_j, "J")
        row("J/output_token", sd.j_per_output_token, pd.j_per_output_token, "")

    # Print aggregate
    print(f"\n{'='*70}")
    print(f"  AGGREGATE")
    print(f"{'='*70}")
    print(f"  Total decisions:      {run.total_decisions}")
    print(f"  Total hops:           {run.total_hops}")
    print(f"  Total energy:         {run.total_energy_j:.1f} J")
    print(f"  Energy per hop:       {run.energy_per_hop_j:.2f} J")
    print(f"  Energy per decision:  {run.energy_per_decision_j:.1f} J")
    print(f"  J per output token:   {run.j_per_output_token:.4f}")

    mode_tag = "serial" if args.serial_only else ("parallel" if args.parallel_only else "both")
    run_dir = make_run_dir("profile", f"{mode_tag}_{args.num_agents}agents")
    save_json(run.to_dict(), run_dir)
    export_hops_csv(run, run_dir)
    print(f"  Total hops recorded: {run.total_hops}")

    await close_client()


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
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
