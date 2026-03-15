"""
Energy Benchmarking: Multi-Agent Consensus Patterns

Compares 4 communication patterns for multi-agent consensus:
  1. Single-Token Select  — ~1 token/agent/round (sgl.select equivalent)
  2. JSON Consensus       — ~15 tokens/agent/round (constrained JSON)
  3. CoT + Select         — ~150 tokens + 1 vote/agent/round
  4. Full NL Debate       — ~350 tokens/agent/round

Core metric: Joules to reach consensus (all agents agree)

All data recorded via unified schema (HopLog → DecisionLog → RunLog).
All LLM calls are async — no thread pool limits, SGLang handles batching.
"""

import asyncio
import json
import re
import sys
import time
import argparse

from testbench.energy import init_zeus, zeus_begin, zeus_end
from testbench.schema import HopLog, DecisionLog, RunLog
from testbench.runner import sglang_chat, execute_decision, close_client, strip_thinking
from testbench.outputs import make_run_dir, save_json, save_csv_rows


async def _gather_with_progress(coros, label: str) -> list:
    """Run coroutines concurrently, printing as each completes."""
    total = len(coros)
    results = [None] * total
    done_count = 0

    async def _wrap(idx, coro):
        nonlocal done_count
        result = await coro
        done_count += 1
        converge = "YES" if result.converged else "NO"
        print(f"    [{done_count:>3}/{total}] {label} task {idx} done: "
              f"{converge} {result.n_rounds}R {result.total_output_tokens}tok "
              f"{result.total_latency_s:.1f}s", flush=True)
        results[idx] = result
        return result

    await asyncio.gather(*[_wrap(i, c) for i, c in enumerate(coros)])
    return results


# ---------------------------------------------------------------------------
# Classification task prompts (customer intent)
# ---------------------------------------------------------------------------

CATEGORIES = ["billing", "technical", "cancellation", "general"]

PROMPTS = [
    "I was charged twice for my subscription last month and I need a refund immediately.",
    "My internet keeps dropping every 30 minutes and I've already restarted the router.",
    "I want to cancel my account effective immediately and get a prorated refund.",
    "What are your business hours and do you have a location near downtown?",
    "The app crashes every time I try to upload a file larger than 10MB.",
    "Can you explain the difference between your premium and basic plans?",
    "I need to update the credit card on file for automatic payments.",
    "My smart home device won't connect to the WiFi after the firmware update.",
    "I'm moving next month and need to transfer my service to a new address.",
    "Your website says my payment failed but my bank shows it went through.",
    "I've been a customer for 5 years and I'm thinking about switching to a competitor.",
    "The remote control for my cable box isn't responding to any button presses.",
    "I'd like to add a second line to my family plan. What are the options?",
    "My bill this month is $40 higher than usual and I don't see any explanation.",
    "I can't log into my account even after resetting my password three times.",
    "Do you offer any senior citizen discounts on your monthly plans?",
    "The picture quality on my streaming is terrible even though I have gigabit internet.",
    "I want to downgrade my plan but keep my current phone number.",
    "Someone made unauthorized purchases on my account and I need it frozen.",
    "How do I set up parental controls on the new set-top box?",
    "I'm not satisfied with the resolution to my previous complaint case #45231.",
    "The technician didn't show up during the scheduled maintenance window.",
    "Can I pause my subscription for two months while I'm traveling abroad?",
    "My voicemail stopped working after the system update last night.",
    "I received a promotional offer in the mail but the code isn't working online.",
    "The data usage shown on my account doesn't match what my phone reports.",
    "I need to dispute a late fee — I set up autopay but it didn't process.",
    "Is there a way to get a paper bill instead of electronic statements?",
    "My home security camera keeps showing offline even though WiFi is fine.",
    "I want to file a formal complaint about the customer service I received.",
    "The new router you sent has different ports than my old one — need help.",
    "Can you waive the activation fee? I saw a competitor offering free activation.",
    "My account was suspended but I've already made the overdue payment.",
    "I need help configuring port forwarding for my gaming console.",
    "What happens to my stored data if I cancel my cloud storage subscription?",
    "The mobile app notifications are delayed by hours — is this a known issue?",
    "I'd like to return the equipment within the 30-day trial period.",
    "My business needs a dedicated static IP address — how do I request one?",
    "The on-demand movie I rented won't play — it just shows a black screen.",
    "I need an itemized breakdown of all charges on my last three invoices.",
    "How do I transfer ownership of my account to my spouse?",
    "The noise cancellation on my new headset produces a buzzing sound.",
    "I want to bundle my internet, TV, and phone services for a discount.",
    "My automated backup hasn't run in two weeks and I'm worried about data loss.",
    "Can you confirm whether my contract has an early termination fee?",
    "The chatbot couldn't help me and I need to speak with a real person.",
    "I accidentally deleted my voicemail greeting — how do I re-record it?",
    "My neighbor is getting faster speeds on the same plan — can you check my line?",
    "I need to schedule a technician visit for Saturday between 9am and noon.",
    "What's your policy on refunding unused prepaid service credit?",
]


# ---------------------------------------------------------------------------
# Pattern-specific message builders + vote extractors
# ---------------------------------------------------------------------------

def _build_select_messages(prompt, prior_votes, categories):
    messages = [
        {"role": "system", "content": "You are a customer service classifier. Output ONLY one word from the allowed categories. No explanation."},
        {"role": "user", "content": (
            f"Classify this customer message into one category.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Message: {prompt}\n"
            + (f"\nPrevious round votes from other agents: {prior_votes}\n" if prior_votes else "")
            + f"\nYour classification (one word only):"
        )}
    ]
    return messages, 5, 0.1, {"chat_template_kwargs": {"enable_thinking": False}}


def _build_json_messages(prompt, prior_votes, categories):
    messages = [
        {"role": "system", "content": (
            "You are a customer service classifier. Respond with ONLY a JSON object.\n"
            '{"answer": "<category>", "confidence": <0.0-1.0>}\n'
            f"Valid categories: {', '.join(categories)}\n"
            "No other text."
        )},
        {"role": "user", "content": (
            f"Message: {prompt}\n"
            + (f"\nPrevious round votes: {json.dumps(prior_votes)}\n" if prior_votes else "")
            + "Classify:"
        )}
    ]
    return messages, 30, 0.3, {"chat_template_kwargs": {"enable_thinking": False}}


def _build_cot_messages(prompt, prior_votes, categories):
    messages = [
        {"role": "system", "content": (
            "You are a customer service classifier. Think carefully step by step.\n"
            "Consider what the customer is really asking for. Analyze the tone, "
            "specific words, and underlying intent before classifying.\n"
            f"Valid categories: {', '.join(categories)}\n"
            "Format your response as:\nReasoning: <your detailed analysis>\nAnswer: <category>"
        )},
        {"role": "user", "content": (
            f"Message: {prompt}\n"
            + (f"\nPrevious round votes: {json.dumps(prior_votes)}\n" if prior_votes else "")
            + "\nThink step by step about the customer's intent, then classify."
        )}
    ]
    return messages, 8000, 0.5, {}  # thinking ENABLED for CoT


def _build_nl_debate_messages(prompt, prior_votes, categories):
    messages = [
        {"role": "system", "content": (
            "You are a customer service expert participating in a group classification exercise.\n"
            "Think deeply and critically about the customer's intent. Consider edge cases, "
            "ambiguity, and the subtle differences between categories.\n"
            "If other agents disagree with you, carefully analyze their reasoning and either "
            "defend your position with new evidence or acknowledge their stronger argument.\n"
            f"Valid categories: {', '.join(categories)}\n"
            "End your response with: Final answer: <category>"
        )},
        {"role": "user", "content": (
            f"Customer message: {prompt}\n"
            + (f"\nOther agents' arguments from previous round:\n{json.dumps(prior_votes, indent=2)}\n" if prior_votes else "")
            + "\nProvide your thorough analysis and final classification."
        )}
    ]
    return messages, 10000, 0.7, {}  # thinking ENABLED for NL debate


PATTERN_BUILDERS = {
    "select": _build_select_messages,
    "json": _build_json_messages,
    "cot_select": _build_cot_messages,
    "nl_debate": _build_nl_debate_messages,
}


def _extract_vote(content: str, pattern: str, categories: list[str]) -> str:
    """Extract the classification vote from LLM output, pattern-aware."""
    content_clean = strip_thinking(content)

    if pattern == "select":
        c = content_clean.strip().lower().strip('"').strip("'").strip(".")
        for cat in categories:
            if cat in c:
                return cat
        return c.split()[0] if c else categories[0]

    if pattern == "json":
        try:
            parsed = json.loads(content_clean)
            vote = parsed.get("answer", "").lower()
        except json.JSONDecodeError:
            fb, lb = content_clean.find('{'), content_clean.rfind('}')
            if fb != -1 and lb > fb:
                try:
                    parsed = json.loads(content_clean[fb:lb+1])
                    vote = parsed.get("answer", "").lower()
                except Exception:
                    vote = content_clean.strip().lower()
            else:
                vote = content_clean.strip().lower()
        if vote not in categories:
            for cat in categories:
                if cat in vote:
                    return cat
            return categories[0]
        return vote

    # cot_select and nl_debate — look for "Answer: <category>" or "Final answer: <category>"
    answer_match = re.search(
        r'(?:final\s+answer|answer|classification|category)\s*[:\-]\s*["\']?(\w+)',
        content_clean, re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).lower()
        if candidate in categories:
            return candidate

    for cat in categories:
        if cat in content_clean.lower()[-100:]:
            return cat

    return categories[0]


# ---------------------------------------------------------------------------
# Async consensus runner (unified schema)
# ---------------------------------------------------------------------------

async def run_consensus_task(
    pattern: str,
    base_url: str,
    model: str,
    n_agents: int,
    max_rounds: int,
    prompt: str,
) -> DecisionLog:
    """Run one consensus task fully async. Returns a DecisionLog."""

    builder = PATTERN_BUILDERS[pattern]
    agent_names = [f"agent_{i}" for i in range(n_agents)]

    def _build_prior_votes(prior_hops: list[HopLog], pattern: str):
        """Build prior votes context from previous round's hops."""
        if not prior_hops:
            return None
        if pattern == "select":
            return [f"{h.agent}: {h.parsed_vote}" for h in prior_hops]
        elif pattern == "json":
            return [{"agent": h.agent, "answer": h.parsed_vote} for h in prior_hops]
        else:
            return [{"agent": h.agent, "vote": h.parsed_vote} for h in prior_hops]

    def round_agent_fn_factory(rnd, agent_name, all_hops):
        """Create an async callable for one agent in one round."""
        # Get prior round's hops
        if rnd == 0:
            prior_hops = []
        else:
            prior_hops = [h for h in all_hops if h.round == rnd - 1]

        prior_votes = _build_prior_votes(prior_hops, pattern)
        messages, max_tokens, temperature, extra_body = builder(prompt, prior_votes, CATEGORIES)

        # Estimate input tokens
        input_chars = sum(len(m["content"]) for m in messages)
        input_tokens_est = input_chars // 4

        async def agent_call():
            content, out_tokens, in_tokens = await sglang_chat(
                base_url, model, messages,
                max_tokens=max_tokens, temperature=temperature,
                extra_body=extra_body,
            )
            vote = _extract_vote(content, pattern, CATEGORIES)
            return {
                "response_text": content,
                "input_tokens": in_tokens or input_tokens_est,
                "output_tokens": out_tokens,
                "parsed_vote": vote,
                "context_length": input_tokens_est,
            }

        return agent_call

    def convergence_check(round_hops: list[HopLog]) -> bool:
        votes = [h.parsed_vote for h in round_hops]
        return len(set(votes)) == 1

    decision = await execute_decision(
        experiment="patterns",
        pattern=pattern,
        model=model,
        prompt=prompt,
        agent_names=agent_names,
        max_rounds=max_rounds,
        round_agent_fn_factory=round_agent_fn_factory,
        convergence_check=convergence_check,
        energy_mode="round",
    )

    return decision


# ---------------------------------------------------------------------------
# Energy normalization: overlapping per-task Zeus windows → true share
# ---------------------------------------------------------------------------

def normalize_batch_energy(decisions: list[DecisionLog], batch_total_j: float):
    """Normalize overlapping per-decision Zeus readings against a batch total.

    Each decision already has raw_energy_j from its own (overlapping) Zeus
    window. We scale so that sum(normalized) == batch_total_j, preserving
    the relative differences between tasks.

        normalized_i = (raw_i / sum(raw_all)) * batch_total_j
    """
    raw_sum = sum(d.raw_energy_j for d in decisions)
    for dec in decisions:
        if raw_sum > 0:
            dec.total_energy_j = round(dec.raw_energy_j / raw_sum * batch_total_j, 4)
        else:
            dec.total_energy_j = round(batch_total_j / len(decisions), 4)
        dec.compute_derived()


# ---------------------------------------------------------------------------
# Test blocks
# ---------------------------------------------------------------------------

async def run_test_headline(base_url, model, n_agents, n_tasks, max_rounds, cooldown=10) -> RunLog:
    """Test 1: All 4 patterns under load — n_tasks fire concurrently per pattern.

    All tasks run simultaneously so the GPU is saturated. One Zeus window
    per pattern batch; energy attributed proportionally by latency.
    """
    print(f"\n{'#'*70}")
    print(f"  TEST 1: HEADLINE COMPARISON (UNDER LOAD)")
    print(f"  {n_agents} agents, {n_tasks} concurrent tasks per pattern, max {max_rounds} rounds")
    print(f"{'#'*70}")

    run = RunLog(experiment="patterns", model=model, config={
        "test": "headline", "n_agents": n_agents, "n_tasks": n_tasks, "max_rounds": max_rounds})

    for pattern in ["select", "json", "cot_select", "nl_debate"]:
        print(f"\n  === Pattern: {pattern} ({n_tasks} concurrent tasks) ===")

        window = f"headline_{pattern}_{int(time.time())}"
        zeus_begin(window)
        t0 = time.perf_counter()

        # Fire ALL tasks concurrently — GPU under full load
        # Each decision gets its own overlapping Zeus window inside execute_decision
        coros = [
            run_consensus_task(pattern, base_url, model, n_agents, max_rounds,
                               PROMPTS[i % len(PROMPTS)])
            for i in range(n_tasks)
        ]
        print(f"    Launched {n_tasks} tasks...", flush=True)
        decs = await _gather_with_progress(coros, pattern)

        batch_elapsed = time.perf_counter() - t0
        batch_energy = zeus_end(window)
        batch_j = batch_energy.get("gpu_energy_j", 0)

        # Normalize: scale overlapping per-task readings so sum == batch total
        normalize_batch_energy(list(decs), batch_j)
        for dec in decs:
            run.decisions.append(dec)

        # Print per-task results
        for i, dec in enumerate(decs):
            converge_str = "YES" if dec.converged else f"NO({dec.final_answer})"
            print(f"    [{i+1:>2}/{n_tasks}] {converge_str:>6} "
                  f"rounds={dec.n_rounds} "
                  f"tokens={dec.total_output_tokens:>5} "
                  f"E={dec.total_energy_j:>7.1f}J "
                  f"E/hop={dec.energy_per_hop_j:>5.1f}J "
                  f"lat={dec.total_latency_s:.1f}s")

        # Per-pattern summary
        n = len(decs)
        conv = sum(1 for d in decs if d.converged)
        avg_e = sum(d.total_energy_j for d in decs) / n
        avg_e_hop = sum(d.energy_per_hop_j for d in decs) / n
        avg_e_dec = sum(d.energy_per_decision_j for d in decs) / n
        avg_rnd = sum(d.n_rounds for d in decs) / n
        total_out = sum(d.total_output_tokens for d in decs)
        tok_per_s = total_out / batch_elapsed if batch_elapsed > 0 else 0

        print(f"\n  {pattern:>12}: converge={conv}/{n} "
              f"avg_E={avg_e:.0f}J avg_E/hop={avg_e_hop:.1f}J "
              f"avg_E/decision={avg_e_dec:.0f}J avg_rounds={avg_rnd:.1f}")
        print(f"  {'':>12}  batch: {batch_j:.0f}J total, {batch_elapsed:.1f}s wall, "
              f"{tok_per_s:.0f} out_tok/s")

        if pattern != "nl_debate":
            print(f"  Cooling down {cooldown}s...")
            await asyncio.sleep(cooldown)

    # Comparison table
    run.compute_derived()
    _print_headline_table(run)
    return run


def _print_headline_table(run: RunLog):
    """Print the headline comparison table from a RunLog."""
    print(f"\n{'='*80}")
    print(f"  HEADLINE RESULTS")
    print(f"{'='*80}")
    print(f"  {'pattern':>12} {'conv%':>6} {'avg_E':>7} {'E/hop':>7} {'E/dec':>7} "
          f"{'avg_tok':>8} {'avg_rnd':>8} {'J/out_tok':>9}")
    print(f"  {'─'*75}")

    for p in ["select", "json", "cot_select", "nl_debate"]:
        decs = [d for d in run.decisions if d.pattern == p]
        if not decs:
            continue
        n = len(decs)
        conv = sum(1 for d in decs if d.converged) / n * 100
        avg_e = sum(d.total_energy_j for d in decs) / n
        avg_e_hop = sum(d.energy_per_hop_j for d in decs) / n
        avg_e_dec = sum(d.energy_per_decision_j for d in decs) / n
        avg_tok = sum(d.total_output_tokens for d in decs) / n
        avg_rnd = sum(d.n_rounds for d in decs) / n
        j_per_out = sum(d.j_per_output_token for d in decs) / n

        print(f"  {p:>12} {conv:>5.0f}% {avg_e:>7.0f} {avg_e_hop:>7.1f} {avg_e_dec:>7.0f} "
              f"{avg_tok:>8.0f} {avg_rnd:>8.1f} {j_per_out:>9.4f}")


async def run_test_scaling(base_url, model, n_tasks, max_rounds, cooldown=10) -> RunLog:
    """Test 2: Agent count scaling under load — Pattern 1 vs 4.

    All n_tasks fire concurrently for each (pattern, n_agents) config.
    Measures how consensus cost scales with agent count when the GPU is busy.
    """
    print(f"\n{'#'*70}")
    print(f"  TEST 2: AGENT COUNT SCALING (UNDER LOAD)")
    print(f"  {n_tasks} concurrent tasks per config")
    print(f"{'#'*70}")

    run = RunLog(experiment="patterns", model=model, config={
        "test": "scaling", "n_tasks": n_tasks, "max_rounds": max_rounds})

    for n_agents in [2, 3, 5, 7]:
        for pattern in ["select", "nl_debate"]:
            print(f"\n  === {pattern}, {n_agents} agents, {n_tasks} concurrent tasks ===")

            window = f"scale_{pattern}_{n_agents}a_{int(time.time())}"
            zeus_begin(window)
            t0 = time.perf_counter()

            coros = [
                run_consensus_task(pattern, base_url, model, n_agents, max_rounds,
                                   PROMPTS[i % len(PROMPTS)])
                for i in range(n_tasks)
            ]
            print(f"    Launched {n_tasks} tasks...", flush=True)
            decs = await _gather_with_progress(coros, f"{pattern}/{n_agents}a")

            batch_elapsed = time.perf_counter() - t0
            batch_energy = zeus_end(window)
            batch_j = batch_energy.get("gpu_energy_j", 0)

            # Normalize overlapping per-task readings against batch total
            normalize_batch_energy(list(decs), batch_j)
            for dec in decs:
                run.decisions.append(dec)

            n = len(decs)
            conv = sum(1 for d in decs if d.converged)
            avg_e = sum(d.total_energy_j for d in decs) / n
            avg_e_hop = sum(d.energy_per_hop_j for d in decs) / n
            avg_rnd = sum(d.n_rounds for d in decs) / n
            total_out = sum(d.total_output_tokens for d in decs)
            tok_per_s = total_out / batch_elapsed if batch_elapsed > 0 else 0

            print(f"    converge={conv}/{n} avg_E={avg_e:.0f}J avg_E/hop={avg_e_hop:.1f}J "
                  f"avg_rounds={avg_rnd:.1f}")
            print(f"    batch: {batch_j:.0f}J, {batch_elapsed:.1f}s wall, {tok_per_s:.0f} out_tok/s")
            await asyncio.sleep(cooldown)

    # Print scaling summary table
    run.compute_derived()
    print(f"\n{'='*80}")
    print(f"  SCALING RESULTS")
    print(f"{'='*80}")
    print(f"  {'pattern':>12} {'agents':>6} {'conv%':>6} {'avg_E':>7} {'E/hop':>7} "
          f"{'E/dec':>7} {'avg_rnd':>7} {'J/out_tok':>9}")
    print(f"  {'─'*70}")
    for n_agents in [2, 3, 5, 7]:
        for pattern in ["select", "nl_debate"]:
            subset = [d for d in run.decisions if d.pattern == pattern and d.n_agents == n_agents]
            if not subset:
                continue
            n = len(subset)
            conv = sum(1 for d in subset if d.converged) / n * 100
            avg_e = sum(d.total_energy_j for d in subset) / n
            avg_e_hop = sum(d.energy_per_hop_j for d in subset) / n
            avg_e_dec = sum(d.energy_per_decision_j for d in subset) / n
            avg_rnd = sum(d.n_rounds for d in subset) / n
            j_out = sum(d.j_per_output_token for d in subset) / n
            print(f"  {pattern:>12} {n_agents:>6} {conv:>5.0f}% {avg_e:>7.0f} {avg_e_hop:>7.1f} "
                  f"{avg_e_dec:>7.0f} {avg_rnd:>7.1f} {j_out:>9.4f}")

    return run


async def run_test_concurrency(base_url, model, n_agents, n_tasks, max_rounds, cooldown=10) -> RunLog:
    """Test 3: Concurrency scaling — how J/decision changes with load.

    Sweeps batch sizes from 1 (serial baseline) up to 64 (should saturate
    or overwhelm a B200). Each batch level fires batch_size concurrent
    consensus tasks, repeated until n_tasks are done. This is the only test
    that intentionally varies utilization to find the efficiency sweet spot.
    """
    BATCH_SIZES = [1, 8, 16, 32, 64]

    print(f"\n{'#'*70}")
    print(f"  TEST 3: CONCURRENCY SCALING (UTILIZATION vs EFFICIENCY)")
    print(f"  {n_agents} agents, {n_tasks} tasks per level, batch_sizes={BATCH_SIZES}")
    print(f"{'#'*70}")

    run = RunLog(experiment="patterns", model=model, config={
        "test": "concurrency", "n_agents": n_agents, "n_tasks": n_tasks,
        "max_rounds": max_rounds, "batch_sizes": BATCH_SIZES})

    # Track results per (pattern, batch_size) for the summary table
    summary_rows = []

    for batch_size in BATCH_SIZES:
        for pattern in ["select", "nl_debate"]:
            print(f"\n  === {pattern}, batch_size={batch_size} "
                  f"({batch_size * n_agents} concurrent LLM calls/round) ===")

            window = f"conc_{pattern}_{batch_size}_{int(time.time())}"
            zeus_begin(window)
            t0 = time.perf_counter()

            # Process n_tasks in waves of batch_size
            all_decs = []
            prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_tasks)]
            n_waves = (n_tasks + batch_size - 1) // batch_size

            for wave_idx, wave_start in enumerate(range(0, n_tasks, batch_size)):
                wave_prompts = prompts[wave_start:wave_start + batch_size]
                coros = [
                    run_consensus_task(pattern, base_url, model, n_agents, max_rounds, p)
                    for p in wave_prompts
                ]
                print(f"    Wave {wave_idx+1}/{n_waves}: {len(coros)} tasks...", flush=True)
                wave_decs = await _gather_with_progress(
                    coros, f"{pattern}/b{batch_size}/w{wave_idx+1}")
                all_decs.extend(wave_decs)

            elapsed = time.perf_counter() - t0
            energy = zeus_end(window)
            batch_j = energy.get("gpu_energy_j", 0)

            # Normalize overlapping per-task readings against batch total
            normalize_batch_energy(all_decs, batch_j)
            for dec in all_decs:
                dec.concurrency = batch_size
                run.decisions.append(dec)

            n = len(all_decs)
            conv = sum(1 for d in all_decs if d.converged)
            avg_e = sum(d.total_energy_j for d in all_decs) / n
            avg_e_hop = sum(d.energy_per_hop_j for d in all_decs) / n
            total_out = sum(d.total_output_tokens for d in all_decs)
            tok_per_s = total_out / elapsed if elapsed > 0 else 0
            j_per_out = batch_j / total_out if total_out > 0 else 0
            avg_watts = batch_j / elapsed if elapsed > 0 else 0
            dec_per_s = n / elapsed if elapsed > 0 else 0
            dec_per_j = n / batch_j if batch_j > 0 else 0

            print(f"    {conv}/{n} converged, {elapsed:.1f}s wall")
            print(f"    {batch_j:.0f}J total, {avg_watts:.0f}W avg, {tok_per_s:.0f} out_tok/s")
            print(f"    avg_E/dec={avg_e:.0f}J, avg_E/hop={avg_e_hop:.1f}J, J/out_tok={j_per_out:.4f}")
            print(f"    {dec_per_s:.2f} decisions/s, {dec_per_j:.4f} decisions/J")

            summary_rows.append({
                "pattern": pattern, "batch": batch_size,
                "conv_pct": conv / n * 100 if n else 0,
                "wall_s": elapsed, "total_j": batch_j, "watts": avg_watts,
                "tok_s": tok_per_s, "j_per_out": j_per_out,
                "avg_e_dec": avg_e, "avg_e_hop": avg_e_hop,
                "dec_per_s": dec_per_s, "dec_per_j": dec_per_j,
            })

            await asyncio.sleep(cooldown)

    # Summary table
    run.compute_derived()
    print(f"\n{'='*100}")
    print(f"  CONCURRENCY SCALING RESULTS")
    print(f"{'='*100}")
    print(f"  {'pattern':>12} {'batch':>5} {'conv%':>6} {'wall':>6} {'J':>7} {'W':>5} "
          f"{'tok/s':>7} {'J/out':>7} {'E/dec':>7} {'E/hop':>6} {'dec/s':>6} {'dec/J':>7}")
    print(f"  {'─'*95}")
    for r in summary_rows:
        print(f"  {r['pattern']:>12} {r['batch']:>5} {r['conv_pct']:>5.0f}% "
              f"{r['wall_s']:>5.0f}s {r['total_j']:>7.0f} {r['watts']:>5.0f} "
              f"{r['tok_s']:>7.0f} {r['j_per_out']:>7.4f} "
              f"{r['avg_e_dec']:>7.0f} {r['avg_e_hop']:>6.1f} "
              f"{r['dec_per_s']:>6.2f} {r['dec_per_j']:>7.4f}")

    return run


# ---------------------------------------------------------------------------
# CSV export (unified schema)
# ---------------------------------------------------------------------------

def export_csv(run: RunLog, run_dir):
    """Export all decisions + hops to CSVs."""
    # Decision-level CSV
    dec_rows = []
    for d in run.decisions:
        dec_rows.append({
            "decision_id": d.decision_id, "experiment": d.experiment,
            "pattern": d.pattern, "model": d.model, "n_agents": d.n_agents,
            "n_rounds": d.n_rounds, "converged": d.converged,
            "final_answer": d.final_answer,
            "total_input_tokens": d.total_input_tokens,
            "total_output_tokens": d.total_output_tokens,
            "total_latency_s": d.total_latency_s,
            "total_energy_j": d.total_energy_j,
            "energy_per_hop_j": d.energy_per_hop_j,
            "energy_per_decision_j": d.energy_per_decision_j,
            "j_per_output_token": d.j_per_output_token,
            "prompt_text": d.prompt_text,
        })

    if dec_rows:
        save_csv_rows(dec_rows, list(dec_rows[0].keys()), run_dir, "decisions.csv")

    # Hop-level CSV
    hop_rows = []
    for d in run.decisions:
        for h in d.hops:
            hop_rows.append({
                "decision_id": h.decision_id, "hop_id": h.hop_id,
                "experiment": h.experiment, "pattern": h.pattern,
                "round": h.round, "agent": h.agent,
                "input_tokens": h.input_tokens, "output_tokens": h.output_tokens,
                "context_length": h.context_length,
                "latency_s": h.latency_s, "energy_j": h.energy_j,
                "energy_mode": h.energy_mode,
                "j_per_output_token": h.j_per_output_token,
                "avg_watts": h.avg_watts,
                "parsed_vote": h.parsed_vote,
            })

    if hop_rows:
        save_csv_rows(hop_rows, list(hop_rows[0].keys()), run_dir, "hops.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_run(run: RunLog, label: str, args):
    """Save a single test run immediately (JSON + CSV)."""
    tag = f"{label}_{args.n_agents}agents_{args.n_tasks}tasks"
    run_dir = make_run_dir("patterns", tag)
    save_json(run.to_dict(), run_dir)
    export_csv(run, run_dir)
    print(f"\n  >> Saved {label}: {run.total_decisions} decisions, "
          f"{run.total_energy_j:.1f}J → {run_dir}", flush=True)
    return run_dir


async def async_main(args):
    all_runs = []

    if args.test in ("headline", "all"):
        run = await run_test_headline(
            args.base_url, args.model, args.n_agents, args.n_tasks, args.max_rounds, args.cooldown)
        _save_run(run, "headline", args)
        all_runs.append(run)

    if args.test in ("scaling", "all"):
        run = await run_test_scaling(
            args.base_url, args.model, args.n_tasks, args.max_rounds, args.cooldown)
        _save_run(run, "scaling", args)
        all_runs.append(run)

    if args.test in ("concurrency", "all"):
        run = await run_test_concurrency(
            args.base_url, args.model, args.n_agents, args.n_tasks, args.max_rounds, args.cooldown)
        _save_run(run, "concurrency", args)
        all_runs.append(run)

    # Also save a merged run if we ran multiple tests
    if len(all_runs) > 1:
        merged = RunLog(experiment="patterns", model=args.model,
                        config={"test": args.test, "n_agents": args.n_agents,
                                "n_tasks": args.n_tasks, "max_rounds": args.max_rounds})
        for r in all_runs:
            merged.decisions.extend(r.decisions)
        merged.compute_derived()
        _save_run(merged, "all_merged", args)
    elif len(all_runs) == 1:
        merged = all_runs[0]
    else:
        return

    # Print aggregate energy metrics
    print(f"\n{'='*70}")
    print(f"  AGGREGATE METRICS")
    print(f"{'='*70}")
    print(f"  Total decisions:         {merged.total_decisions}")
    print(f"  Total hops:              {merged.total_hops}")
    print(f"  Total energy:            {merged.total_energy_j:.1f} J")
    print(f"  Energy per hop:          {merged.energy_per_hop_j:.2f} J")
    print(f"  Energy per decision:     {merged.energy_per_decision_j:.1f} J")
    print(f"  J per output token:      {merged.j_per_output_token:.4f}")

    await close_client()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent consensus energy benchmarking")
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--test", choices=["headline", "scaling", "concurrency", "all"], default="all")
    args = parser.parse_args()

    init_zeus()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
