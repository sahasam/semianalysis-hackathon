"""
Energy Benchmarking: Multi-Agent Consensus Patterns

Compares 4 communication patterns for multi-agent consensus:
  1. Single-Token Select  — ~1 token/agent/round (sgl.select equivalent)
  2. JSON Consensus       — ~15 tokens/agent/round (constrained JSON)
  3. CoT + Select         — ~150 tokens + 1 vote/agent/round
  4. Full NL Debate       — ~350 tokens/agent/round

Core metric: Joules to reach consensus (all agents agree)

Data schema matches Hackathonplan.md exactly.
"""

import json
import re
import time
import uuid
import csv
import argparse
import concurrent.futures
from dataclasses import dataclass, asdict, field
from datetime import datetime

import requests as http_requests


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
        print(f"Zeus active on GPU {gpu_idx}")
    except Exception as e:
        print(f"Zeus unavailable ({e})")
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
# Data schema (matches Hackathonplan.md)
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    task_id: str
    round_number: int
    round_tokens: int
    round_gpu_energy_j: float
    round_latency_s: float
    votes: list
    agreement_fraction: float

@dataclass
class TaskRecord:
    task_id: str
    timestamp: str
    pattern: str           # select | json | cot_select | nl_debate
    model: str
    n_agents: int
    n_rounds_to_consensus: int
    converged: bool
    final_answer: str
    total_tokens_generated: int
    tokens_per_agent_round: float
    total_latency_s: float
    gpu_energy_j: float
    total_energy_j: float  # gpu only on B200 (no RAPL)
    prompt_text: str
    concurrency: int
    rounds: list           # list of RoundRecord dicts


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
# SGLang API helpers
# ---------------------------------------------------------------------------

def sglang_chat(base_url, model, messages, max_tokens=500, temperature=0.7, extra_body=None):
    """Call SGLang's OpenAI-compatible chat API. Returns content and usage."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)

    resp = http_requests.post(
        f"{base_url}/v1/chat/completions",
        json=body, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage.get("completion_tokens", len(content) // 4), usage.get("prompt_tokens", 0)


# ---------------------------------------------------------------------------
# Pattern implementations
# ---------------------------------------------------------------------------

def pattern_select(base_url, model, agent_id, prompt, prior_votes, categories):
    """Pattern 1: Single-token select — constrained to one category token."""
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

    content, out_tokens, in_tokens = sglang_chat(
        base_url, model, messages,
        max_tokens=5, temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    # Extract vote
    content_lower = content.strip().lower().strip('"').strip("'").strip(".")
    vote = None
    for cat in categories:
        if cat in content_lower:
            vote = cat
            break
    if not vote:
        vote = content_lower.split()[0] if content_lower else categories[0]

    return vote, out_tokens, in_tokens


def pattern_json(base_url, model, agent_id, prompt, prior_votes, categories):
    """Pattern 2: JSON consensus — structured JSON with answer + confidence."""
    messages = [
        {"role": "system", "content": (
            "You are a customer service classifier. Respond with ONLY a JSON object.\n"
            'Format: {"answer": "<category>", "confidence": <0.0-1.0>}\n'
            f"Valid categories: {', '.join(categories)}\n"
            "No other text."
        )},
        {"role": "user", "content": (
            f"Message: {prompt}\n"
            + (f"\nPrevious round votes: {json.dumps(prior_votes)}\n" if prior_votes else "")
            + "Classify:"
        )}
    ]

    content, out_tokens, in_tokens = sglang_chat(
        base_url, model, messages,
        max_tokens=30, temperature=0.3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    # Parse
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(content)
        vote = parsed.get("answer", "").lower()
    except json.JSONDecodeError:
        fb, lb = content.find('{'), content.rfind('}')
        if fb != -1 and lb > fb:
            try:
                parsed = json.loads(content[fb:lb+1])
                vote = parsed.get("answer", "").lower()
            except:
                vote = content.strip().lower()
        else:
            vote = content.strip().lower()

    if vote not in categories:
        for cat in categories:
            if cat in vote:
                vote = cat
                break
        else:
            vote = categories[0]

    return vote, out_tokens, in_tokens


def pattern_cot_select(base_url, model, agent_id, prompt, prior_votes, categories):
    """Pattern 3: Chain-of-thought reasoning + constrained vote."""
    messages = [
        {"role": "system", "content": (
            "You are a customer service classifier. Think step by step, then give your final answer.\n"
            f"Valid categories: {', '.join(categories)}\n"
            "Format your response as:\nReasoning: <your analysis>\nAnswer: <category>"
        )},
        {"role": "user", "content": (
            f"Message: {prompt}\n"
            + (f"\nPrevious round votes: {json.dumps(prior_votes)}\n" if prior_votes else "")
            + "\nThink step by step about the customer's intent, then classify."
        )}
    ]

    content, out_tokens, in_tokens = sglang_chat(
        base_url, model, messages,
        max_tokens=250, temperature=0.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    # Extract final answer
    content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    vote = None
    # Look for "Answer: <category>"
    answer_match = re.search(r'(?:answer|classification|category)\s*[:\-]\s*(\w+)', content_clean, re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).lower()
        if candidate in categories:
            vote = candidate

    if not vote:
        for cat in categories:
            if cat in content_clean.lower().split('\n')[-1]:
                vote = cat
                break
    if not vote:
        vote = categories[0]

    return vote, out_tokens, in_tokens


def pattern_nl_debate(base_url, model, agent_id, prompt, prior_votes, categories):
    """Pattern 4: Full NL debate — paragraph-length argument + extracted vote."""
    messages = [
        {"role": "system", "content": (
            "You are a customer service expert participating in a group classification exercise.\n"
            "Argue for your classification. Consider other agents' perspectives.\n"
            f"Valid categories: {', '.join(categories)}\n"
            "End your response with: Final answer: <category>"
        )},
        {"role": "user", "content": (
            f"Customer message: {prompt}\n"
            + (f"\nOther agents' arguments from previous round:\n{json.dumps(prior_votes, indent=2)}\n" if prior_votes else "")
            + "\nProvide your detailed argument and final classification."
        )}
    ]

    content, out_tokens, in_tokens = sglang_chat(
        base_url, model, messages,
        max_tokens=500, temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    vote = None
    answer_match = re.search(r'(?:final\s+answer|classification|category)\s*[:\-]\s*["\']?(\w+)', content_clean, re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).lower()
        if candidate in categories:
            vote = candidate

    if not vote:
        for cat in categories:
            if cat in content_clean.lower()[-100:]:
                vote = cat
                break
    if not vote:
        vote = categories[0]

    return vote, out_tokens, in_tokens


PATTERN_FNS = {
    "select": pattern_select,
    "json": pattern_json,
    "cot_select": pattern_cot_select,
    "nl_debate": pattern_nl_debate,
}


# ---------------------------------------------------------------------------
# Consensus runner
# ---------------------------------------------------------------------------

def run_consensus_task(
    pattern: str,
    base_url: str,
    model: str,
    n_agents: int,
    max_rounds: int,
    prompt: str,
    concurrency: int = 1,
) -> TaskRecord:
    """Run one consensus task with full energy measurement."""

    task_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()
    pattern_fn = PATTERN_FNS[pattern]

    all_votes_by_round = []  # list of list of (agent_id, vote, out_tokens, in_tokens)
    round_records = []
    total_tokens = 0

    # Full task energy window
    task_window = f"task_{task_id}"
    zeus_begin(task_window)
    task_start = time.perf_counter()

    for rnd in range(max_rounds):
        # Build prior votes context
        if rnd == 0:
            prior_votes = None
        else:
            prev = all_votes_by_round[-1]
            if pattern == "select":
                prior_votes = [f"Agent {v[0]}: {v[1]}" for v in prev]
            elif pattern == "json":
                prior_votes = [{"agent": v[0], "answer": v[1]} for v in prev]
            else:
                prior_votes = [{"agent": v[0], "vote": v[1]} for v in prev]

        # Per-round energy window
        round_window = f"round_{task_id}_{rnd}"
        zeus_begin(round_window)
        rnd_start = time.perf_counter()

        # All agents vote in parallel
        round_votes = []
        round_tokens = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_agents) as pool:
            futs = {}
            for agent_id in range(n_agents):
                futs[pool.submit(pattern_fn, base_url, model, agent_id, prompt, prior_votes, CATEGORIES)] = agent_id

            for f in concurrent.futures.as_completed(futs):
                agent_id = futs[f]
                vote, out_tok, in_tok = f.result()
                round_votes.append((agent_id, vote, out_tok, in_tok))
                round_tokens += out_tok

        rnd_elapsed = time.perf_counter() - rnd_start
        round_energy = zeus_end(round_window)

        total_tokens += round_tokens
        all_votes_by_round.append(round_votes)

        # Check agreement
        votes_only = [v[1] for v in round_votes]
        majority = max(set(votes_only), key=votes_only.count)
        agreement = votes_only.count(majority) / len(votes_only)
        converged = len(set(votes_only)) == 1

        round_records.append(RoundRecord(
            task_id=task_id,
            round_number=rnd + 1,
            round_tokens=round_tokens,
            round_gpu_energy_j=round_energy.get("gpu_energy_j", 0),
            round_latency_s=round(rnd_elapsed, 4),
            votes=votes_only,
            agreement_fraction=round(agreement, 4),
        ))

        if converged:
            break

    task_elapsed = time.perf_counter() - task_start
    task_energy = zeus_end(task_window)

    n_rounds = len(round_records)
    final_votes = [v[1] for v in all_votes_by_round[-1]]
    final_answer = max(set(final_votes), key=final_votes.count)
    did_converge = len(set(final_votes)) == 1

    return TaskRecord(
        task_id=task_id,
        timestamp=timestamp,
        pattern=pattern,
        model=model,
        n_agents=n_agents,
        n_rounds_to_consensus=n_rounds,
        converged=did_converge,
        final_answer=final_answer,
        total_tokens_generated=total_tokens,
        tokens_per_agent_round=round(total_tokens / (n_agents * n_rounds), 1) if n_rounds > 0 else 0,
        total_latency_s=round(task_elapsed, 4),
        gpu_energy_j=task_energy.get("gpu_energy_j", 0),
        total_energy_j=task_energy.get("gpu_energy_j", 0),
        prompt_text=prompt[:200],
        concurrency=concurrency,
        rounds=[asdict(r) for r in round_records],
    )


# ---------------------------------------------------------------------------
# Test blocks
# ---------------------------------------------------------------------------

def run_test_headline(base_url, model, n_agents, n_tasks, max_rounds, cooldown=10):
    """Test 1: All 4 patterns, n_tasks each. The headline comparison."""
    print(f"\n{'#'*70}")
    print(f"  TEST 1: HEADLINE COMPARISON")
    print(f"  {n_agents} agents, {n_tasks} tasks per pattern, max {max_rounds} rounds")
    print(f"{'#'*70}")

    results = {}

    for pattern in ["select", "json", "cot_select", "nl_debate"]:
        print(f"\n  === Pattern: {pattern} ===")
        tasks = []

        for i in range(n_tasks):
            prompt = PROMPTS[i % len(PROMPTS)]
            task = run_consensus_task(pattern, base_url, model, n_agents, max_rounds, prompt)
            tasks.append(task)

            converge_str = "YES" if task.converged else f"NO({task.final_answer})"
            print(f"    [{i+1:>2}/{n_tasks}] {converge_str:>6} "
                  f"rounds={task.n_rounds_to_consensus} "
                  f"tokens={task.total_tokens_generated:>5} "
                  f"E={task.gpu_energy_j:>7.1f}J "
                  f"lat={task.total_latency_s:.1f}s")

        # Summary for this pattern
        converged = [t for t in tasks if t.converged]
        total_e = sum(t.gpu_energy_j for t in tasks)
        total_tok = sum(t.total_tokens_generated for t in tasks)
        avg_e = total_e / len(tasks) if tasks else 0
        avg_tok = total_tok / len(tasks) if tasks else 0
        avg_rounds = sum(t.n_rounds_to_consensus for t in tasks) / len(tasks) if tasks else 0

        results[pattern] = {
            "tasks": [asdict(t) for t in tasks],
            "n_tasks": len(tasks),
            "n_converged": len(converged),
            "convergence_rate": len(converged) / len(tasks) if tasks else 0,
            "avg_energy_j": round(avg_e, 1),
            "total_energy_j": round(total_e, 1),
            "avg_tokens": round(avg_tok, 0),
            "avg_rounds": round(avg_rounds, 2),
            "avg_tok_per_agent_round": round(avg_tok / (n_agents * avg_rounds), 1) if avg_rounds > 0 else 0,
        }

        print(f"\n  {pattern:>12}: converge={len(converged)}/{len(tasks)} "
              f"avg_E={avg_e:.0f}J avg_tok={avg_tok:.0f} avg_rounds={avg_rounds:.1f}")

        # Cooldown between patterns
        if pattern != "nl_debate":
            print(f"  Cooling down {cooldown}s...")
            time.sleep(cooldown)

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  HEADLINE RESULTS")
    print(f"{'='*70}")
    print(f"  {'pattern':>12} {'conv%':>6} {'avg_J':>7} {'avg_tok':>8} "
          f"{'tok/a/r':>8} {'avg_rnd':>8} {'J/tok':>7} {'ratio':>6}")
    print(f"  {'─'*65}")

    baseline_e = results.get("nl_debate", {}).get("avg_energy_j", 1)
    for p in ["select", "json", "cot_select", "nl_debate"]:
        r = results[p]
        j_per_tok = r["avg_energy_j"] / r["avg_tokens"] if r["avg_tokens"] > 0 else 0
        ratio = baseline_e / r["avg_energy_j"] if r["avg_energy_j"] > 0 else 0
        print(f"  {p:>12} {r['convergence_rate']*100:>5.0f}% {r['avg_energy_j']:>7.0f} "
              f"{r['avg_tokens']:>8.0f} {r['avg_tok_per_agent_round']:>8.1f} "
              f"{r['avg_rounds']:>8.1f} {j_per_tok:>7.2f} {ratio:>5.1f}x")

    return results


def run_test_scaling(base_url, model, n_tasks, max_rounds, cooldown=10):
    """Test 2: Agent count scaling — Pattern 1 vs 4."""
    print(f"\n{'#'*70}")
    print(f"  TEST 2: AGENT COUNT SCALING")
    print(f"{'#'*70}")

    results = {}
    for n_agents in [2, 3, 5, 7]:
        for pattern in ["select", "nl_debate"]:
            key = f"{pattern}_{n_agents}a"
            print(f"\n  === {pattern}, {n_agents} agents ===")
            tasks = []
            for i in range(n_tasks):
                t = run_consensus_task(pattern, base_url, model, n_agents, max_rounds, PROMPTS[i % len(PROMPTS)])
                tasks.append(t)
            avg_e = sum(t.gpu_energy_j for t in tasks) / len(tasks)
            avg_tok = sum(t.total_tokens_generated for t in tasks) / len(tasks)
            results[key] = {"n_agents": n_agents, "pattern": pattern,
                           "avg_energy_j": round(avg_e, 1), "avg_tokens": round(avg_tok, 0),
                           "tasks": [asdict(t) for t in tasks]}
            print(f"    avg_E={avg_e:.0f}J avg_tok={avg_tok:.0f}")
            time.sleep(cooldown)

    return results


def run_test_concurrency(base_url, model, n_agents, n_tasks, max_rounds, cooldown=10):
    """Test 4: Concurrency — multiple consensus tasks in parallel."""
    print(f"\n{'#'*70}")
    print(f"  TEST 4: CONCURRENCY SCALING")
    print(f"{'#'*70}")

    results = {}
    for conc in [1, 4, 8]:
        for pattern in ["select", "nl_debate"]:
            key = f"{pattern}_c{conc}"
            print(f"\n  === {pattern}, concurrency={conc} ===")

            window = f"conc_{pattern}_{conc}_{int(time.time())}"
            zeus_begin(window)
            t0 = time.perf_counter()

            tasks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                futs = []
                for i in range(n_tasks):
                    futs.append(pool.submit(
                        run_consensus_task, pattern, base_url, model,
                        n_agents, max_rounds, PROMPTS[i % len(PROMPTS)], conc))
                for f in concurrent.futures.as_completed(futs):
                    tasks.append(f.result())

            elapsed = time.perf_counter() - t0
            energy = zeus_end(window)

            total_e = energy.get("gpu_energy_j", 0)
            decisions = sum(1 for t in tasks if t.converged)
            decisions_per_j = decisions / total_e if total_e > 0 else 0

            results[key] = {
                "pattern": pattern, "concurrency": conc,
                "total_energy_j": round(total_e, 1),
                "total_time_s": round(elapsed, 1),
                "decisions": decisions,
                "decisions_per_joule": round(decisions_per_j, 6),
                "tasks": [asdict(t) for t in tasks],
            }
            print(f"    {decisions} decisions, {total_e:.0f}J, "
                  f"{decisions_per_j:.4f} decisions/J, {elapsed:.1f}s")
            time.sleep(cooldown)

    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(all_results, filename):
    """Export all task records to CSV."""
    rows = []
    for test_name, test_data in all_results.items():
        if isinstance(test_data, dict):
            for key, val in test_data.items():
                if isinstance(val, dict) and "tasks" in val:
                    for task in val["tasks"]:
                        task["test"] = test_name
                        task["config"] = key
                        rows.append(task)

    if not rows:
        return

    # Flatten — remove nested rounds for CSV
    flat_fields = [
        "test", "config", "task_id", "timestamp", "pattern", "model",
        "n_agents", "n_rounds_to_consensus", "converged", "final_answer",
        "total_tokens_generated", "tokens_per_agent_round",
        "total_latency_s", "gpu_energy_j", "total_energy_j",
        "prompt_text", "concurrency"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  CSV exported: {filename} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent consensus energy benchmarking")
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--test", choices=["headline", "scaling", "concurrency", "all"], default="all")
    args = parser.parse_args()

    init_zeus()

    all_results = {}

    if args.test in ("headline", "all"):
        all_results["headline"] = run_test_headline(
            args.base_url, args.model, args.n_agents, args.n_tasks, args.max_rounds, args.cooldown)

    if args.test in ("scaling", "all"):
        all_results["scaling"] = run_test_scaling(
            args.base_url, args.model, min(args.n_tasks, 10), args.max_rounds, args.cooldown)

    if args.test in ("concurrency", "all"):
        all_results["concurrency"] = run_test_concurrency(
            args.base_url, args.model, args.n_agents, min(args.n_tasks, 12), args.max_rounds, args.cooldown)

    # Save full results
    ts = int(time.time())
    with open(f"results_patterns_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results: results_patterns_{ts}.json")

    # Export CSV
    export_csv(all_results, f"results_patterns_{ts}.csv")

    print("\nDone.")
