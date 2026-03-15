"""
ARC-AGI-2 Consensus Benchmark

Measures: accuracy improvement × energy cost across 4 communication patterns.

Core question: Does multi-agent consensus improve ARC-AGI-2 accuracy,
and is the energy cost of NL debate justified vs structured voting?

Each task:
  - Agents see the ARC training examples (input→output demonstrations)
  - Agents predict the test output grid
  - Consensus protocol: compare predictions, debate, re-predict
  - Measure: did consensus converge? Was the answer correct? Energy cost?
"""

import json
import re
import os
import time
import uuid
import csv
import glob
import random
import argparse
import concurrent.futures
from dataclasses import dataclass, asdict
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
# ARC grid helpers
# ---------------------------------------------------------------------------

def grid_to_str(grid):
    """Convert a grid (list of lists) to a readable string."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def str_to_grid(s):
    """Parse a grid string back to list of lists. Robust to various formats."""
    s = s.strip()
    # Remove markdown code blocks
    s = re.sub(r'```[a-z]*\n?', '', s).strip()
    # Remove <think>...</think>
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()

    lines = []
    for line in s.split("\n"):
        line = line.strip().strip("|").strip()
        if not line:
            continue
        # Extract numbers from the line
        nums = re.findall(r'\d+', line)
        if nums:
            lines.append([int(n) for n in nums])

    return lines if lines else None


def grids_equal(g1, g2):
    """Check if two grids are identical."""
    if g1 is None or g2 is None:
        return False
    if len(g1) != len(g2):
        return False
    for r1, r2 in zip(g1, g2):
        if len(r1) != len(r2):
            return False
        if r1 != r2:
            return False
    return True


def grid_to_compact(grid):
    """Compact grid representation for voting comparison."""
    if grid is None:
        return "NONE"
    return "|".join(",".join(str(c) for c in row) for row in grid)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_arc_tasks(data_dir, n_tasks=None, split="evaluation"):
    """Load ARC-AGI-2 tasks from directory."""
    task_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(task_dir, "*.json")))

    if n_tasks:
        files = files[:n_tasks]

    tasks = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        task_id = os.path.basename(f).replace(".json", "")
        tasks.append({"id": task_id, "train": data["train"], "test": data["test"]})

    return tasks


def format_arc_prompt(task, test_idx=0):
    """Format an ARC task as a text prompt with demonstrations."""
    parts = ["Here is a pattern transformation puzzle. Study the examples, then predict the output for the test input.\n"]

    for i, example in enumerate(task["train"]):
        parts.append(f"Example {i+1}:")
        parts.append(f"Input:\n{grid_to_str(example['input'])}")
        parts.append(f"Output:\n{grid_to_str(example['output'])}")
        parts.append("")

    parts.append("Now predict the output for this test input:")
    parts.append(f"Input:\n{grid_to_str(task['test'][test_idx]['input'])}")
    parts.append("\nOutput the grid as rows of space-separated numbers. Nothing else.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SGLang API
# ---------------------------------------------------------------------------

def sglang_chat(base_url, model, messages, max_tokens=500, temperature=0.7, extra_body=None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)

    resp = http_requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage.get("completion_tokens", len(content) // 4), usage.get("prompt_tokens", 0)


# ---------------------------------------------------------------------------
# Pattern implementations for ARC
# ---------------------------------------------------------------------------

def arc_pattern_select(base_url, model, agent_id, task_prompt, prior_answers, ground_truth_shape):
    """Pattern 1: Quick grid prediction — short, no thinking."""
    messages = [
        {"role": "system", "content": "You are solving an ARC-AGI puzzle. Output ONLY the predicted grid as rows of space-separated numbers. No explanation."},
        {"role": "user", "content": task_prompt
         + (f"\n\nOther agents' predictions from previous round:\n" +
            "\n---\n".join(f"Agent {a[0]}:\n{a[1]}" for a in prior_answers)
            if prior_answers else "")}
    ]

    content, out_tok, in_tok = sglang_chat(
        base_url, model, messages,
        max_tokens=200, temperature=0.3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    grid = str_to_grid(content)
    return grid, grid_to_compact(grid), out_tok, in_tok


def arc_pattern_json(base_url, model, agent_id, task_prompt, prior_answers, ground_truth_shape):
    """Pattern 2: JSON response with grid + confidence."""
    messages = [
        {"role": "system", "content": (
            "You are solving an ARC-AGI puzzle. Respond with ONLY a JSON object:\n"
            '{"grid": [[row1], [row2], ...], "confidence": 0.0-1.0, "pattern_description": "brief"}\n'
            "No other text."
        )},
        {"role": "user", "content": task_prompt
         + (f"\n\nOther agents' predictions:\n{json.dumps([{'agent': a[0], 'grid': a[1]} for a in prior_answers])}"
            if prior_answers else "")}
    ]

    content, out_tok, in_tok = sglang_chat(
        base_url, model, messages,
        max_tokens=500, temperature=0.3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    grid = None
    try:
        parsed = json.loads(content)
        grid = parsed.get("grid")
    except:
        fb, lb = content.find('{'), content.rfind('}')
        if fb != -1 and lb > fb:
            try:
                parsed = json.loads(content[fb:lb+1])
                grid = parsed.get("grid")
            except:
                pass
    if grid is None:
        grid = str_to_grid(content)

    return grid, grid_to_compact(grid), out_tok, in_tok


def arc_pattern_cot(base_url, model, agent_id, task_prompt, prior_answers, ground_truth_shape):
    """Pattern 3: Chain of thought + grid prediction. Thinking enabled."""
    messages = [
        {"role": "system", "content": (
            "You are an expert ARC-AGI puzzle solver. Analyze the pattern carefully.\n"
            "Think about: symmetry, rotation, color mapping, scaling, tiling, borders, fills.\n"
            "After your analysis, output the predicted grid as rows of space-separated numbers.\n"
            "Format: first your analysis, then 'ANSWER:' followed by the grid."
        )},
        {"role": "user", "content": task_prompt
         + (f"\n\nOther agents' analyses and predictions from previous round:\n" +
            "\n---\n".join(f"Agent {a[0]}: {a[1]}" for a in prior_answers)
            if prior_answers else "")}
    ]

    content, out_tok, in_tok = sglang_chat(
        base_url, model, messages,
        max_tokens=8000, temperature=0.5,
        extra_body={}  # thinking ON
    )

    # Extract grid after "ANSWER:" or from end of response
    content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    answer_match = re.search(r'ANSWER\s*:?\s*\n?(.*)', content_clean, re.DOTALL | re.IGNORECASE)
    if answer_match:
        grid = str_to_grid(answer_match.group(1))
    else:
        # Try last few lines
        grid = str_to_grid("\n".join(content_clean.split("\n")[-15:]))

    return grid, grid_to_compact(grid), out_tok, in_tok


def arc_pattern_debate(base_url, model, agent_id, task_prompt, prior_answers, ground_truth_shape):
    """Pattern 4: Full NL debate with deep reasoning. Thinking enabled."""
    messages = [
        {"role": "system", "content": (
            "You are an expert ARC-AGI puzzle solver in a group debate.\n"
            "Carefully analyze the transformation pattern. Consider multiple hypotheses:\n"
            "- What changes between input and output?\n"
            "- Are there symmetries, rotations, reflections?\n"
            "- Color substitutions or mappings?\n"
            "- Size changes, tiling, cropping?\n"
            "- Object detection, movement, interaction?\n\n"
            "If other agents disagree with you, critically evaluate their reasoning.\n"
            "Defend your analysis or acknowledge better reasoning.\n"
            "End with 'ANSWER:' followed by your predicted grid (rows of space-separated numbers)."
        )},
        {"role": "user", "content": task_prompt
         + (f"\n\nOther agents' detailed analyses from previous round:\n" +
            "\n===\n".join(f"Agent {a[0]}:\n{a[1]}" for a in prior_answers)
            if prior_answers else "")}
    ]

    content, out_tok, in_tok = sglang_chat(
        base_url, model, messages,
        max_tokens=10000, temperature=0.7,
        extra_body={}  # thinking ON
    )

    content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    answer_match = re.search(r'ANSWER\s*:?\s*\n?(.*)', content_clean, re.DOTALL | re.IGNORECASE)
    if answer_match:
        grid = str_to_grid(answer_match.group(1))
    else:
        grid = str_to_grid("\n".join(content_clean.split("\n")[-15:]))

    return grid, grid_to_compact(grid), out_tok, in_tok


ARC_PATTERNS = {
    "select": arc_pattern_select,
    "json": arc_pattern_json,
    "cot": arc_pattern_cot,
    "debate": arc_pattern_debate,
}


# ---------------------------------------------------------------------------
# Consensus runner for ARC
# ---------------------------------------------------------------------------

@dataclass
class ArcTaskResult:
    task_id: str
    pattern: str
    n_agents: int
    n_rounds: int
    converged: bool
    correct: bool
    total_tokens: int
    gpu_energy_j: float
    latency_s: float
    agent_answers: list  # per-round votes
    ground_truth: str


def run_arc_consensus(
    pattern, base_url, model, n_agents, max_rounds, task, test_idx=0
):
    """Run consensus on a single ARC task."""
    task_prompt = format_arc_prompt(task, test_idx)
    ground_truth = task["test"][test_idx]["output"]
    gt_compact = grid_to_compact(ground_truth)
    gt_shape = (len(ground_truth), len(ground_truth[0]) if ground_truth else 0)

    pattern_fn = ARC_PATTERNS[pattern]
    all_answers = []  # list of lists of (agent_id, compact_answer, full_content)
    total_tokens = 0

    window = f"arc_{task['id']}_{pattern}_{int(time.time()*1000)}"
    zeus_begin(window)
    t0 = time.perf_counter()

    for rnd in range(max_rounds):
        prior = all_answers[-1] if all_answers else None

        round_answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_agents) as pool:
            futs = {}
            for agent_id in range(n_agents):
                futs[pool.submit(pattern_fn, base_url, model, agent_id,
                                task_prompt, prior, gt_shape)] = agent_id

            for f in concurrent.futures.as_completed(futs):
                agent_id = futs[f]
                grid, compact, out_tok, in_tok = f.result()
                round_answers.append((agent_id, compact, grid_to_str(grid) if grid else "FAILED"))
                total_tokens += out_tok

        all_answers.append(round_answers)

        # Check consensus: do all agents agree?
        compacts = [a[1] for a in round_answers]
        if len(set(compacts)) == 1:
            break  # Unanimous

    elapsed = time.perf_counter() - t0
    energy = zeus_end(window)

    # Final answer = majority vote from last round
    last_compacts = [a[1] for a in all_answers[-1]]
    majority = max(set(last_compacts), key=last_compacts.count)
    converged = len(set(last_compacts)) == 1
    correct = majority == gt_compact

    return ArcTaskResult(
        task_id=task["id"],
        pattern=pattern,
        n_agents=n_agents,
        n_rounds=len(all_answers),
        converged=converged,
        correct=correct,
        total_tokens=total_tokens,
        gpu_energy_j=energy.get("gpu_energy_j", 0),
        latency_s=round(elapsed, 2),
        agent_answers=[[a[1] for a in rnd] for rnd in all_answers],
        ground_truth=gt_compact,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    tasks = load_arc_tasks(args.data_dir, args.n_tasks, args.split)
    print(f"Loaded {len(tasks)} ARC-AGI-2 tasks from {args.split}")

    all_results = {}

    for pattern in args.patterns:
        print(f"\n{'='*60}")
        print(f"  PATTERN: {pattern} | {len(tasks)} tasks | {args.n_agents} agents | max {args.max_rounds} rounds")
        print(f"{'='*60}")

        results = []
        for i, task in enumerate(tasks):
            r = run_arc_consensus(
                pattern, args.base_url, args.model,
                args.n_agents, args.max_rounds, task
            )
            results.append(r)

            status = "CORRECT" if r.correct else ("CONVERGED" if r.converged else "NO-CONV")
            print(f"  [{i+1:>3}/{len(tasks)}] {task['id']} {status:>9} "
                  f"rounds={r.n_rounds} tok={r.total_tokens:>6} "
                  f"E={r.gpu_energy_j:>8.0f}J lat={r.latency_s:.1f}s")

        # Summary
        correct = sum(1 for r in results if r.correct)
        converged = sum(1 for r in results if r.converged)
        total_e = sum(r.gpu_energy_j for r in results)
        total_tok = sum(r.total_tokens for r in results)
        avg_e = total_e / len(results) if results else 0
        avg_tok = total_tok / len(results) if results else 0

        all_results[pattern] = {
            "results": [asdict(r) for r in results],
            "n_tasks": len(results),
            "accuracy": correct / len(results) if results else 0,
            "convergence_rate": converged / len(results) if results else 0,
            "total_energy_j": round(total_e, 1),
            "avg_energy_j": round(avg_e, 1),
            "avg_tokens": round(avg_tok, 0),
            "total_tokens": total_tok,
            "n_correct": correct,
            "n_converged": converged,
        }

        print(f"\n  {pattern}: accuracy={correct}/{len(results)} ({correct/len(results)*100:.1f}%) "
              f"converge={converged}/{len(results)} "
              f"avg_E={avg_e:.0f}J avg_tok={avg_tok:.0f}")

        # Cooldown
        if pattern != args.patterns[-1]:
            print(f"  Cooling down {args.cooldown}s...")
            time.sleep(args.cooldown)

    # Comparison
    print(f"\n{'='*60}")
    print(f"  ARC-AGI-2 CONSENSUS BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  {'pattern':>8} {'acc%':>6} {'conv%':>6} {'avg_J':>8} {'avg_tok':>8} "
          f"{'J/tok':>7} {'J/correct':>10}")
    print(f"  {'─'*60}")

    for p in args.patterns:
        r = all_results[p]
        j_per_tok = r["avg_energy_j"] / r["avg_tokens"] if r["avg_tokens"] > 0 else 0
        j_per_correct = r["total_energy_j"] / r["n_correct"] if r["n_correct"] > 0 else float('inf')
        print(f"  {p:>8} {r['accuracy']*100:>5.1f}% {r['convergence_rate']*100:>5.0f}% "
              f"{r['avg_energy_j']:>8.0f} {r['avg_tokens']:>8.0f} "
              f"{j_per_tok:>7.2f} {j_per_correct:>10.0f}")

    # Save
    ts = int(time.time())
    outfile = f"results_arc_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {outfile}")

    # CSV
    csvfile = f"results_arc_{ts}.csv"
    rows = []
    for p, data in all_results.items():
        for task in data["results"]:
            task["pattern"] = p
            rows.append(task)
    if rows:
        with open(csvfile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV exported: {csvfile} ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI-2 consensus energy benchmark")
    parser.add_argument("--data-dir", default="arc-agi-2/data")
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--patterns", nargs="+", default=["select", "json", "cot", "debate"])
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--cooldown", type=int, default=15)
    args = parser.parse_args()

    init_zeus()
    run_benchmark(args)
