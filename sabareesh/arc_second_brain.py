"""
ARC-AGI-2: Second Brain Approach

Instead of consensus (multiple agents converging on same answer),
agents collaborate in a pipeline — each adding a different cognitive
capability to solve the puzzle:

Pipeline:
  1. Observer   — describes what changes between input→output examples
  2. Theorist   — formalizes the observation into precise transformation rules
  3. Solver     — applies the rules to the test input, produces grid
  4. Verifier   — checks the solution against training examples, fixes errors

We measure the energy cost of each hop and the accuracy at each stage:
  - 1 brain (solver only)
  - 2 brains (observer → solver)
  - 3 brains (observer → theorist → solver)
  - 4 brains (observer → theorist → solver → verifier)

Core question: what's the marginal energy cost and accuracy gain of
each additional brain in the pipeline?
"""

import json
import re
import os
import glob
import time
import argparse
from dataclasses import dataclass, asdict

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
# ARC helpers
# ---------------------------------------------------------------------------

def grid_to_str(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)

def str_to_grid(s):
    s = s.strip()
    s = re.sub(r'```[a-z]*\n?', '', s).strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
    lines = []
    for line in s.split("\n"):
        line = line.strip().strip("|").strip()
        if not line:
            continue
        nums = re.findall(r'\d+', line)
        if nums:
            lines.append([int(n) for n in nums])
    return lines if lines else None

def grids_equal(g1, g2):
    if g1 is None or g2 is None:
        return False
    if len(g1) != len(g2):
        return False
    return all(r1 == r2 for r1, r2 in zip(g1, g2))

def grid_to_compact(grid):
    if grid is None:
        return "NONE"
    return "|".join(",".join(str(c) for c in row) for row in grid)

def load_arc_tasks(data_dir, n_tasks=None, split="evaluation"):
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

def format_examples(task):
    parts = []
    for i, ex in enumerate(task["train"]):
        parts.append(f"Example {i+1}:")
        parts.append(f"Input:\n{grid_to_str(ex['input'])}")
        parts.append(f"Output:\n{grid_to_str(ex['output'])}")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SGLang API
# ---------------------------------------------------------------------------

def llm_call(base_url, model, messages, max_tokens=4000, temperature=0.5, thinking=True):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    resp = http_requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    out_tok = usage.get("completion_tokens", len(content) // 4)
    in_tok = usage.get("prompt_tokens", 0)

    # Strip thinking for downstream use
    clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return clean, out_tok, in_tok


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_observer(base_url, model, task, test_idx=0):
    """Stage 1: Observe and describe the transformation pattern."""
    examples = format_examples(task)
    messages = [
        {"role": "system", "content": (
            "You are an expert pattern analyst for ARC-AGI puzzles.\n"
            "Study the input→output examples carefully.\n"
            "Describe EXACTLY what transformation is applied:\n"
            "- What stays the same? What changes?\n"
            "- Are there color mappings, translations, rotations, reflections?\n"
            "- Are objects detected, moved, scaled, or merged?\n"
            "- What is the relationship between input size and output size?\n"
            "Be precise and specific. Reference exact grid positions and colors."
        )},
        {"role": "user", "content": f"{examples}\nDescribe the transformation pattern in detail."}
    ]
    return llm_call(base_url, model, messages, max_tokens=4000, temperature=0.3, thinking=True)


def stage_theorist(base_url, model, task, observation, test_idx=0):
    """Stage 2: Formalize observation into precise rules."""
    examples = format_examples(task)
    messages = [
        {"role": "system", "content": (
            "You are a rule formalizer for ARC-AGI puzzles.\n"
            "Given an observation of the transformation pattern, write PRECISE step-by-step rules.\n"
            "Rules should be algorithmic — specific enough that a programmer could implement them.\n"
            "Use grid coordinates, color values, and explicit conditions.\n"
            "Format as numbered steps."
        )},
        {"role": "user", "content": (
            f"{examples}\n\n"
            f"Pattern observation from the analyst:\n{observation}\n\n"
            "Now formalize this into precise, step-by-step transformation rules."
        )}
    ]
    return llm_call(base_url, model, messages, max_tokens=4000, temperature=0.3, thinking=True)


def stage_solver(base_url, model, task, context="", test_idx=0):
    """Stage 3: Apply rules to test input, produce grid."""
    examples = format_examples(task)
    test_input = grid_to_str(task["test"][test_idx]["input"])

    system = "You are an ARC-AGI puzzle solver. Predict the output grid for the test input."
    if context:
        system += f"\n\nYou have been given analysis from other experts to help you:\n{context}"
    system += "\n\nOutput ONLY the predicted grid as rows of space-separated numbers. End with 'ANSWER:' then the grid."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{examples}\nTest input:\n{test_input}\n\nPredict the output grid."}
    ]

    content, out_tok, in_tok = llm_call(base_url, model, messages, max_tokens=4000, temperature=0.3, thinking=True)

    # Extract grid
    answer_match = re.search(r'ANSWER\s*:?\s*\n?(.*)', content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        grid = str_to_grid(answer_match.group(1))
    else:
        grid = str_to_grid("\n".join(content.split("\n")[-15:]))

    return grid, out_tok, in_tok, content


def stage_verifier(base_url, model, task, proposed_grid, solver_reasoning, test_idx=0):
    """Stage 4: Verify solution against training examples, fix if needed."""
    examples = format_examples(task)
    test_input = grid_to_str(task["test"][test_idx]["input"])
    proposed_str = grid_to_str(proposed_grid) if proposed_grid else "FAILED TO PRODUCE GRID"

    messages = [
        {"role": "system", "content": (
            "You are a verification expert for ARC-AGI puzzles.\n"
            "Check if the proposed solution is consistent with the training examples.\n"
            "Apply the same transformation rule to each training input and see if you get the training output.\n"
            "If the proposed solution has errors, fix them.\n"
            "Output the CORRECTED grid (or the same grid if correct).\n"
            "End with 'ANSWER:' then the final grid as rows of space-separated numbers."
        )},
        {"role": "user", "content": (
            f"{examples}\n\nTest input:\n{test_input}\n\n"
            f"Proposed solution:\n{proposed_str}\n\n"
            f"Solver's reasoning:\n{solver_reasoning[:500]}\n\n"
            "Verify this solution. Is it consistent with the training examples? If not, correct it."
        )}
    ]

    content, out_tok, in_tok = llm_call(base_url, model, messages, max_tokens=4000, temperature=0.3, thinking=True)

    answer_match = re.search(r'ANSWER\s*:?\s*\n?(.*)', content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        grid = str_to_grid(answer_match.group(1))
    else:
        grid = str_to_grid("\n".join(content.split("\n")[-15:]))

    return grid, out_tok, in_tok


# ---------------------------------------------------------------------------
# Pipeline configurations
# ---------------------------------------------------------------------------

def run_pipeline(base_url, model, task, n_brains, test_idx=0):
    """Run the second-brain pipeline with N stages."""
    ground_truth = task["test"][test_idx]["output"]
    hops = []
    total_tokens = 0

    # Full pipeline window
    window = f"pipe_{task['id']}_{n_brains}b_{int(time.time()*1000)}"
    zeus_begin(window)
    t0 = time.perf_counter()

    observation = ""
    rules = ""
    grid = None
    solver_reasoning = ""

    # Stage 1: Observer (if n_brains >= 2)
    if n_brains >= 2:
        hw = f"obs_{task['id']}_{int(time.time()*1000)}"
        zeus_begin(hw)
        observation, out_tok, in_tok = stage_observer(base_url, model, task, test_idx)
        hop_e = zeus_end(hw)
        total_tokens += out_tok
        hops.append({"stage": "observer", "tokens": out_tok, "energy_j": hop_e.get("gpu_energy_j", 0),
                      "latency_s": hop_e.get("time_s", 0)})

    # Stage 2: Theorist (if n_brains >= 3)
    if n_brains >= 3:
        hw = f"theo_{task['id']}_{int(time.time()*1000)}"
        zeus_begin(hw)
        rules, out_tok, in_tok = stage_theorist(base_url, model, task, observation, test_idx)
        hop_e = zeus_end(hw)
        total_tokens += out_tok
        hops.append({"stage": "theorist", "tokens": out_tok, "energy_j": hop_e.get("gpu_energy_j", 0),
                      "latency_s": hop_e.get("time_s", 0)})

    # Stage 3: Solver (always)
    context = ""
    if observation:
        context += f"Pattern observation:\n{observation}\n\n"
    if rules:
        context += f"Transformation rules:\n{rules}\n\n"

    hw = f"solve_{task['id']}_{int(time.time()*1000)}"
    zeus_begin(hw)
    grid, out_tok, in_tok, solver_reasoning = stage_solver(base_url, model, task, context, test_idx)
    hop_e = zeus_end(hw)
    total_tokens += out_tok
    hops.append({"stage": "solver", "tokens": out_tok, "energy_j": hop_e.get("gpu_energy_j", 0),
                  "latency_s": hop_e.get("time_s", 0)})

    # Stage 4: Verifier (if n_brains >= 4)
    if n_brains >= 4 and grid is not None:
        hw = f"verify_{task['id']}_{int(time.time()*1000)}"
        zeus_begin(hw)
        grid, out_tok, in_tok = stage_verifier(base_url, model, task, grid, solver_reasoning, test_idx)
        hop_e = zeus_end(hw)
        total_tokens += out_tok
        hops.append({"stage": "verifier", "tokens": out_tok, "energy_j": hop_e.get("gpu_energy_j", 0),
                      "latency_s": hop_e.get("time_s", 0)})

    elapsed = time.perf_counter() - t0
    total_energy = zeus_end(window)

    correct = grids_equal(grid, ground_truth)

    return {
        "task_id": task["id"],
        "n_brains": n_brains,
        "correct": correct,
        "total_tokens": total_tokens,
        "gpu_energy_j": total_energy.get("gpu_energy_j", 0),
        "latency_s": round(elapsed, 2),
        "hops": hops,
        "predicted": grid_to_compact(grid),
        "ground_truth": grid_to_compact(ground_truth),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Second Brain benchmark")
    parser.add_argument("--data-dir", default="arc-agi-2/data")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--base-url", default="http://localhost:25000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--brains", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    init_zeus()
    tasks = load_arc_tasks(args.data_dir, args.n_tasks, args.split)
    print(f"Loaded {len(tasks)} ARC-AGI-2 tasks\n")

    all_results = {}

    for n_brains in args.brains:
        stages = {1: "solver only", 2: "observer→solver",
                  3: "observer→theorist→solver", 4: "observer→theorist→solver→verifier"}
        print(f"\n{'='*60}")
        print(f"  {n_brains} BRAIN(S): {stages[n_brains]}")
        print(f"{'='*60}")

        results = []
        for i, task in enumerate(tasks):
            r = run_pipeline(args.base_url, args.model, task, n_brains)
            results.append(r)

            status = "CORRECT" if r["correct"] else "WRONG"
            hop_str = " → ".join(f"{h['stage']}({h['tokens']}tok,{h['energy_j']:.0f}J)" for h in r["hops"])
            print(f"  [{i+1:>3}/{len(tasks)}] {task['id']} {status:>7} "
                  f"tok={r['total_tokens']:>5} E={r['gpu_energy_j']:>7.0f}J "
                  f"lat={r['latency_s']:.1f}s")
            print(f"    Pipeline: {hop_str}")

        # Summary
        correct = sum(1 for r in results if r["correct"])
        total_e = sum(r["gpu_energy_j"] for r in results)
        total_tok = sum(r["total_tokens"] for r in results)
        avg_e = total_e / len(results)

        all_results[n_brains] = {
            "n_brains": n_brains,
            "pipeline": stages[n_brains],
            "n_tasks": len(results),
            "n_correct": correct,
            "accuracy": correct / len(results),
            "total_energy_j": round(total_e, 1),
            "avg_energy_j": round(avg_e, 1),
            "avg_tokens": round(total_tok / len(results), 0),
            "results": results,
        }

        print(f"\n  {n_brains} brains: accuracy={correct}/{len(results)} ({correct/len(results)*100:.1f}%) "
              f"avg_E={avg_e:.0f}J avg_tok={total_tok/len(results):.0f}")

    # Comparison
    print(f"\n{'='*60}")
    print(f"  SECOND BRAIN RESULTS")
    print(f"{'='*60}")
    print(f"  {'brains':>6} {'pipeline':<35} {'acc%':>6} {'avg_J':>8} {'avg_tok':>8} {'J/correct':>10}")
    print(f"  {'─'*75}")

    for nb in args.brains:
        r = all_results[nb]
        j_per_correct = r["total_energy_j"] / r["n_correct"] if r["n_correct"] > 0 else float('inf')
        stages_map = {1: "solver only", 2: "observer→solver",
                      3: "obs→theo→solver", 4: "obs→theo→solver→verify"}
        print(f"  {nb:>6} {stages_map[nb]:<35} {r['accuracy']*100:>5.1f}% "
              f"{r['avg_energy_j']:>8.0f} {r['avg_tokens']:>8.0f} {j_per_correct:>10.0f}")

    # Save
    ts = int(time.time())
    outfile = f"results_second_brain_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")


if __name__ == "__main__":
    main()
