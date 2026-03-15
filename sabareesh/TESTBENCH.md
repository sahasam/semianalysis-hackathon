# Test Bench Guide

How to run the energy benchmarking experiments on the Slurm GPU cluster.

## Prerequisites

```bash
# SSH into the cluster
ssh user25@35.84.33.219

# Activate environment
source ~/miniforge3/bin/activate
conda activate bench
export HF_HOME=~/hf-cache
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1

# Install Zeus (one-time)
pip install zeus-ml

# Code lives here
cd ~/sabareesh
```

## Quick Start

Submit any experiment as a Slurm batch job:

```bash
sbatch run-patterns.sh     # 4-pattern comparison (the headline experiment)
sbatch run-stress.sh       # Panel scaling sweep
sbatch run-efficiency.sh   # KV cache x concurrency sweep
sbatch run-profile.sh      # Per-hop energy profiling

# Monitor
squeue --me                        # Check job status
tail -f logs/patterns-NNN.out      # Watch output live
cat logs/patterns-NNN.err          # Check for errors
scancel NNN                        # Cancel a job
```

## Experiments

### 1. Pattern Comparison (`consensus_patterns.py`)

Compares 4 multi-agent communication patterns for consensus energy cost.

```bash
# Full suite: headline + scaling + concurrency tests
sbatch run-patterns.sh

# Or run manually with a running SGLang server:
python consensus_patterns.py --test headline --n-agents 3 --n-tasks 20
python consensus_patterns.py --test scaling --n-tasks 10
python consensus_patterns.py --test concurrency --n-tasks 12
python consensus_patterns.py --test all
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--test` | `all` | Which test: `headline`, `scaling`, `concurrency`, `all` |
| `--n-agents` | 3 | Number of agents per consensus task |
| `--n-tasks` | 20 | Tasks per pattern (headline) or per config |
| `--max-rounds` | 5 | Maximum consensus rounds before giving up |
| `--cooldown` | 10 | Seconds between pattern switches (thermal stability) |
| `--base-url` | `http://localhost:25000` | SGLang server URL |
| `--model` | `Qwen/Qwen3.5-27B` | Model name |

**What it measures per task:**
- `pattern` — select / json / cot_select / nl_debate
- `total_tokens_generated` — total output tokens
- `gpu_energy_j` — GPU energy via Zeus (joules)
- `n_rounds_to_consensus` — rounds to reach agreement
- `converged` — did all agents agree?
- Per-round: `round_tokens`, `round_gpu_energy_j`, `votes`, `agreement_fraction`

**Output:** `results_patterns_<timestamp>.json` + `.csv`

**The 4 patterns:**

| Pattern | Thinking | Max Tokens | What agents produce |
|---------|----------|-----------|-------------------|
| `select` | OFF | 5 | Single category word |
| `json` | OFF | 30 | `{"answer": "billing", "confidence": 0.85}` |
| `cot_select` | ON | 8,000 | Full reasoning chain + answer |
| `nl_debate` | ON | 10,000 | Deep argument with thinking |

### 2. Panel Scaling (`stress_test.py`)

Scales concurrent consensus panels to find GPU saturation point.

```bash
sbatch run-stress.sh

# Or manually:
python stress_test.py --num-panels 7 --num-agents 5 --max-rounds 8
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-panels` | 3 | Concurrent consensus panels |
| `--num-agents` | 5 | Agents per panel |
| `--max-rounds` | 10 | Max rounds per panel |
| `--temperature` | 0.8 | LLM temperature |
| `--max-tokens` | 400 | Max tokens per agent response |

**Output:** `results_stress_<panels>p_<agents>a_<timestamp>.json`

### 3. KV Cache + Concurrency Sweep (`efficiency_sweep.py`)

Measures J/output_token across KV cache sizes and concurrency levels.

```bash
sbatch run-efficiency.sh

# Or manually (requires restarting server per cache config):
python efficiency_sweep.py --num-panels 7 --num-agents 5 --mem-fraction 0.88
python efficiency_sweep.py --summarize   # Print summary table
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-panels` | 3 | Concurrent panels |
| `--num-agents` | 5 | Agents per panel |
| `--mem-fraction` | 0.80 | SGLang `--mem-fraction-static` value |
| `--summarize` | false | Print summary of all efficiency results |

**Output:** `results_eff_m<frac>_p<panels>_<timestamp>.json`

The batch job (`run-efficiency.sh`) automatically sweeps 3 cache sizes × 6 concurrency levels = 18 data points.

### 4. Per-Hop Energy Profiling (`detailed_profile.py`)

Exact energy measurement per individual agent call.

```bash
sbatch run-profile.sh

# Or manually:
python detailed_profile.py --num-agents 5 --max-rounds 5
python detailed_profile.py --serial-only    # Only exact per-hop
python detailed_profile.py --parallel-only  # Only batched mode
```

**Two modes:**
- **Serial** — agents run one at a time. Zeus measures exact J per hop. The ground truth.
- **Parallel** — agents run concurrently. Energy attributed proportionally by latency.

**Analysis views:**
- Energy by round (shows context growth effect)
- Energy by agent persona (which expert costs most?)
- Energy vs context length correlation
- Serial vs parallel comparison

**Output:** `results_detailed_profile_<timestamp>.json`

### 5. NL Consensus with Multi-Panel (`consensus.py`)

The original living-model debate orchestration.

```bash
sbatch run-consensus.sh

# Or manually:
python consensus.py --topic "your topic here" --num-agents 7 --max-rounds 15
python consensus.py --multi-panel   # 3 concurrent panels
```

## Starting SGLang Server Manually

If you want to run experiments interactively instead of via `sbatch`:

```bash
# Start server
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-27B \
  --tp 1 \
  --port 25000 &

# Wait for it
for i in $(seq 1 60); do
  curl -s http://localhost:25000/health > /dev/null 2>&1 && break
  sleep 5
done

# Now run experiments
python consensus_patterns.py --test headline

# Kill server when done
kill %1
```

For CUDA graphs (higher throughput), remove `--disable-cuda-graph`. For larger KV cache, add `--mem-fraction-static 0.9`.

## Output Files

All results are JSON with full per-task and per-round data:

```
results_patterns_*.json     # 4-pattern comparison
results_patterns_*.csv      # Flat CSV for spreadsheets
results_stress_*.json       # Panel scaling
results_eff_*.json          # Efficiency sweep
results_detailed_*.json     # Per-hop profiling
results_consensus_*.json    # NL consensus debates
results_multipanel_*.json   # Multi-panel comparison
```

## Derived Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| energy_per_consensus | `gpu_energy_j` per task | J |
| energy_per_token | `gpu_energy_j / total_tokens` | J/tok |
| consensus_per_joule | `1 / energy_per_consensus` | 1/J |
| energy_ratio | `energy(nl_debate) / energy(this_pattern)` | × |
| convergence_rate | `count(converged) / count(all)` | % |
