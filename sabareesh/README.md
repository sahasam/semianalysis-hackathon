# Multi-Agent Consensus: Energy Benchmarking on B200

## Core Thesis

**Structured communication patterns (single-token select, JSON) are 10-20× more energy efficient than natural language debate for reaching the same multi-agent consensus.**

We measure **energy-to-agreement** — how many Joules it takes for N agents to converge on the same decision — across 4 communication patterns on an NVIDIA B200 GPU serving Qwen3.5-27B via SGLang.

## Setup

### Architecture
```
                    ┌─────────────────────────────────┐
                    │     Consensus Orchestrator       │
                    │  (LangGraph / Python threads)    │
                    └──────────┬──────────────────────┘
                               │ N agents per round
                               ▼
                    ┌─────────────────────────────────┐
                    │   SGLang Inference Server        │
                    │   Qwen3.5-27B (tp=1)            │
                    │   RadixAttention + CUDA graphs   │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │   NVIDIA B200 (183 GB HBM)      │
                    │   Zeus energy monitoring (NVML)  │
                    └─────────────────────────────────┘
```

### Hardware
- **GPU**: NVIDIA B200, 183 GB HBM3e
- **Model**: Qwen3.5-27B (hybrid Mamba+MoE, Feb 2026) — 55 GB in bf16
- **Cluster**: Slurm-managed GPU nodes, shared Lustre filesystem

### Software Stack
- **SGLang** — inference server with RadixAttention KV cache, CUDA graphs
- **LangGraph** — agent orchestration (directed graphs with state)
- **Zeus** (`zeus-ml`) — GPU energy measurement via NVML (per-window joules)
- **AIPerf** — throughput/latency profiling

## Consensus Protocol

All patterns follow the same protocol. Only the **communication format** differs:

```
Round 1: Each of N agents independently answers (can't see others)
Round 2+: Each agent sees ALL previous answers, responds again
Check: If all agents agree → consensus reached, record total energy
Max: 5 rounds. If no consensus → record as "no convergence"
```

## The 4 Patterns

| # | Pattern | Tokens/Agent/Round | Thinking | Description |
|---|---------|-------------------|----------|-------------|
| 1 | **Single-Token Select** | ~1-3 | OFF | Constrained to one category word. Energy floor. |
| 2 | **JSON Consensus** | ~15-20 | OFF | `{"answer": "billing", "confidence": 0.85}` |
| 3 | **CoT + Select** | ~500-2000 | **ON** | Full reasoning chain + constrained vote |
| 4 | **NL Debate** | ~1000-5000 | **ON** | Deep argument with thinking. Up to 10K tokens. |

Patterns 3 and 4 have **Qwen3.5 thinking enabled** — the model generates a `<think>...</think>` internal reasoning chain before the visible response. This dramatically increases token consumption.

## Running Experiments

### Prerequisites
```bash
source ~/miniforge3/bin/activate && conda activate bench
export HF_HOME=~/hf-cache LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export SGLANG_DISABLE_CUDNN_CHECK=1
pip install zeus-ml langgraph langchain-openai
```

### Experiment 1: Headline Comparison (4 patterns)
```bash
sbatch run-patterns.sh
# Runs: 20 tasks × 4 patterns × 3 agents = 240+ consensus decisions
# Output: results_patterns_*.json + CSV
```

### Experiment 2: GPU Saturation (panel scaling)
```bash
sbatch run-stress.sh
# Scales 1→14 concurrent consensus panels (5→70 agents)
# Finds peak throughput and energy efficiency
```

### Experiment 3: KV Cache + Concurrency Sweep
```bash
sbatch run-efficiency.sh
# 3 cache sizes × 6 concurrency levels = 18 data points
# Measures J/output_token across the full operating range
```

### Experiment 4: Per-Hop Energy Profiling
```bash
sbatch run-profile.sh
# Serial mode: exact Zeus measurement per individual agent hop
# Parallel mode: time-attributed energy per hop
# Shows energy vs context length, energy by agent persona
```

## Key Results So Far

### Batching Efficiency
| Concurrent Agents | Output tok/s | J/output_token | vs baseline |
|------------------|-------------|----------------|-------------|
| 5 (1 panel) | 381 | 1.744 | 1.0× |
| 15 (3 panels) | 919 | 0.780 | 2.2× |
| 35 (7 panels) | 1,378 | 0.553 | 3.2× |
| 100 (20 panels) | 2,275 | 0.369 | **4.7×** |
| 150 (30 panels) | 2,155 | 0.363 | **4.8×** |

### Serial vs Parallel (exact per-hop)
- Serial: **6.9 J/output_token** (1 agent at a time, GPU mostly idle)
- Parallel: **1.8 J/output_token** (5 concurrent, batched)
- **3.9× more efficient with batching**

## Data Schema

Every consensus task produces a structured record:

```json
{
  "task_id": "a1b2c3d4",
  "pattern": "nl_debate",
  "n_agents": 3,
  "n_rounds_to_consensus": 2,
  "converged": true,
  "total_tokens_generated": 4521,
  "gpu_energy_j": 3847.2,
  "total_latency_s": 12.3,
  "rounds": [
    {
      "round_number": 1,
      "round_tokens": 2891,
      "round_gpu_energy_j": 2435.1,
      "votes": ["billing", "billing", "technical"],
      "agreement_fraction": 0.67
    }
  ]
}
```

## Files

| File | Purpose |
|------|---------|
| `consensus_patterns.py` | 4-pattern comparison harness with full data schema |
| `consensus.py` | NL debate orchestration with multi-panel support |
| `stress_test.py` | Panel scaling sweep |
| `efficiency_sweep.py` | KV cache × concurrency sweep |
| `detailed_profile.py` | Per-hop serial/parallel energy profiling |
| `REPORT.md` | Full experiment report with all results |
| `run-*.sh` | Slurm batch job scripts |

## Team
- **Sabareesh** — consensus orchestration, energy benchmarking, GPU efficiency
- **Sahas** — environment setup, initial Zeus integration, infrastructure
