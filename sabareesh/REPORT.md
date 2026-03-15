# Energy-Efficient Multi-Agent Orchestration on GPU Inference Infrastructure

## Team: Sabareesh | Hackathon 2026

---

## 1. Problem Statement

How do you efficiently run communication-heavy multi-agent LLM workloads on GPU inference infrastructure? Specifically:

- **What is the energy cost** of agent-to-agent communication patterns?
- **How does concurrency** (number of simultaneous agents) affect GPU utilization and energy efficiency?
- **What orchestration patterns** maximize throughput while minimizing joules-per-request?

## 2. Architecture

### Stack
- **Model**: Qwen3.5-27B (hybrid Mamba+MoE architecture, Feb 2026)
- **Inference Server**: SGLang (with RadixAttention KV cache, CUDA graphs)
- **Orchestration**: LangGraph (directed agent graphs with state management)
- **Energy Measurement**: Zeus (`zeus-ml`) — per-node GPU energy in joules
- **Benchmarking**: AIPerf (latency, throughput, TTFT profiling)
- **Hardware**: NVIDIA B200 GPU (183 GB VRAM), Slurm-managed cluster

### Agent Pattern: Consensus

We implemented a **round-based consensus protocol** — the most communication-heavy multi-agent pattern:

```
Round 1: All N agents generate initial positions (parallel fan-out)
Round 2: Each agent reads ALL other positions, updates their own (parallel)
Round 3: Repeat until agreement score >= threshold
Final:   Aggregator summarizes consensus
```

Why consensus is ideal for benchmarking:
- **Parallel fan-out** per round — tests batching efficiency
- **Shared system prompt** across all agents — tests RadixAttention KV cache reuse
- **Growing context** each round (positions accumulate) — tests memory pressure
- **Multiple rounds** — sustained, bursty GPU load with idle gaps between rounds
- **Structured JSON output** — tests constrained decoding

### Agent Personas (7 experts for living model discussion)
1. Systems Architect — distributed state and fault tolerance
2. ML Researcher — continual learning and catastrophic forgetting
3. Infrastructure Engineer — GPU scheduling and memory management
4. Knowledge Graph Specialist — structured knowledge and RAG
5. Security & Governance Lead — auditing, versioning, compliance
6. Product Architect — user-facing AI and personalization
7. Neuroscience Researcher — bio-inspired memory architectures

## 3. Experiments & Results

### Experiment 1: Baseline Energy Measurement (Teammate's work)

Simple 2-agent pipeline (drafter → reviewer) on Qwen2.5-7B-Instruct.

| Metric | Value |
|--------|-------|
| GPU | NVIDIA B200 |
| Model | Qwen2.5-7B-Instruct |
| Drafter hop | 5.01s, 2,012 J |
| Reviewer hop | 4.61s, 1,946 J |
| Total pipeline | 9.63s, 3,958 J |
| Orchestration overhead | ~0 J |

**Finding**: LangGraph orchestration adds negligible energy overhead — virtually all energy is spent on LLM inference.

### Experiment 2: Consensus — Single Panel (Qwen3.5-27B)

7 agents debating "living model" architecture, consensus threshold 0.92.

| Metric | Value |
|--------|-------|
| Model | Qwen3.5-27B |
| Agents | 7 |
| Rounds to consensus | 3 |
| Total LLM requests | 22 |
| Total wall time | 48.0s |
| Total GPU energy | 20,318 J |
| **Effective req/s** | **0.46** |
| **Joules per request** | **924 J** |
| Joules per round | 5,818 J |

**Server-side observations** (from SGLang logs):
- Decode throughput: ~184 tok/s with 7 concurrent requests
- KV cache utilization: **only 2%** — massively underutilized
- GPU is idle between rounds while Python processes results

### Experiment 3: Multi-Panel — 3 Concurrent Panels (21 agents)

3 independent consensus panels running simultaneously, each with 7 agents debating different aspects of living model design.

| Metric | Single Panel | Multi-Panel (3x) | Change |
|--------|-------------|-------------------|--------|
| Total requests | 22 | **66** | 3x |
| Wall time | 48.0s | **48.8s** | +1.7% |
| Effective req/s | 0.46 | **1.35** | **+193%** |
| Total energy | 20,318 J | **24,242 J** | +19% |
| **J per request** | **924 J** | **367 J** | **-60%** |

**Key finding**: 3x the work in essentially the same wall time, consuming only 19% more energy. Energy per request dropped by 60%. The GPU was severely underutilized with a single panel.

### Experiment 4: Panel Scaling Sweep (CUDA graphs enabled)

Scaling from 1 to 14 concurrent panels (5 to 70 simultaneous agents) with CUDA graphs enabled to find the GPU saturation point.

| Panels | Max Concurrent | Total Reqs | Wall Time | req/s | J/req | Avg Watts |
|--------|---------------|------------|-----------|-------|-------|-----------|
| 1 | 5 | 15 | 14.8s | 1.02 | 615 | 625 W |
| 3 | 15 | 40 | 15.3s | 2.61 | 275 | 718 W |
| 5 | 25 | 90 | 26.1s | 3.45 | 220 | 757 W |
| 7 | 35 | 120 | 29.6s | 4.06 | 192 | 779 W |
| 10 | 50 | 150 | 33.1s | 4.53 | 175 | 792 W |
| **14** | **70** | **235** | **48.7s** | **4.83** | **167** | **808 W** |

**Scaling from 1→14 panels: 4.7x throughput, 3.7x more energy efficient, GPU power draw only +29%.**

Key observations:
- Throughput (req/s) shows **sub-linear scaling** — still climbing at 14 panels, not yet saturated
- Energy efficiency (J/req) follows a **power-law decay** — biggest gains are 1→3 panels
- GPU power draw increases modestly (625W→808W) while doing 4.7x more useful work
- Round latency increases from ~5s (1 panel) to ~12s (14 panels) but wall time per panel stays manageable
- All 14 panels (70 concurrent agents) reached consensus successfully

## 4. Key Findings

### 4.1 GPU Utilization Gap

The most striking finding: **a single-panel consensus workload uses only 2% of available KV cache** on a B200 with 183 GB VRAM. The GPU is fundamentally underutilized by sequential agent orchestration patterns.

This matters because:
- You're paying for 183 GB of VRAM but using 3 GB
- The GPU idles between rounds (bursty workload)
- Energy is wasted on idle GPU power draw

### 4.2 Concurrency is the Primary Efficiency Lever

Moving from 1 to 3 concurrent panels reduced energy-per-request by **60%** with minimal impact on latency. This is because:
- SGLang batches concurrent requests efficiently
- Shared system prompts enable KV cache reuse
- Overlapping rounds from different panels fill GPU idle gaps

### 4.3 Full Efficiency Sweep — J/output_token across KV Cache Sizes and Concurrency

We swept 3 KV cache allocations × 6 concurrency levels = 18 data points:

| KV Cache (`mem_fraction`) | Concurrent | Output tok/s | **J/output_token** | Watts |
|---------------------------|-----------|-------------|-------------------|-------|
| 0.80 | 5 | 381 | 1.744 | 664 W |
| 0.80 | 15 | 919 | 0.780 | 717 W |
| 0.80 | 35 | 1,378 | 0.553 | 763 W |
| 0.80 | 70 | 1,964 | 0.414 | 814 W |
| **0.80** | **100** | **2,275** | **0.369** | **840 W** |
| 0.80 | 150 | 2,082 | 0.380 | 791 W |
| 0.88 | 5 | 409 | 1.683 | 688 W |
| 0.88 | 15 | 852 | 0.826 | 704 W |
| 0.88 | 35 | 1,331 | 0.555 | 738 W |
| 0.88 | 70 | 1,868 | 0.422 | 788 W |
| 0.88 | 100 | 2,140 | 0.378 | 808 W |
| 0.88 | 150 | 2,028 | 0.374 | 758 W |
| 0.93 | 5 | 401 | 1.720 | 690 W |
| 0.93 | 15 | 939 | 0.760 | 714 W |
| 0.93 | 35 | 1,330 | 0.557 | 742 W |
| 0.93 | 70 | 1,754 | 0.449 | 787 W |
| 0.93 | 100 | 2,163 | 0.380 | 822 W |
| **0.93** | **150** | **2,155** | **0.363** | **782 W** |

**Peak efficiency: 0.363 J/output_token** at `mem_fraction=0.93`, 150 concurrent agents.

Key observations:
- **5x efficiency gain** from 5→100 concurrent (1.74 → 0.37 J/tok)
- **Throughput saturates at ~2,200 output tok/s** regardless of cache config
- **KV cache size barely matters until 150 concurrent** — all 3 configs converge at ~0.37 J/tok at 100 concurrent
- **Sweet spot: 100 concurrent (20 panels × 5 agents)** — best throughput (2,275 tok/s) with near-optimal efficiency
- **Beyond 100 concurrent, throughput drops slightly** — GPU becomes compute-bound, not memory-bound
- GPU power increases only 664→840W (+26%) while doing 6x more useful work

### 4.4 Orchestration Overhead is Negligible

LangGraph's orchestration overhead is <1% of total energy. The entire energy budget is spent on LLM inference. Optimization efforts should focus on:
- Maximizing GPU utilization via concurrency
- Reducing unnecessary token generation (shorter prompts, constrained output)
- Leveraging KV cache reuse through prompt engineering

### 4.5 Qwen3.5-27B on B200: Practical Performance

| Metric | Value |
|--------|-------|
| Single-request decode | ~46 tok/s (CUDA graphs on) |
| 7-concurrent decode | ~184 tok/s |
| 100-concurrent output | ~2,275 tok/s |
| Prefill throughput | ~3,000 tok/s |
| Model memory | ~55 GB (bf16) |
| Available KV cache | 100-125 GB (depending on config) |
| Peak efficiency | 0.363 J/output_token at 150 concurrent |
| Efficiency sweet spot | 0.369 J/output_token at 100 concurrent |

## 5. What the Consensus Agents Actually Decided

**Topic**: Best architecture for a living, non-stateless AI model

**Consensus reached in 3 rounds** (avg agreement 0.95, avg confidence 0.95):

> A three-tier hybrid architecture comprising:
> 1. **Immutable frozen core weights** — never modified at serving time
> 2. **Versioned external memory** (Knowledge Graph + vector store) — append-only, cryptographically signed
> 3. **Hot-swappable PEFT adapters** (LoRA) — versioned, signed, rollback-able
>
> Key design principles:
> - Memory consolidation via offline "hippocampal replay" distillation (bio-inspired)
> - Raft-based consensus for distributed memory updates
> - All memory transitions pass through staged validation with human-in-the-loop
> - "Forgetting" via deterministic node deprecation, not weight modification

## 6. Code & Artifacts

| File | Description |
|------|-------------|
| `consensus.py` | Full orchestration layer with Zeus energy monitoring, multi-panel support |
| `stress_test.py` | Panel scaling sweep for GPU saturation testing |
| `run-consensus.sh` | Slurm job: single + multi-panel consensus runs |
| `run-stress.sh` | Slurm job: 1-14 panel scaling sweep |
| `experiment.py` | Teammate's original 2-agent energy measurement |
| `results_consensus_*.json` | Structured results from all consensus experiments |
| `results_multipanel_*.json` | Multi-panel comparison results |
| `results_stress_*.json` | Scaling sweep results |

## 7. Next Steps

1. **Complete panel scaling sweep** — find the exact saturation point where J/req stops improving
2. **SGLang server tuning** — `--mem-fraction-static`, `--chunked-prefill-size`, `--schedule-policy`
3. **Additional communication patterns** — fan-out/fan-in, debate, chain-of-experts
4. **Per-agent temperature control** — skeptic gets high temp, pragmatist gets low temp
5. **Cross-model comparison** — same workload on Qwen3.5-9B vs 27B: energy per token of intelligence

---

*Generated during Semianalysis Hackathon, March 2026*
*Infrastructure: NVIDIA B200 GPU cluster, SGLang + LangGraph + Zeus*
