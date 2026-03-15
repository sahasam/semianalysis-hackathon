"""
Unified data schema for all testbench experiments.

Every experiment records the same core structures:
  HopLog       — single agent LLM call (atomic measurement unit)
  DecisionLog  — one consensus task from first round to convergence
  RunLog       — one experiment run (collection of decisions)

This ensures consistent analysis regardless of which experiment produced the data.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


def _id() -> str:
    return uuid.uuid4().hex[:8]


def _ts() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# HopLog — single agent LLM call
# ---------------------------------------------------------------------------

@dataclass
class HopLog:
    """Atomic measurement unit: one agent, one LLM call."""

    # Identity
    hop_id: str = field(default_factory=_id)
    decision_id: str = ""
    experiment: str = ""           # patterns | consensus | profile | efficiency | experiment
    pattern: str = ""              # select | json | cot_select | nl_debate | freeform

    # Position in the run
    round: int = 0
    agent: str = ""

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_length: int = 0        # cumulative prompt size (grows each round)

    # Timing
    latency_s: float = 0.0

    # Energy (exact in serial mode, attributed in parallel mode)
    energy_j: float = 0.0
    energy_mode: str = ""          # "exact" | "attributed" | "none"
    j_per_output_token: float = 0.0
    j_per_total_token: float = 0.0
    avg_watts: float = 0.0

    # Agent output (truncated for storage)
    response_text: str = ""
    parsed_vote: str = ""          # for classification patterns
    confidence: float = 0.0        # for consensus patterns
    agreement: float = 0.0         # for consensus patterns
    key_point: str = ""

    timestamp: str = field(default_factory=_ts)

    def compute_derived(self):
        """Fill in derived energy-per-token metrics."""
        if self.energy_j > 0:
            if self.output_tokens > 0:
                self.j_per_output_token = round(self.energy_j / self.output_tokens, 6)
            if self.total_tokens > 0:
                self.j_per_total_token = round(self.energy_j / self.total_tokens, 6)
            if self.latency_s > 0:
                self.avg_watts = round(self.energy_j / self.latency_s, 1)
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# DecisionLog — one consensus task (first round → convergence)
# ---------------------------------------------------------------------------

@dataclass
class DecisionLog:
    """One consensus decision: all rounds from start to convergence/max."""

    # Identity
    decision_id: str = field(default_factory=_id)
    experiment: str = ""
    pattern: str = ""
    model: str = ""

    # Configuration
    n_agents: int = 0
    max_rounds: int = 0
    concurrency: int = 0          # 0 = unbounded async

    # Outcome
    n_rounds: int = 0
    converged: bool = False
    final_answer: str = ""
    prompt_text: str = ""

    # Aggregated token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Aggregated timing
    total_latency_s: float = 0.0

    # Aggregated energy
    raw_energy_j: float = 0.0            # overlapping Zeus reading (before normalization)
    total_energy_j: float = 0.0          # normalized energy (actual share of GPU)
    energy_per_hop_j: float = 0.0
    energy_per_decision_j: float = 0.0   # = total_energy_j (alias for clarity)
    j_per_output_token: float = 0.0
    j_per_total_token: float = 0.0

    # All hops in this decision
    hops: list[HopLog] = field(default_factory=list)

    timestamp: str = field(default_factory=_ts)

    def compute_derived(self):
        """Aggregate metrics from hops."""
        if self.hops:
            self.total_input_tokens = sum(h.input_tokens for h in self.hops)
            self.total_output_tokens = sum(h.output_tokens for h in self.hops)
            self.total_tokens = self.total_input_tokens + self.total_output_tokens
            self.n_rounds = max(h.round for h in self.hops) + 1

            hop_energy = sum(h.energy_j for h in self.hops)
            # Use the measured total if set, otherwise sum hops
            if self.total_energy_j == 0:
                self.total_energy_j = hop_energy

            self.energy_per_decision_j = self.total_energy_j
            n_hops = len(self.hops)
            self.energy_per_hop_j = round(self.total_energy_j / n_hops, 4) if n_hops > 0 else 0

            if self.total_output_tokens > 0:
                self.j_per_output_token = round(self.total_energy_j / self.total_output_tokens, 6)
            if self.total_tokens > 0:
                self.j_per_total_token = round(self.total_energy_j / self.total_tokens, 6)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hops"] = [h.to_dict() for h in self.hops]
        return d


# ---------------------------------------------------------------------------
# RunLog — one experiment run
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    """One complete experiment run: collection of decisions."""

    # Identity
    run_id: str = field(default_factory=_id)
    experiment: str = ""
    model: str = ""
    timestamp: str = field(default_factory=_ts)

    # Configuration (arbitrary dict for experiment-specific params)
    config: dict = field(default_factory=dict)

    # Decisions
    decisions: list[DecisionLog] = field(default_factory=list)

    # Aggregated metrics
    total_energy_j: float = 0.0
    total_time_s: float = 0.0
    total_hops: int = 0
    total_decisions: int = 0

    # Derived
    energy_per_hop_j: float = 0.0
    energy_per_decision_j: float = 0.0
    j_per_output_token: float = 0.0

    def compute_derived(self):
        """Aggregate from decisions."""
        if self.decisions:
            for d in self.decisions:
                d.compute_derived()

            self.total_decisions = len(self.decisions)
            self.total_hops = sum(len(d.hops) for d in self.decisions)

            dec_energy = sum(d.total_energy_j for d in self.decisions)
            if self.total_energy_j == 0:
                self.total_energy_j = dec_energy

            if self.total_hops > 0:
                self.energy_per_hop_j = round(self.total_energy_j / self.total_hops, 4)
            if self.total_decisions > 0:
                self.energy_per_decision_j = round(self.total_energy_j / self.total_decisions, 4)

            total_out = sum(d.total_output_tokens for d in self.decisions)
            if total_out > 0:
                self.j_per_output_token = round(self.total_energy_j / total_out, 6)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decisions"] = [dec.to_dict() for dec in self.decisions]
        return d
