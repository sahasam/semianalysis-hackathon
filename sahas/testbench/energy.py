"""
Shared Zeus energy monitoring — single source of truth for all experiments.

Usage:
    from testbench.energy import init_zeus, zeus_begin, zeus_end

    init_zeus()
    zeus_begin("my_window")
    # ... do work ...
    result = zeus_end("my_window")  # {"time_s": ..., "gpu_energy_j": ...}
"""

_zeus_available = False
_monitor = None


def init_zeus(gpu_index=0):
    """Initialize Zeus energy monitoring on the given GPU. Graceful fallback."""
    global _zeus_available, _monitor
    try:
        import torch
        from zeus.monitor import ZeusMonitor
        gpu_idx = gpu_index if gpu_index >= 0 else torch.cuda.current_device()
        _monitor = ZeusMonitor(gpu_indices=[gpu_idx])
        _zeus_available = True
        print(f"Zeus energy monitoring active on GPU {gpu_idx}")
    except Exception as e:
        print(f"Zeus not available ({e}), running without energy measurement")
        _zeus_available = False


def zeus_begin(window_name: str):
    """Start an energy measurement window."""
    if _zeus_available and _monitor:
        _monitor.begin_window(window_name)


def zeus_end(window_name: str) -> dict:
    """End an energy measurement window. Returns dict with time_s and gpu_energy_j, or {}."""
    if _zeus_available and _monitor:
        m = _monitor.end_window(window_name)
        return {"time_s": round(m.time, 4), "gpu_energy_j": round(m.total_energy, 4)}
    return {}


def is_available() -> bool:
    """Check if Zeus monitoring is active."""
    return _zeus_available
