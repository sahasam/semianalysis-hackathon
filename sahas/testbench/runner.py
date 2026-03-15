"""
Async consensus runner with integrated energy logging.

All agent LLM calls go through this runner. No thread pools — just async
coroutines. SGLang handles batching/scheduling on the GPU side.

Two call paths:
  1. Raw HTTP via httpx (for patterns with precise token control)
  2. LangChain ChatOpenAI.ainvoke (for freeform consensus)

Both paths produce HopLog records in the unified schema.
"""

import asyncio
import concurrent.futures
import json
import re
import time
import uuid
from typing import Optional, Callable, Awaitable

import urllib.error
import urllib.request

from testbench.schema import HopLog, DecisionLog
from testbench.energy import zeus_begin, zeus_end


# ---------------------------------------------------------------------------
# Async SGLang HTTP client (stdlib only)
#
# Uses a large custom ThreadPoolExecutor so hundreds of concurrent urllib
# requests each get their own thread immediately (no queuing behind the
# default executor's ~20-thread limit). SGLang handles batching on the GPU.
# ---------------------------------------------------------------------------

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=512)

async def close_client():
    """Shut down the thread pool."""
    _EXECUTOR.shutdown(wait=False)


def _sync_sglang_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    extra_body: Optional[dict],
) -> tuple[str, int, int]:
    """Synchronous SGLang call via urllib (runs inside thread pool)."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return (
        content,
        usage.get("completion_tokens", len(content) // 4),
        usage.get("prompt_tokens", 0),
    )


_MAX_RETRIES = 3

async def sglang_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 500,
    temperature: float = 0.7,
    extra_body: Optional[dict] = None,
) -> tuple[str, int, int]:
    """Async SGLang chat completion. Returns (content, output_tokens, input_tokens)."""
    loop = asyncio.get_running_loop()
    for attempt in range(_MAX_RETRIES):
        try:
            return await loop.run_in_executor(
                _EXECUTOR,
                _sync_sglang_chat, base_url, model, messages,
                max_tokens, temperature, extra_body,
            )
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/{_MAX_RETRIES}] SGLang timeout, "
                      f"retrying in {wait}s...", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"    [FAILED] SGLang call failed after {_MAX_RETRIES} attempts: {e}",
                      flush=True)
                raise


# ---------------------------------------------------------------------------
# Async LangChain wrapper
# ---------------------------------------------------------------------------

async def langchain_ainvoke(llm, messages: list[dict]) -> str:
    """Call LangChain ChatOpenAI asynchronously. Returns content string."""
    response = await llm.ainvoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# JSON response parser (shared across all patterns)
# ---------------------------------------------------------------------------

def parse_json_response(content: str) -> dict:
    """Extract a JSON object from LLM output, handling thinking blocks and markdown."""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    if "```" in content:
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if m:
            content = m.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fb, lb = content.find('{'), content.rfind('}')
    if fb != -1 and lb > fb:
        try:
            return json.loads(content[fb:lb + 1])
        except json.JSONDecodeError:
            pass

    return None


def strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks from content."""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Hop executor — wraps any async agent call with timing + energy + HopLog
# ---------------------------------------------------------------------------

async def execute_hop(
    agent_fn: Callable[..., Awaitable[dict]],
    *,
    experiment: str,
    pattern: str,
    decision_id: str,
    round_num: int,
    agent: str,
    energy_mode: str = "none",
) -> HopLog:
    """
    Execute a single agent call and return a fully-populated HopLog.

    agent_fn must be an async callable that returns a dict with at least:
        - response_text: str
        - input_tokens: int
        - output_tokens: int
    And optionally:
        - parsed_vote, confidence, agreement, key_point, context_length
    """
    hop = HopLog(
        decision_id=decision_id,
        experiment=experiment,
        pattern=pattern,
        round=round_num,
        agent=agent,
        energy_mode=energy_mode,
    )

    window_id = f"hop_{hop.hop_id}"

    if energy_mode == "exact":
        zeus_begin(window_id)

    t0 = time.perf_counter()
    result = await agent_fn()
    hop.latency_s = round(time.perf_counter() - t0, 4)

    if energy_mode == "exact":
        energy = zeus_end(window_id)
        hop.energy_j = energy.get("gpu_energy_j", 0)

    # Populate from agent result
    hop.response_text = result.get("response_text", "")[:500]
    hop.input_tokens = result.get("input_tokens", 0)
    hop.output_tokens = result.get("output_tokens", 0)
    hop.total_tokens = hop.input_tokens + hop.output_tokens
    hop.context_length = result.get("context_length", hop.input_tokens)
    hop.parsed_vote = result.get("parsed_vote", "")
    hop.confidence = result.get("confidence", 0.0)
    hop.agreement = result.get("agreement", 0.0)
    hop.key_point = result.get("key_point", "")[:200]

    hop.compute_derived()
    return hop


# ---------------------------------------------------------------------------
# Round executor — fire all agents concurrently, measure round energy
# ---------------------------------------------------------------------------

async def execute_round(
    agent_fns: list[tuple[str, Callable[..., Awaitable[dict]]]],
    *,
    experiment: str,
    pattern: str,
    decision_id: str,
    round_idx: int,
    energy_mode: str = "round",
) -> list[HopLog]:
    """
    Execute all agents in a round concurrently.

    agent_fns: list of (agent_name, async_callable) tuples.

    energy_mode controls how energy is measured:
        "exact"  — each hop gets its own Zeus window (serial execution)
        "round"  — one Zeus window for the whole round, attributed by latency
        "none"   — no energy measurement
    """
    if energy_mode == "exact":
        # Serial execution with per-hop measurement
        hops = []
        for agent_name, fn in agent_fns:
            hop = await execute_hop(
                fn,
                experiment=experiment,
                pattern=pattern,
                decision_id=decision_id,
                round_num=round_idx,
                agent=agent_name,
                energy_mode="exact",
            )
            hops.append(hop)
        return hops

    # Concurrent execution
    round_window = f"round_{decision_id}_{round}"
    if energy_mode == "round":
        zeus_begin(round_window)

    tasks = []
    for agent_name, fn in agent_fns:
        tasks.append(execute_hop(
            fn,
            experiment=experiment,
            pattern=pattern,
            decision_id=decision_id,
            round_num=round_idx,
            agent=agent_name,
            energy_mode="none",  # we attribute after
        ))

    hops = await asyncio.gather(*tasks)
    hops = list(hops)

    if energy_mode == "round":
        energy = zeus_end(round_window)
        round_j = energy.get("gpu_energy_j", 0)

        # Attribute energy proportionally by latency
        total_latency = sum(h.latency_s for h in hops)
        for h in hops:
            frac = h.latency_s / total_latency if total_latency > 0 else 1.0 / len(hops)
            h.energy_j = round(round_j * frac, 4)
            h.energy_mode = "attributed"
            h.compute_derived()

    return hops


# ---------------------------------------------------------------------------
# Decision executor — run rounds until consensus or max_rounds
# ---------------------------------------------------------------------------

async def execute_decision(
    *,
    experiment: str,
    pattern: str,
    model: str,
    prompt: str,
    agent_names: list[str],
    max_rounds: int,
    round_agent_fn_factory: Callable,
    convergence_check: Callable[[list[HopLog]], bool],
    energy_mode: str = "round",
    concurrency: int = 0,
) -> DecisionLog:
    """
    Run a full consensus decision.

    Args:
        round_agent_fn_factory: Called as factory(round, agent_name, prior_hops)
            → returns an async callable that produces the agent result dict.
        convergence_check: Called with the latest round's hops → bool.
        energy_mode: "exact" | "round" | "none"
    """
    decision = DecisionLog(
        experiment=experiment,
        pattern=pattern,
        model=model,
        n_agents=len(agent_names),
        max_rounds=max_rounds,
        concurrency=concurrency,
        prompt_text=prompt[:200],
    )

    dec_window = f"decision_{decision.decision_id}"
    zeus_begin(dec_window)
    t0 = time.perf_counter()

    all_hops: list[HopLog] = []

    for rnd in range(max_rounds):
        # Build agent callables for this round
        agent_fns = []
        for agent_name in agent_names:
            fn = round_agent_fn_factory(rnd, agent_name, all_hops)
            agent_fns.append((agent_name, fn))

        round_hops = await execute_round(
            agent_fns,
            experiment=experiment,
            pattern=pattern,
            decision_id=decision.decision_id,
            round_idx=rnd,
            energy_mode=energy_mode,
        )

        all_hops.extend(round_hops)

        # Progress: show votes/responses per round
        votes = [h.parsed_vote for h in round_hops if h.parsed_vote]
        out_tok = sum(h.output_tokens for h in round_hops)
        lat = max((h.latency_s for h in round_hops), default=0)
        vote_str = ",".join(votes) if votes else f"{len(round_hops)} hops"
        print(f"      [{decision.decision_id}] R{rnd} {pattern}: [{vote_str}] "
              f"{out_tok}tok {lat:.1f}s", flush=True)

        # Check convergence
        if convergence_check(round_hops):
            decision.converged = True
            break

    decision.total_latency_s = round(time.perf_counter() - t0, 4)
    dec_energy = zeus_end(dec_window)
    decision.raw_energy_j = dec_energy.get("gpu_energy_j", 0)
    decision.total_energy_j = decision.raw_energy_j  # caller may normalize

    decision.hops = all_hops

    # Extract final answer from last round
    if all_hops:
        last_round = max(h.round for h in all_hops)
        last_hops = [h for h in all_hops if h.round == last_round]
        votes = [h.parsed_vote for h in last_hops if h.parsed_vote]
        if votes:
            decision.final_answer = max(set(votes), key=votes.count)
        else:
            # For freeform consensus, use agreement/confidence
            summaries = [h.response_text for h in last_hops if h.response_text]
            if summaries:
                decision.final_answer = summaries[0][:200]

    decision.compute_derived()
    return decision
