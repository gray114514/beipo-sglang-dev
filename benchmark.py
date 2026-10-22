#!/usr/bin/env python3
from __future__ import annotations

"""
LLM streaming benchmark tool.

Two modes:
  sequential  – send requests in dataset order with fixed concurrency
  replay      – send requests at original timestamps (scaled) with max concurrency cap

Usage examples:
  python benchmark.py --mode sequential --concurrency 16 \
      --api-url http://localhost:8000/v1/chat/completions \
      --api-key sk-xxx --dataset dataset.jsonl

  python benchmark.py --mode replay --scale-factor 0.01 --concurrency 256 \
      --api-url http://localhost:8000/v1/chat/completions \
      --api-key sk-xxx --dataset dataset.jsonl

  # limit to first N requests
  python benchmark.py --mode sequential --concurrency 8 --num-requests 500 ...

  # repeat the dataset without waiting for each pass to finish
  python benchmark.py --mode sequential --concurrency 64 --repeat 3 ...

  # decode only, using original messages and fake KV
  python benchmark.py --decode-only --mode sequential --concurrency 128 \
      --api-url http://localhost:8000/v1/chat/completions --dataset dataset.jsonl
"""

import argparse
import asyncio
import json
import os
import resource
import secrets
import socket
import ssl
import sys
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import aiohttp

# SGLang recognizes this sentinel locally; it is not a real prefill address.
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


# ── result types ─────────────────────────────────────────────────────


@dataclass
class RequestResult:
    idx: int
    prompt_tokens: int
    actual_completion_tokens: int | None = None
    num_choices: int = 1
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    total_latency_ms: float = 0.0
    error: str | None = None
    repeat_index: int = 0


# ── progress bar ─────────────────────────────────────────────────────


def render_progress_bar(
    completed: int, total: int, ok: int, err: int, bar_width: int = 30
) -> str:
    """渲染一行进度条文本, 不含换行, 用于配合 '\\r' 原地刷新。"""
    total = max(total, 1)
    frac = min(completed / total, 1.0)
    filled = int(bar_width * frac)
    bar = "#" * filled + "-" * (bar_width - filled)
    return f"[{bar}] {completed}/{total} ok={ok} err={err} ({frac * 100:5.1f}%)"


# ── SSE streaming request ────────────────────────────────────────────


async def iter_sse_data(content):
    """Read SSE events without aiohttp.readline's limit on cumulative text."""
    pending = b""
    data_lines = []
    async for data in content.iter_any():
        lines = (pending + data).split(b"\n")
        pending = lines.pop()
        for line in lines:
            line = line.rstrip(b"\r")
            if not line:
                if data_lines:
                    yield b"\n".join(data_lines)
                    data_lines = []
            elif line.startswith(b"data:"):
                data_lines.append(line[5:].removeprefix(b" "))
    if pending.startswith(b"data:"):
        data_lines.append(pending[5:].removeprefix(b" ").rstrip(b"\r"))
    if data_lines:
        yield b"\n".join(data_lines)


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    body: dict,
    idx: int,
    rec: dict,
    expected_output_len: int | None = None,
) -> RequestResult:
    result = RequestResult(
        idx=idx,
        prompt_tokens=rec.get("prompt_tokens", 0),
        num_choices=body.get("n") or 1,
    )

    t_start = time.monotonic()
    t_first_token = None
    t_last_token = None
    completion_tokens = 0
    first_chunk_tokens = 0
    reported_prompt_tokens = None
    saw_completion_usage = False
    timed_completion_tokens = 0
    finished_choices = set()
    choice_tokens = {}
    saw_done = False

    try:
        async with session.post(url, json=body, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                result.error = f"HTTP {resp.status}: {text[:2000]}"
                result.total_latency_ms = (time.monotonic() - t_start) * 1000
                return result

            async for payload in iter_sse_data(resp.content):
                if payload.strip() == b"[DONE]":
                    saw_done = True
                    break
                chunk = json.loads(payload)

                if chunk.get("error"):
                    result.error = f"SSE error: {json.dumps(chunk['error'], ensure_ascii=False)[:2000]}"
                    break

                choices = chunk.get("choices") or []
                for choice in choices:
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        finished_choices.add(choice.get("index", 0))
                        if reason in ("abort", "error"):
                            result.error = f"Generation aborted: {json.dumps(choice, ensure_ascii=False)[:2000]}"
                if result.error:
                    break
                usage = chunk.get("usage") or {}
                if "completion_tokens" in usage:
                    count = usage["completion_tokens"]
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid completion_tokens in chat usage")
                    if len(choices) == 1:
                        # SGLang continuous usage is per choice; final usage
                        # (choices=[]) aggregates all choices when n > 1.
                        index = choices[0].get("index", 0)
                        if count < choice_tokens.get(index, 0):
                            raise ValueError(
                                "Decreasing completion_tokens for a chat choice"
                            )
                        choice_tokens[index] = count
                        count = sum(choice_tokens.values())
                    if count < completion_tokens:
                        raise ValueError(
                            "Decreasing cumulative completion_tokens in chat stream"
                        )
                    # A final usage summary is accounting, not a token arrival.
                    if count > completion_tokens and choices:
                        now = time.monotonic()
                        if t_first_token is None:
                            t_first_token = now
                            first_chunk_tokens = count
                        t_last_token = now
                        timed_completion_tokens = count
                    completion_tokens = count
                    saw_completion_usage = True
                if "prompt_tokens" in usage:
                    reported_prompt_tokens = usage["prompt_tokens"]
                    if (
                        type(reported_prompt_tokens) is not int
                        or reported_prompt_tokens < 0
                    ):
                        raise ValueError("Invalid prompt_tokens in chat usage")
                    result.prompt_tokens = reported_prompt_tokens

    except Exception as e:
        result.error = f"{type(e).__name__}: {e!r}"  # 不再用 str(e)
        result.total_latency_ms = (time.monotonic() - t_start) * 1000
        print(f"  [req#{idx}] EXCEPTION: {result.error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()

    if result.error is None and not saw_done:
        result.error = "Incomplete SSE stream: missing [DONE]"
    if result.error is None:
        if len(finished_choices) < result.num_choices:
            result.error = "Incomplete chat response: missing finish_reason"
        elif reported_prompt_tokens is None or not saw_completion_usage:
            result.error = "Incomplete chat response: missing token usage"
        elif (
            expected_output_len is not None
            and completion_tokens != expected_output_len * result.num_choices
        ):
            result.error = (
                f"Output length mismatch: requested {expected_output_len} per choice "
                f"x {result.num_choices}, server reported {completion_tokens} total"
            )

    t_end = time.monotonic()
    result.total_latency_ms = (t_end - t_start) * 1000
    result.actual_completion_tokens = completion_tokens

    if t_first_token is not None:
        result.ttft_ms = (t_first_token - t_start) * 1000

    # A chat SSE event may contain multiple tokens (or no visible text).
    # Only time token increments observed after the first event.
    if (
        t_first_token is not None
        and t_last_token is not None
        and completion_tokens > first_chunk_tokens
        and timed_completion_tokens == completion_tokens
        and result.num_choices == 1
    ):
        decode_duration = (t_last_token - t_first_token) * 1000
        result.tpot_ms = decode_duration / (completion_tokens - first_chunk_tokens)

    return result


# ── build request body (force streaming) ─────────────────────────────


def decode_output_length(rec: dict) -> int:
    value = rec.get("completion_tokens")
    if type(value) is not int or value < 0:
        raise ValueError(
            f"completion_tokens must be an integer >= 0, got {value!r} "
            f"(dataset index {rec.get('dataset_idx', 'unknown')})"
        )
    return value


def decode_input_length(rec: dict) -> int:
    value = rec.get("prompt_tokens")
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"prompt_tokens must be an integer > 0 for fakeid input, got {value!r} "
            f"(dataset index {rec.get('dataset_idx', 'unknown')})"
        )
    return value


def build_request_body(
    rec: dict,
    decode_only: bool = False,
    decode_input: str = "real",
) -> dict:
    body = dict(rec["request_body"])
    body["stream"] = True
    body["stream_options"] = dict(body.get("stream_options") or {})
    body["stream_options"]["include_usage"] = True
    body["stream_options"]["continuous_usage_stats"] = True
    # Leave DP placement to the current server rather than historical ranks.
    for key in ("routed_dp_rank", "data_parallel_rank", "disagg_prefill_dp_rank"):
        body.pop(key, None)
    if decode_only:
        output_len = decode_output_length(rec)
        if output_len == 0:
            raise ValueError(
                "Zero-output records have no decode work; skip them before scheduling"
            )
        # Fixed-length decode work must not stop at a string or completed grammar.
        # Keep tool definitions in the prompt, but disable tool generation: SGLang
        # otherwise derives a grammar and (for Kimi K3) an extra stop string from
        # tools, even when the request has no explicit stop or response_format.
        for key in (
            "stop",
            "stop_token_ids",
            "stop_regex",
            "response_format",
            "regex",
            "ebnf",
            "json_schema",
            "structural_tag",
        ):
            body.pop(key, None)
        body["tool_choice"] = "none"
        if body.get("chat_template_kwargs"):
            template_kwargs = dict(body["chat_template_kwargs"])
            if "tool_choice" in template_kwargs:
                template_kwargs["tool_choice"] = "none"
            template_kwargs.pop("response_format", None)
            body["chat_template_kwargs"] = template_kwargs
        body.update(
            {
                "max_tokens": output_len,
                "max_completion_tokens": output_len,
                "ignore_eos": True,
                "bootstrap_host": FAKE_BOOTSTRAP_HOST,
                # Match SGLang Router's 63-bit random room IDs. The server
                # decides whether/how to use this key for its routing policy.
                "bootstrap_room": secrets.randbits(63),
            }
        )
        if decode_input == "fakeid":
            # SGLang chat accepts input_ids as an override and skips template
            # tokenization. Keep the original body for its other request options.
            body["input_ids"] = [1] * decode_input_length(rec)
    return body


def load_records(path: str, num_requests: int = 0):
    records = []
    with open(path) as f:
        for idx, line in enumerate(f):
            if num_requests > 0 and idx >= num_requests:
                break
            rec = json.loads(line)
            body = rec.get("request_body")
            if (
                not isinstance(body, dict)
                or not isinstance(body.get("messages"), list)
                or not body["messages"]
            ):
                raise ValueError(
                    f"Dataset line {idx + 1}: request_body.messages must be a non-empty list"
                )
            rec["dataset_idx"] = idx
            rec.setdefault("delay_ms", 0)
            records.append(rec)
    return records


# ── mode: sequential ─────────────────────────────────────────────────


async def run_sequential(
    records: list[dict],
    url: str,
    api_key: str,
    concurrency: int,
    on_result,
    request_timeout: float = 0,
    on_start=None,
    decode_only: bool = False,
    disable_connection_reuse: bool = False,
    decode_input: str = "real",
    repeat: int = 1,
    connect_timeout: float = 0,
):
    # One shared iterator crosses pass boundaries immediately. Keep only one
    # worker per slot instead of allocating a task for every repeated request.
    pending = (
        (repeat_index, idx, rec)
        for repeat_index in range(repeat)
        for idx, rec in enumerate(records)
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with aiohttp.ClientSession(
        # The worker count is the only concurrency limit. aiohttp otherwise caps
        # this session at 100 connections, even when concurrency is higher.
        connector=aiohttp.TCPConnector(
            limit=0, limit_per_host=0, force_close=disable_connection_reuse
        ),
        # Dataset records are independent requests, not one browser session.
        cookie_jar=aiohttp.DummyCookieJar(),
        timeout=aiohttp.ClientTimeout(
            total=request_timeout, sock_connect=connect_timeout or None
        ),
    ) as session:

        async def worker():
            for repeat_index, idx, rec in pending:
                body = build_request_body(rec, decode_only, decode_input)
                if on_start is not None:
                    on_start()
                result = await send_streaming_request(
                    session,
                    url,
                    headers,
                    body,
                    rec.get("dataset_idx", idx),
                    rec,
                    expected_output_len=(
                        rec["completion_tokens"] if decode_only else None
                    ),
                )
                result.repeat_index = repeat_index
                on_result(result)

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(concurrency, len(records) * repeat))
        ]
        await asyncio.gather(*tasks)


# ── mode: replay ─────────────────────────────────────────────────────


async def run_replay(
    records: list[dict],
    url: str,
    api_key: str,
    scale_factor: float,
    concurrency: int,
    on_result,
    request_timeout: float = 0,
    on_start=None,
    decode_only: bool = False,
    disable_connection_reuse: bool = False,
    decode_input: str = "real",
    repeat: int = 1,
    connect_timeout: float = 0,
):
    # concurrency <= 0 表示不限并发，用所有遍的总请求数作为上限。
    semaphore = asyncio.Semaphore(
        concurrency if concurrency > 0 else max(len(records) * repeat, 1)
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    pass_span_ms = max((rec["delay_ms"] for rec in records), default=0)

    t0 = asyncio.get_event_loop().time()

    async with aiohttp.ClientSession(
        # Keep the replay semaphore as the only concurrency limit, including
        # concurrency=0 (unlimited).
        connector=aiohttp.TCPConnector(
            limit=0, limit_per_host=0, force_close=disable_connection_reuse
        ),
        cookie_jar=aiohttp.DummyCookieJar(),
        timeout=aiohttp.ClientTimeout(
            total=request_timeout, sock_connect=connect_timeout or None
        ),
    ) as session:

        async def worker(repeat_index: int, idx: int, rec: dict):
            scheduled_delay = (
                (repeat_index * pass_span_ms + rec["delay_ms"]) / 1000.0 * scale_factor
            )
            now = asyncio.get_event_loop().time() - t0
            wait = scheduled_delay - now
            if wait > 0:
                await asyncio.sleep(wait)

            async with semaphore:
                body = build_request_body(rec, decode_only, decode_input)
                if on_start is not None:
                    on_start()
                result = await send_streaming_request(
                    session,
                    url,
                    headers,
                    body,
                    rec.get("dataset_idx", idx),
                    rec,
                    expected_output_len=(
                        rec["completion_tokens"] if decode_only else None
                    ),
                )
            result.repeat_index = repeat_index
            on_result(result)

        tasks = [
            asyncio.create_task(worker(repeat_index, i, r))
            for repeat_index in range(repeat)
            for i, r in enumerate(records)
        ]
        await asyncio.gather(*tasks)


# ── report generation ────────────────────────────────────────────────


def percentiles_float(sorted_vals: list[float], ps: list[float]) -> list[float]:
    n = len(sorted_vals)
    if n == 0:
        return [0.0] * len(ps)
    out = []
    for p in ps:
        idx = min(int(p / 100.0 * n), n - 1)
        out.append(sorted_vals[idx])
    return out


def bar_chart(buckets: list[tuple[str, int]], max_bar: int = 40) -> str:
    peak = max(c for _, c in buckets) if buckets else 1
    lines = []
    for label, count in buckets:
        bar_len = int(count / peak * max_bar) if peak > 0 else 0
        lines.append(f"  {label:>14s} | {'█' * bar_len} {count}")
    return "\n".join(lines)


def build_ms_histogram(
    sorted_vals: list[float], edges: list[float]
) -> list[tuple[str, int]]:
    buckets = []
    vi = 0
    n = len(sorted_vals)
    for i, edge in enumerate(edges):
        lo = edges[i - 1] if i > 0 else 0
        label = f"{lo:.0f}-{edge:.0f}ms"
        count = 0
        while vi < n and sorted_vals[vi] < edge:
            count += 1
            vi += 1
        buckets.append((label, count))
    rest = n - vi
    if rest > 0:
        buckets.append((f"{edges[-1]:.0f}ms+", rest))
    return buckets


def generate_report(
    results: list[RequestResult],
    mode: str,
    wall_clock_s: float,
    args,
) -> str:
    total = len(results)
    errors = [r for r in results if r.error]
    ok = [r for r in results if r.error is None]

    ttft_vals = sorted(r.ttft_ms for r in ok if r.ttft_ms is not None)
    tpot_vals = sorted(r.tpot_ms for r in ok if r.tpot_ms is not None)
    latency_vals = sorted(r.total_latency_ms for r in ok)

    total_completion = sum(r.actual_completion_tokens or 0 for r in ok)
    total_prompt = sum(r.prompt_tokens for r in ok)

    ps = [1, 5, 10, 25, 50, 75, 80, 90, 95, 99]
    p_labels = [f"P{p}" for p in ps]

    ttft_ps = percentiles_float(ttft_vals, ps)
    tpot_ps = percentiles_float(tpot_vals, ps)
    latency_ps = percentiles_float(latency_vals, ps)

    ttft_edges = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000]
    tpot_edges = [5, 10, 20, 50, 100, 200, 500, 1000]
    latency_edges = [500, 1000, 2000, 5000, 10000, 30000, 60000, 120000]

    ttft_hist = build_ms_histogram(ttft_vals, ttft_edges)
    tpot_hist = build_ms_histogram(tpot_vals, tpot_edges)
    latency_hist = build_ms_histogram(latency_vals, latency_edges)

    def fmt(v: float) -> str:
        return f"{v:.2f}"

    def table(headers, rows):
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        lines = []
        lines.append(
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        )
        lines.append(
            "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
                + " |"
            )
        return "\n".join(lines)

    lines = ["# Benchmark Report", ""]

    lines += [
        "## Configuration",
        "",
        f"- **Mode**: {mode}",
        f"- **API URL**: {args.api_url}",
        f"- **Dataset**: {args.dataset}",
        f"- **Repeat**: {args.repeat} (passes may overlap; one shared concurrency limit)",
        f"- **Num requests**: {total}",
        f"- **HTTP connection reuse**: {'disabled' if args.disable_connection_reuse else 'enabled'}",
        f"- **Request timeout**: {str(args.request_timeout) + ' s' if args.request_timeout else 'disabled'}",
        f"- **Connect timeout**: {str(args.connect_timeout) + ' s' if args.connect_timeout else 'disabled'}",
        "- **Sampling**: original request values; decode-only changes output limits and disables EOS/stop/grammar termination",
        "- **Token counts**: server usage",
    ]
    if args.decode_only:
        input_description = (
            "fake token IDs" if args.decode_input == "fakeid" else "original messages"
        )
        lines += [
            f"- **Workload**: decode only ({input_description}, fake KV, ignore_eos=true)",
            f"- **Decode input**: {args.decode_input}",
            "- **Output limit**: historical completion_tokens per choice; n is preserved",
            "- **Output constraints**: stop/response-format/grammar fields removed; tool_choice=none, tool definitions preserved",
            f"- **Skipped zero-output records**: {args.skipped_zero_output}",
            "- **Temperature**: original request value; omitted values use server defaults",
            "- **DP routing**: server policy; random bootstrap room per request, no explicit DP rank",
        ]
    else:
        lines.append(
            "- **Output limit**: original request_body; server applies its limits and EOS/stop behavior"
        )
    if mode == "sequential":
        lines.append(f"- **Concurrency**: {args.concurrency}")
    else:
        lines.append(f"- **Scale factor**: {args.scale_factor}")
        lines.append(
            f"- **Concurrency**: {args.concurrency if args.concurrency > 0 else 'unlimited'}"
        )
    lines += [""]

    lines += [
        "## Summary",
        "",
        f"- **Total requests**: {total}",
        f"- **Successful**: {len(ok)}",
        f"- **Errors**: {len(errors)}",
        f"- **Wall clock time**: {wall_clock_s:.2f}s",
        (
            f"- **Throughput**: {len(ok)/wall_clock_s:.2f} req/s"
            if wall_clock_s > 0
            else ""
        ),
        f"- **Total prompt tokens**: {total_prompt}",
        f"- **Total completion tokens**: {total_completion}",
        (
            f"- **Completion tok/s**: {total_completion/wall_clock_s:.2f}"
            if wall_clock_s > 0
            else ""
        ),
        "",
    ]

    if args.decode_only:
        decode_tokens = sum(
            max((r.actual_completion_tokens or 0) - r.num_choices, 0) for r in ok
        )
        lines += [
            f"- **Decode tokens excluding synthetic handoff**: {decode_tokens}",
            (
                f"- **Decode tok/s excluding synthetic handoff**: {decode_tokens/wall_clock_s:.2f}"
                if wall_clock_s > 0
                else ""
            ),
            "",
            "SGLang completion counts include one synthetic prefill handoff token per generated choice. "
            "TTFT measures time to the first token-count update, including queueing; "
            "it is not real-prefill TTFT. TPOT uses token-count increments after the first event "
            "and is unavailable without complete continuous usage, for a single-event response, or for n > 1. "
            "Final-only usage does not provide TTFT. Throughput uses the whole client run, "
            "not GPU kernel time. Fake KV results do not measure real-workload DSpark acceptance.",
            "",
        ]

    def add_metric_section(title, vals, ps_vals, hist):
        if not vals:
            lines.append(f"## {title}")
            lines.append("")
            lines.append("No data.")
            lines.append("")
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- **Count**: {len(vals)}")
        lines.append(f"- **Min**: {fmt(vals[0])} ms")
        lines.append(f"- **Max**: {fmt(vals[-1])} ms")
        lines.append(f"- **Mean**: {fmt(sum(vals)/len(vals))} ms")
        lines.append("")
        lines.append("### Percentiles")
        lines.append("")
        p_rows = [[l, f"{v:.2f} ms"] for l, v in zip(p_labels, ps_vals)]
        lines.append(table(["Percentile", "Value"], p_rows))
        lines.append("")
        lines.append("### Histogram")
        lines.append("")
        lines.append("```")
        lines.append(bar_chart(hist))
        lines.append("```")
        lines.append("")

    add_metric_section("TTFT (Time To First Token)", ttft_vals, ttft_ps, ttft_hist)
    add_metric_section("TPOT (Time Per Output Token)", tpot_vals, tpot_ps, tpot_hist)
    add_metric_section("Total Latency", latency_vals, latency_ps, latency_hist)

    if errors:
        lines.append("## Errors (first 20)")
        lines.append("")
        for r in errors[:20]:
            lines.append(f"- pass#{r.repeat_index + 1} req#{r.idx}: {r.error[:200]}")
        lines.append("")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="LLM streaming benchmark")
    p.add_argument("--mode", required=True, choices=["sequential", "replay"])
    p.add_argument("--api-url", required=True)
    p.add_argument("--api-key", default="")
    p.add_argument(
        "--decode-only",
        action="store_true",
        help="send chat requests with fake KV and historical output lengths; disable EOS, stop, and grammar termination",
    )
    p.add_argument(
        "--decode-input",
        choices=["real", "fakeid"],
        default="real",
        help="decode-only input: original messages (default real), or token ID 1 repeated prompt_tokens times (fakeid)",
    )
    p.add_argument("--dataset", default="dataset.jsonl")
    p.add_argument(
        "--num-requests",
        type=int,
        default=0,
        help="take the first N dataset records per pass; 0 = all",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="total dataset passes without waiting between passes (positive integer; default 1)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="in-flight request limit in either mode; replay accepts 0 for unlimited (default: sequential 16, replay 0)",
    )
    p.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="replay mode: time scale (0.01 = 100x faster, default 1.0 = original speed)",
    )
    p.add_argument(
        "--output",
        default="",
        help="Markdown report path; empty disables report generation",
    )
    p.add_argument(
        "--errors-file",
        default="",
        help="append failed requests as JSONL; empty disables error logging",
    )
    p.add_argument(
        "--model-override", default="", help="override model name in request body"
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the pre-run connectivity probe",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=0,
        help="per-request total timeout in seconds (default 0: disabled)",
    )
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=15.0,
        help=(
            "per-request TCP connect timeout in seconds (default 15; use 0 to disable). "
            "Applies only to establishing the connection, not the response; "
            "an unreachable backend behind a load balancer otherwise blocks "
            "on the OS-level connect timeout (~75s) instead of failing fast"
        ),
    )
    p.add_argument(
        "--disable-connection-reuse",
        action="store_true",
        help="open a new TCP connection per request for connection-level load balancers",
    )
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be greater than 0")
    if args.decode_input == "fakeid" and not args.decode_only:
        p.error("--decode-input fakeid requires --decode-only")
    if args.concurrency is None:
        args.concurrency = 16 if args.mode == "sequential" else 0
    if args.mode == "sequential" and args.concurrency <= 0:
        p.error("--concurrency must be greater than 0 in sequential mode")
    if args.decode_only:
        parts = urlsplit(args.api_url)
        path = parts.path.rstrip("/")
        if not path:
            args.api_url = parts._replace(path="/v1/chat/completions").geturl()
        elif not path.endswith("/v1/chat/completions"):
            p.error(
                "--decode-only requires a direct Decode HTTP URL ending in /v1/chat/completions, or a base URL"
            )
        else:
            args.api_url = parts._replace(path=path).geturl()
    return args


def check_fd_limit(concurrency: int) -> str | None:
    """检查进程的 fd 软限制是否够撑住 concurrency 条并发连接。返回问题描述, None 表示够用。

    TCPConnector 在这里配的是 limit=0 (不设上限), concurrency 才是真正的并发闸门,
    所以并发一旦超过 shell 的 fd 软限制 (常见默认值远低于 ulimit -n 显示的硬限制),
    就会在跑到几百个请求后才炸出 EMFILE, 而不是一开跑就报错。
    """
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # 给 stdio/数据集文件/错误日志/DNS 等已占用的 fd 留出余量。
    headroom = 64
    needed = concurrency + headroom
    if needed <= soft_limit:
        return None
    raisable = hard_limit if hard_limit != resource.RLIM_INFINITY else needed
    if raisable >= needed:
        return (
            f"并发 {concurrency} 需要约 {needed} 个文件描述符, "
            f"但当前进程的 fd 软限制只有 {soft_limit} (硬限制 {hard_limit})。"
            f" 运行前先执行 `ulimit -n {needed}` (或更大的值) 再启动本脚本。"
        )
    return (
        f"并发 {concurrency} 需要约 {needed} 个文件描述符, "
        f"超过了这台机器的 fd 硬限制 {hard_limit}。"
        f" 需要用 launchctl/系统设置调高 maxfiles 后再重试, 或调低 --concurrency。"
    )


def validate_api_url(url: str) -> str | None:
    """检查 --api-url 是否可用。返回问题描述, None 表示看起来正常。

    畸形 URL 会让每条请求都栽在 DNS 上, 而且报错里看不出原始 URL,
    所以宁可开跑前就拒绝。
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return (
            f"scheme 解析成了 {parts.scheme!r}, 必须是 http 或 https。"
            " URL 是不是漏了 'http://' 前缀?"
        )
    host = parts.hostname
    if not host:
        return "解析不出主机名。URL 可能不完整, 或 '//' 写成了 '/'。"
    if host in ("http", "https"):
        return (
            f"主机名被解析成 {host!r} —— URL 里有两个 scheme,"
            f" 形如 'https://http://1.2.3.4:8000/...'。"
            " 检查拼 URL 的地方是不是给已带 scheme 的值又加了前缀。"
        )
    return None


async def preflight(url: str, timeout_s: float = 10.0) -> str | None:
    """开跑前连一次目标端口。返回问题描述, None 表示连通。

    只做 TCP 连接 (https 则含 TLS 握手), 不发 HTTP 请求 ——
    目的是把连接层的配置错误 (scheme 用错/端口不对/服务没起)
    在发 1500 条请求之前就暴露出来, 而不是收 1500 条一样的栈。
    """
    parts = urlsplit(url)
    host = parts.hostname
    use_tls = parts.scheme == "https"
    port = parts.port or (443 if use_tls else 80)

    # 用默认校验, 与 aiohttp 真实请求的行为一致 ——
    # 探测若放宽校验, 就会在证书不可信时给出假绿灯。
    ssl_ctx = ssl.create_default_context() if use_tls else None

    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=timeout_s,
        )
        return None
    except ssl.SSLCertVerificationError as e:
        return (
            f"TLS 证书校验失败: {e}."
            " 真实请求用同样的校验, 所以会同样失败。"
            " 若该端点本就是明文, 把 https:// 改成 http://;"
            " 若确实是 TLS, 把签发 CA 加进信任库。"
        )
    except ssl.SSLError as e:
        msg = str(e)
        if "WRONG_VERSION_NUMBER" in msg or "record layer failure" in msg:
            return (
                f"TLS 握手失败 ({msg})。"
                f" {host}:{port} 讲的是明文 HTTP, 但 URL 用了 https://。"
                " 把 --api-url 的 https:// 改成 http://。"
            )
        return f"TLS 握手失败: {msg}"
    except TimeoutError:
        return (
            f"连接 {host}:{port} 超过 {timeout_s:.0f}s 无响应。"
            " 端口不对, 或被防火墙/安全组挡了。"
        )
    except OSError as e:
        hint = ""
        if isinstance(e, ConnectionRefusedError):
            hint = " 端口上没有服务在监听 —— 端口号是否写错, 或服务没起?"
        elif isinstance(e, socket.gaierror):
            hint = " 主机名解析不了 —— 拼写错误, 或该环境没有对应 DNS?"
        return f"连不上 {host}:{port}: {type(e).__name__}: {e}.{hint}"
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ssl.SSLError):
                pass


def main():
    args = parse_args()

    problem = validate_api_url(args.api_url)
    if problem:
        print(f"ERROR: --api-url 不可用: {args.api_url!r}", file=sys.stderr)
        print(f"       {problem}", file=sys.stderr)
        sys.exit(2)

    if args.concurrency > 0:
        problem = check_fd_limit(args.concurrency)
        if problem:
            print(f"ERROR: {problem}", file=sys.stderr)
            sys.exit(2)

    parts = urlsplit(args.api_url)
    print(
        f"Target: {args.api_url}  "
        f"(host={parts.hostname} port={parts.port or ('443' if parts.scheme == 'https' else '80')})",
        file=sys.stderr,
    )

    if not args.skip_preflight:
        problem = asyncio.run(preflight(args.api_url))
        if problem:
            print(f"ERROR: 连接层探测失败: {problem}", file=sys.stderr)
            print(
                "       (确认目标没问题的话, 可用 --skip-preflight 跳过)",
                file=sys.stderr,
            )
            sys.exit(2)
        print("  preflight OK", file=sys.stderr)

    print(f"Loading dataset from {args.dataset} ...", file=sys.stderr)
    try:
        records = load_records(args.dataset, args.num_requests)
        loaded_count = len(records)
        if args.decode_only:
            records = [r for r in records if decode_output_length(r) > 0]
            if args.decode_input == "fakeid":
                for rec in records:
                    decode_input_length(rec)
        skipped_per_pass = loaded_count - len(records)
        args.skipped_zero_output = skipped_per_pass * args.repeat
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"  loaded {loaded_count} records", file=sys.stderr)
    if args.skipped_zero_output:
        print(
            f"  skipped {skipped_per_pass} zero-output records per pass (no decode work)",
            file=sys.stderr,
        )
    total_requests = len(records) * args.repeat
    if args.decode_only:
        print(
            f"  decode only: input={args.decode_input}, original sampling, fake KV, ignore_eos=true, "
            "stop/grammar disabled, tool_choice=none; "
            "output length per choice comes from recorded completion_tokens",
            file=sys.stderr,
        )

    if args.model_override:
        for r in records:
            r["request_body"]["model"] = args.model_override

    results: list[RequestResult] = []
    started = 0
    completed = 0
    error_count = 0
    errors_handle = (
        open(args.errors_file, "a", encoding="utf-8") if args.errors_file else None
    )

    def on_start():
        nonlocal started
        started += 1

    def on_result(r: RequestResult):
        nonlocal completed, error_count
        results.append(r)
        completed += 1
        if r.error:
            error_count += 1
            print(
                f"  [pass#{r.repeat_index + 1} req#{r.idx}] ERROR: {r.error}",
                file=sys.stderr,
            )
            sys.stderr.flush()
        if r.error and errors_handle:
            errors_handle.write(
                json.dumps(
                    {
                        "idx": r.idx,
                        "repeat_index": r.repeat_index,
                        "prompt_tokens": r.prompt_tokens,
                        "actual_completion_tokens": r.actual_completion_tokens,
                        "ttft_ms": (
                            round(r.ttft_ms, 3) if r.ttft_ms is not None else None
                        ),
                        "tpot_ms": (
                            round(r.tpot_ms, 3) if r.tpot_ms is not None else None
                        ),
                        "total_latency_ms": round(r.total_latency_ms, 3),
                        "error": r.error,
                    }
                )
                + "\n"
            )
            errors_handle.flush()

    def show_progress(final=False):
        line = render_progress_bar(
            completed, total_requests, completed - error_count, error_count
        )
        line += (
            f" started={started} in_flight={started - completed}"
            f" pending={total_requests - started}"
            f" elapsed={time.monotonic() - t_wall_start:.1f}s"
        )
        is_tty = sys.stderr.isatty()
        line = "  " + line
        if is_tty:
            with suppress(OSError):
                width = os.get_terminal_size(sys.stderr.fileno()).columns
                if len(line) >= width:
                    line = (
                        f" {completed}/{total_requests} ok={completed - error_count} "
                        f"err={error_count} in={started - completed} "
                        f"{time.monotonic() - t_wall_start:.0f}s"
                    )[: max(width - 1, 0)]
        prefix = "\r\033[K" if is_tty else ""
        end = "\n" if final or not is_tty else ""
        print(f"{prefix}{line}", end=end, file=sys.stderr, flush=True)

    async def with_progress(run):
        async def heartbeat():
            while True:
                await asyncio.sleep(1)
                show_progress()

        show_progress()
        ticker = asyncio.create_task(heartbeat())
        try:
            await run
        finally:
            ticker.cancel()
            with suppress(asyncio.CancelledError):
                await ticker
            show_progress(final=True)

    concurrency = args.concurrency if args.concurrency > 0 else "unlimited"
    print(
        f"Starting benchmark: mode={args.mode}, requests={total_requests}, concurrency={concurrency}, repeat={args.repeat}",
        file=sys.stderr,
    )
    t_wall_start = time.monotonic()

    if args.mode == "sequential":
        asyncio.run(
            with_progress(
                run_sequential(
                    records,
                    args.api_url,
                    args.api_key,
                    args.concurrency,
                    on_result,
                    args.request_timeout,
                    on_start,
                    decode_only=args.decode_only,
                    disable_connection_reuse=args.disable_connection_reuse,
                    decode_input=args.decode_input,
                    repeat=args.repeat,
                    connect_timeout=args.connect_timeout,
                )
            )
        )
    else:
        total_span_s = max((r["delay_ms"] for r in records), default=0) / 1000.0
        scaled_span = total_span_s * args.scale_factor * args.repeat
        print(
            f"  replay span: {total_span_s:.1f}s per pass -> {scaled_span:.1f}s scheduled total (factor={args.scale_factor}, repeat={args.repeat})",
            file=sys.stderr,
        )
        asyncio.run(
            with_progress(
                run_replay(
                    records,
                    args.api_url,
                    args.api_key,
                    args.scale_factor,
                    args.concurrency,
                    on_result,
                    args.request_timeout,
                    on_start,
                    decode_only=args.decode_only,
                    disable_connection_reuse=args.disable_connection_reuse,
                    decode_input=args.decode_input,
                    repeat=args.repeat,
                    connect_timeout=args.connect_timeout,
                )
            )
        )

    wall_clock_s = time.monotonic() - t_wall_start
    print(f"Benchmark finished in {wall_clock_s:.2f}s", file=sys.stderr)

    results.sort(key=lambda r: (r.repeat_index, r.idx))

    report = generate_report(results, args.mode, wall_clock_s, args)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)

    if errors_handle:
        errors_handle.close()


if __name__ == "__main__":
    main()

