# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context OpenAI completions benchmark for SM120 bring-up.

This benchmark is intentionally simple: it sends exact prompt token IDs to avoid
tokenizer differences in the HTTP hot path, salts every request so prefix cache
hits cannot inflate throughput, and reports generated-token and total-token
throughput.
"""

import argparse
import concurrent.futures
import json
import random
import time
import urllib.request
from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class RequestResult:
    wall_s: float
    prompt_tokens: int
    output_tokens: int


def _build_prompt_token_ids(tokenizer, prompt_tokens: int, request_idx: int) -> list[int]:
    rng = random.Random(0x5EED_120 + request_idx)
    pieces: list[str] = []
    token_ids: list[int] = []

    while len(token_ids) < prompt_tokens:
        for _ in range(1024):
            pieces.append(f"req{request_idx:03x}_{rng.getrandbits(48):012x}")
        text = " ".join(pieces)
        token_ids = tokenizer.encode(text, add_special_tokens=False)

    return token_ids[:prompt_tokens]


def _post_completion(
    url: str,
    model: str,
    prompt_token_ids: list[int],
    output_tokens: int,
    request_id: str,
    timeout_s: int,
) -> RequestResult:
    body = {
        "model": model,
        "prompt": prompt_token_ids,
        "add_special_tokens": False,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "cache_salt": request_id,
        "request_id": request_id,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    wall_s = time.perf_counter() - start
    usage = payload.get("usage") or {}
    return RequestResult(
        wall_s=wall_s,
        prompt_tokens=int(usage.get("prompt_tokens", len(prompt_token_ids))),
        output_tokens=int(usage.get("completion_tokens", output_tokens)),
    )


def _run_concurrency(
    args: argparse.Namespace,
    prompts: list[list[int]],
    concurrency: int,
) -> None:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _post_completion,
                args.url,
                args.model,
                prompts[i],
                args.output_tokens,
                f"{args.label}-c{concurrency}-r{i}",
                args.timeout_s,
            )
            for i in range(concurrency)
        ]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - start

    prompt_tokens = sum(result.prompt_tokens for result in results)
    output_tokens = sum(result.output_tokens for result in results)
    total_tokens = prompt_tokens + output_tokens
    avg_request_s = sum(result.wall_s for result in results) / len(results)
    print(
        f"c={concurrency:<2d} wall={wall_s:8.3f}s avg/req={avg_request_s:8.3f}s "
        f"prompt={prompt_tokens} output={output_tokens} "
        f"output_tok/s={output_tokens / wall_s:8.2f} "
        f"total_tok/s={total_tokens / wall_s:8.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--label", default="long-context")
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    max_concurrency = max(args.concurrency)
    prompts = [
        _build_prompt_token_ids(tokenizer, args.prompt_tokens, i)
        for i in range(max_concurrency)
    ]
    assert all(len(prompt) == args.prompt_tokens for prompt in prompts)

    print(
        f"=== bench label={args.label} model={args.model} "
        f"prompt_tokens={args.prompt_tokens} output_tokens={args.output_tokens} ===",
        flush=True,
    )
    if args.warmup_output_tokens > 0:
        _post_completion(
            args.url,
            args.model,
            prompts[0],
            args.warmup_output_tokens,
            f"{args.label}-warmup",
            args.timeout_s,
        )

    for concurrency in args.concurrency:
        _run_concurrency(args, prompts, concurrency)


if __name__ == "__main__":
    main()
