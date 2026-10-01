"""Phase 0 Gate — can this machine run the model fast enough?

This is the first script in the project and the only one that can invalidate
the rest of the plan, so it runs before any application code is written.

It measures the two phases of generation separately, because they are bound by
different bottlenecks and reporting them together hides which one dominates:

  prefill  the model reads the whole prompt      compute-bound
  decode   the model writes one token at a time  memory-bandwidth-bound

A RAG prompt is mostly prefill (four retrieved passages), so prefill speed is
what decides whether this design is viable on a CPU.

IMPORTANT — every run uses a UNIQUE prompt. Ollama caches the KV state of a
prompt prefix, so re-sending an identical prompt skips prefill entirely and
reports an impossible ~11,000 tok/s. The first version of this script did
exactly that and produced a falsely optimistic verdict. In real use each
question retrieves different passages, so prefill genuinely runs every time;
the unique marker injected at the head of each prompt reproduces that.

Usage:  python scripts/00_smoke_test.py
Writes: results/smoke_test.json
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from pathlib import Path

import requests

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "smoke_test.json"

SYSTEM = (
    "You answer questions about wastewater treatment aeration control using only "
    "the provided excerpts. Cite every factual claim with a bracketed number, e.g. [2]. "
    "If the excerpts do not contain the answer, reply exactly: NOT_IN_CORPUS. "
    "Do not use outside knowledge."
)

# Representative domain prose at roughly the length the real pipeline retrieves
# (~300 tokens per passage), so the tokenizer sees a realistic distribution.
PASSAGE_BODY = (
    "In sequencing batch reactor operation the aerobic phase is conventionally terminated on a "
    "fixed timer, which ignores the substantial day-to-day variation in influent ammonia load. "
    "Real-time control instead infers the endpoint from characteristic inflections in the online "
    "sensor signals. The dissolved oxygen concentration remains suppressed while nitrifying "
    "organisms exert high oxygen demand, and rises sharply once ammonium is depleted, producing a "
    "distinct elbow in the dissolved oxygen trajectory. The pH profile exhibits a corresponding "
    "minimum, frequently termed the ammonia valley, arising from the acidification associated with "
    "nitrification followed by partial recovery once that reaction ceases. During the subsequent "
    "anoxic phase the oxidation reduction potential declines steeply while nitrate serves as the "
    "terminal electron acceptor, and the slope flattens abruptly at the point of nitrate exhaustion, "
    "a feature commonly described as the nitrate knee. Reported results indicate that first "
    "derivatives of the oxidation reduction potential and pH signals provide more reliable detection "
    "than the raw values, since the derivative isolates the change in trend from slow baseline drift "
    "attributable to temperature and probe fouling. Full scale implementations report aeration energy "
    "savings in the range of fourteen to thirty-two percent relative to fixed duration operation, "
    "with total nitrogen removal maintained or improved. One practical caveat concerns airflow "
    "modulation: when the blower output is varied within a phase to hold a dissolved oxygen setpoint, "
    "the control action itself introduces bends in the measured curve that can be mistaken for the "
    "biological endpoint, so once-per-cycle airflow control is preferred to keep the signal clean."
)

PASSAGES = [
    ("Real-time phase control in SBR systems", 2010, 4),
    ("Derivative-based endpoint detection", 2019, 7),
    ("Energy savings from event-driven aeration", 2021, 3),
    ("Sensor drift and control-induced artefacts", 2023, 11),
]

QUESTION = "When should aeration be stopped, and which signal is most reliable for detecting it?"


def build_prompt(n_passages: int) -> str:
    """Build a RAG-shaped prompt with a unique head so prefill is never cached."""
    marker = f"Session reference {uuid.uuid4().hex}. Ignore this line.\n\n"
    blocks = [
        f"[{i}] {title} ({year}), p.{page}\n{PASSAGE_BODY}"
        for i, (title, year, page) in enumerate(PASSAGES[:n_passages], start=1)
    ]
    return marker + "\n\n".join(blocks) + f"\n\nQuestion: {QUESTION}"


def call(prompt: str, num_ctx: int, num_predict: int = 200) -> dict:
    started = time.perf_counter()
    response = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": "30m",
            # num_ctx is set explicitly on every request: server defaults have
            # changed across versions, and silently getting 2048 would truncate
            # context and look like a retrieval failure.
            "options": {
                "temperature": 0,
                "seed": 0,
                "top_p": 1.0,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        },
        timeout=900,
    )
    response.raise_for_status()
    body = response.json()
    body["_wall_seconds"] = time.perf_counter() - started
    return body


def summarise(body: dict) -> dict:
    ns = 1_000_000_000
    prompt_tokens = body.get("prompt_eval_count", 0)
    prompt_seconds = body.get("prompt_eval_duration", 0) / ns
    output_tokens = body.get("eval_count", 0)
    output_seconds = body.get("eval_duration", 0) / ns
    return {
        "prompt_tokens": prompt_tokens,
        "prefill_seconds": round(prompt_seconds, 2),
        "prefill_tok_per_sec": round(prompt_tokens / prompt_seconds, 1) if prompt_seconds else None,
        "output_tokens": output_tokens,
        "decode_seconds": round(output_seconds, 2),
        "decode_tok_per_sec": round(output_tokens / output_seconds, 1) if output_seconds else None,
        "load_seconds": round(body.get("load_duration", 0) / ns, 2),
        "wall_seconds": round(body["_wall_seconds"], 2),
    }


def measure(label: str, n_passages: int, num_ctx: int, repeats: int = 2) -> dict:
    print(f"{label}  ({n_passages} passages, num_ctx={num_ctx}) ...")
    samples = []
    for attempt in range(repeats):
        result = summarise(call(build_prompt(n_passages), num_ctx=num_ctx))
        samples.append(result)
        print(
            f"  run {attempt + 1}: prefill {result['prompt_tokens']} tok in "
            f"{result['prefill_seconds']}s ({result['prefill_tok_per_sec']} tok/s) | "
            f"decode {result['output_tokens']} tok ({result['decode_tok_per_sec']} tok/s) | "
            f"total {result['wall_seconds']}s"
        )

    aggregate = {
        "n_passages": n_passages,
        "num_ctx": num_ctx,
        "prompt_tokens": samples[0]["prompt_tokens"],
        "prefill_tok_per_sec_median": round(
            statistics.median(s["prefill_tok_per_sec"] for s in samples), 1
        ),
        "decode_tok_per_sec_median": round(
            statistics.median(s["decode_tok_per_sec"] for s in samples), 1
        ),
        "prefill_seconds_median": round(statistics.median(s["prefill_seconds"] for s in samples), 2),
        "decode_seconds_median": round(statistics.median(s["decode_seconds"] for s in samples), 2),
        "wall_seconds_median": round(statistics.median(s["wall_seconds"] for s in samples), 2),
        "wall_seconds_all": [s["wall_seconds"] for s in samples],
    }
    print(f"  -> median end to end: {aggregate['wall_seconds_median']}s\n")
    return aggregate


def verdict(seconds: float) -> str:
    if seconds < 45:
        return "PASS - proceed with the 3B model"
    if seconds <= 120:
        return "MARGINAL - consider qwen2.5:1.5b, or fewer passages per prompt"
    return "FAIL - abandon generation, use the extractive fallback"


def main() -> int:
    print(f"model  {MODEL}")
    print("every run uses a unique prompt, so prefill is never served from cache\n")

    # k is the design lever: prefill dominates and scales with prompt length,
    # so measuring the curve tells us what to set it to.
    runs = {
        f"k{k}_ctx4096": measure(f"{k} passages", n_passages=k, num_ctx=4096)
        for k in (4, 3, 2)
    }

    headline = runs["k3_ctx4096"]["wall_seconds_median"]
    decision = verdict(headline)

    payload = {
        "model": MODEL,
        "note": "unique prompt per run; prefill is genuine, not cache-served",
        "runs": runs,
        "chosen_k": 3,
        "headline_seconds_per_question": headline,
        "verdict": decision,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 62)
    print("latency vs passages per prompt")
    for k in (4, 3, 2):
        run = runs[f"k{k}_ctx4096"]
        marker = "  <- chosen" if k == 3 else ""
        print(
            f"  k={k}  {run['prompt_tokens']:>5} tok  "
            f"{run['wall_seconds_median']:>6.1f}s{marker}"
        )
    print()
    print(f"at k=3: {decision}")
    print("=" * 62)
    print(f"written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
