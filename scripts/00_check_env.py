"""Phase 0, Block 1 — verify the environment is actually usable.

Installing a package and being able to use it are different things. This
checks that every dependency imports, that the embedding model runs end to
end on this CPU, and — critically — that torch did not sneak in as a
transitive dependency, since the whole point of choosing fastembed over
sentence-transformers was to keep it out.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import time


def check_imports() -> list[tuple[str, str]]:
    results = []
    for module, label in [
        ("fitz", "pymupdf"),
        ("fastembed", "fastembed"),
        ("numpy", "numpy"),
        ("rank_bm25", "rank-bm25"),
        ("requests", "requests"),
        ("yaml", "pyyaml"),
        ("pytest", "pytest"),
    ]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            results.append((label, f"OK  {version}"))
        except Exception as exc:  # noqa: BLE001 - we want the failure text
            results.append((label, f"FAIL  {exc}"))
    return results


def check_torch_absent() -> tuple[bool, str]:
    """torch must NOT be installed. See the RAM constraint in CLAUDE.md."""
    found = importlib.util.find_spec("torch") is not None
    return (not found, "torch absent, as intended" if not found else "torch IS INSTALLED — investigate")


def check_embedding_roundtrip() -> tuple[bool, str]:
    """Embed two sentences and confirm the similar pair scores higher.

    This is the real test: it downloads the model on first run, executes an
    ONNX session on CPU, and verifies the output is numerically sensible.
    """
    import numpy as np
    from fastembed import TextEmbedding

    started = time.perf_counter()
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir="./cache/fastembed")
    load_seconds = time.perf_counter() - started

    passages = [
        "The DO elbow marks the end of nitrification in an SBR cycle.",
        "Settling tanks require periodic cleaning to prevent clogging.",
    ]
    query = "When does nitrification finish?"

    started = time.perf_counter()
    doc_vecs = np.array(list(model.embed(passages)))
    query_vec = np.array(list(model.query_embed([query])))[0]
    embed_seconds = time.perf_counter() - started

    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    query_vec /= np.linalg.norm(query_vec)
    scores = doc_vecs @ query_vec

    ok = scores[0] > scores[1]
    detail = (
        f"dim={doc_vecs.shape[1]}  load={load_seconds:.1f}s  embed={embed_seconds:.2f}s\n"
        f"      relevant passage   {scores[0]:+.3f}\n"
        f"      irrelevant passage {scores[1]:+.3f}"
    )
    return ok, detail


def main() -> int:
    print(f"Python  {sys.version.split()[0]}   {platform.platform()}")
    print()

    failures = 0

    print("Imports")
    for label, status in check_imports():
        print(f"  {label:<14} {status}")
        failures += status.startswith("FAIL")

    print()
    ok, detail = check_torch_absent()
    print(f"torch check     {'OK' if ok else 'FAIL'}  {detail}")
    failures += not ok

    print()
    print("Embedding round-trip (downloads the model on first run)")
    try:
        ok, detail = check_embedding_roundtrip()
        print(f"  {'OK' if ok else 'FAIL'}  {detail}")
        failures += not ok
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        failures += 1

    print()
    print("PASS — environment is usable" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
