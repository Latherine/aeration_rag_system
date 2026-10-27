"""Two simple charts for the retrieval comparison, in matplotlib's default style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; never pops a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHODS = ("bm25", "dense", "hybrid")
METHOD_LABEL = {"bm25": "Keyword (BM25)", "dense": "Meaning (dense)", "hybrid": "Hybrid"}
KIND_LABEL = {"shared": "Paper's own words", "abbreviation": "Abbreviation", "paraphrase": "Everyday words"}


def overall(found: dict[str, int], n_questions: int, k: int, path: Path) -> Path:
    """Number of questions (out of n) where the correct passage was in the top k."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = [found[m] for m in METHODS]
    bars = ax.bar([METHOD_LABEL[m] for m in METHODS], counts)
    ax.bar_label(bars)
    ax.set_ylim(0, n_questions)
    ax.set_ylabel("Number of questions")
    ax.set_title(f"Correct passage found in the top {k} ({n_questions} questions)")
    return _save(fig, path)


def by_kind(found_share: dict[str, dict[str, float]], counts: dict[str, int], path: Path) -> Path:
    """Top-3 success rate by how the question is worded."""
    kinds = list(counts)
    x = np.arange(len(kinds))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, method in enumerate(METHODS):
        values = [found_share[method][kind] * 100 for kind in kinds]
        bars = ax.bar(x + (i - 1) * width, values, width, label=METHOD_LABEL[method])
        ax.bar_label(bars, fmt="%.0f%%")
    ax.set_xticks(x, [f"{KIND_LABEL[k]}\n({counts[k]} questions)" for k in kinds])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Correct passage in top 3 (%)")
    ax.set_title("Correct passage found in the top 3, by question wording")
    ax.legend()
    return _save(fig, path)


def glossary_effect(plain: dict[str, int], expanded: dict[str, int], n_questions: int, k: int,
                    path: Path) -> Path:
    """Questions found in the top k, each method without and with glossary expansion."""
    x = np.arange(len(METHODS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, found, label in ((-width / 2, plain, "Without glossary"), (width / 2, expanded, "With glossary")):
        bars = ax.bar(x + offset, [found[m] for m in METHODS], width, label=label)
        ax.bar_label(bars)
    ax.set_xticks(x, [METHOD_LABEL[m] for m in METHODS])
    ax.set_ylim(0, n_questions)
    ax.set_ylabel("Number of questions")
    ax.set_title(f"Correct passage found in the top {k} ({n_questions} questions)")
    ax.legend()
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
