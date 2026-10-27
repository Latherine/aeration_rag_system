"""config.yaml -> frozen dataclasses. Every tunable value lives in config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int
    overlap_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class RetrievalConfig:
    k: int
    rrf_k: int
    fusion_pool: int


@dataclass(frozen=True)
class GeneratorConfig:
    model: str
    fallback_model: str
    host: str
    num_ctx: int
    num_predict: int
    temperature: float
    seed: int
    keep_alive: str


@dataclass(frozen=True)
class Config:
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generator: GeneratorConfig
    embedding_model: str
    embedding_cache: Path
    query_prefix: str
    papers: Path
    manifest: Path
    glossary: Path
    cache: Path
    results: Path


def load_config(path: Path = ROOT / "config.yaml") -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    return Config(
        chunking=ChunkingConfig(**raw["chunking"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        generator=GeneratorConfig(**raw["generator"]),
        embedding_model=raw["embedding"]["model"],
        embedding_cache=ROOT / raw["embedding"]["cache_dir"],
        query_prefix=raw["embedding"]["query_prefix"],
        papers=ROOT / paths["papers"],
        manifest=ROOT / paths["manifest"],
        glossary=ROOT / paths["glossary"],
        cache=ROOT / paths["cache"],
        results=ROOT / paths["results"],
    )
