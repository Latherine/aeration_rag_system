# Aeration QA — Project Instructions

A **retrieval-augmented question-answering assistant for wastewater-treatment aeration control**.
Ask it an operator/engineer question in plain language; it retrieves the relevant passages from a small
corpus of real papers and answers with citations back to paper, page and section.

This is a personal research/portfolio project. It is **not** a course deliverable and is not tied to
any other repo's conventions.

## Why this project exists

It grew out of a literature review of four families of aeration-control approaches: fixed-schedule
optimization via process simulation; real-time phase control from sensor "bending points" (the ammonia
valley in pH, the DO elbow, the nitrate knee and nitrate apex in ORP); model-based DO trajectory
optimization; and ML-based aeration control. Aeration electricity is reported to be more than half of a
treatment plant's running cost, so the domain matters — but the deliverable here is a *language* system
over that literature, not a control system.

**Important framing:** an earlier attempt at this idea (`sbr-transition-transformer/`, in the user's
separate `water-quality-forecasting` repo) built a transformer that localizes phase transitions in
synthetic pH/ORP/DO curves. It is **paused, not abandoned** — but it is a numeric time-series model that
merely *uses* the transformer architecture. It is not a language model. This project is the genuine
language-model project.

## Mandatory rules (do not violate)

These are hardware and licensing constraints, not preferences. Each one has cost someone a wasted
afternoon at least once, so do not re-litigate them without new evidence.

- **CPU only. 7.7 GB total RAM, of which ~3–4 GB is actually usable** (Windows takes the rest).
  The machine is an i5-1235U with Intel Iris Xe integrated graphics — there is no CUDA and no usable
  GPU acceleration.
- **Never propose fine-tuning or training a model.** On this hardware it is not slow, it is infeasible.
  The generator is always an off-the-shelf quantized open-weight model.
- **Generator is capped at 1.5B–3B parameters, Q4 quantized.** A 7B model at Q4 is ~4.5 GB and will
  page to disk, at which point throughput collapses by an order of magnitude and the symptom looks like
  "the model got slow" rather than "out of RAM".
- **Never use a reasoning/"thinking" model.** At 6–10 tokens/sec a 1,500-token reasoning trace is three
  minutes of wall clock before the answer even starts. Use explicitly non-thinking `-Instruct` variants.
- **No paid API, ever** — no OpenAI, no Anthropic, no hosted embedding service. Everything runs locally
  on open weights. This is a hard requirement from the user, not a cost optimization.
- **Never add `torch` or `sentence-transformers`.** They would pull a ~2.5 GB wheel and ~1 GB of
  resident memory at inference, which is fatal on a 3–4 GB budget shared with the LLM. `fastembed`
  (ONNX) does the same job in ~200 MB resident.
- **Never add langchain or llama-index.** They would hide the retrieval mechanics that this project
  exists to measure. Building retrieval by hand *is* the project.
- **Embedding and generation must never run at the same time.** Ollama is a separate resident process;
  the ONNX embedding session is in ours. Index-building embeds and exits; the interactive path embeds
  one query vector, then generates. Set `OLLAMA_MAX_LOADED_MODELS=1`.
- **Always set `num_ctx` explicitly on every Ollama request.** Server defaults have changed across
  versions; silently getting 2048 truncates the context and presents as a retrieval failure.
- **BGE embeddings need a query prefix on queries but not on passages.** fastembed 0.8.1 does **not**
  add it — `query_embed()` returns the same vector as `embed()` (verified 2026-10-13), so
  `Embedder.embed_query` prepends it explicitly. Forgetting it raises no error and silently weakens
  dense retrieval. `tests/test_retrieval.py::test_query_and_passage_embeddings_differ` guards this; do
  not delete that test.
- **PyMuPDF is AGPL-3.0.** This was a conscious, accepted tradeoff (it is the only extractor that gives
  block bounding boxes and font sizes, which are required to reconstruct two-column academic papers).
  The README must state it. If this repo is ever made public, the licence implication is real.
- **Never commit the PDFs** in `data/papers/` — they are copyrighted. `data/MANIFEST.csv` records
  provenance (including a sha256 per file) so someone else can obtain the same corpus.

## Evaluation philosophy

Inherited deliberately from the user's `water-quality-forecasting` repo, where a trivial persistence
baseline beat every ML model in 7 of 9 cases and an LSTM was deleted from the pipeline for not earning
its complexity. The same skepticism applies here:

- **BM25 keyword search is a strong baseline, not a strawman.** On a ~10-paper corpus dense in rare
  jargon (*ammonia valley*, *nitrate knee*, *DO elbow*) there is no vocabulary mismatch for embeddings
  to bridge, and BM25 may well win outright. "BM25 beat the neural embeddings" is a legitimate,
  publishable headline — write it up as a finding, not a failure.
- **Report paired bootstrap confidence intervals, not point estimates.** A positive difference whose CI
  crosses zero is noise.
- **LLM-as-judge is refused, on purpose.** The only judge available locally is the same weak model being
  evaluated, and a model judging its own output has a documented self-preference bias. Declining a
  metric with a stated reason is a stronger methodological signal than producing numbers that look
  rigorous and mean nothing. If you add automatic answer metrics, label them honestly as proxies.

## Stack, and what was rejected

| Component | Choice | Rejected alternative and why |
|---|---|---|
| Generator | Ollama + `qwen2.5:3b-instruct-q4_K_M` | `llama-cpp-python` — prebuilt wheels lag new CPython minors; source build needs MSVC + CMake |
| Embeddings | `fastembed` + BGE-small-en-v1.5 | `sentence-transformers` — drags in torch (RAM) |
| Vector store | numpy brute-force cosine | FAISS — wheel risk for zero benefit at ~400 chunks (a 0.6 MB matrix) |
| Keyword search | `rank-bm25` | `bm25s` is ~100× faster and you will never notice at this scale |
| PDF extraction | PyMuPDF | `pypdf` returns interleaved garbage on two-column papers |
| HTTP to Ollama | `requests` | the `ollama` client pulls httpx + pydantic (Rust extension = wheel risk) |

Environment: **Windows 11, Python 3.13**. PyMuPDF ships a Stable-ABI wheel covering 3.10–3.14 and
fastembed/onnxruntime both have 3.13 wheels, so there is no known wheel gap. If fastembed ever breaks,
pin `onnxruntime==1.22.*` first.

## Design decisions worth knowing before changing code

- **Chunks are ~300 tokens with 60-token overlap, sentence-aligned.** BGE-small silently truncates at
  512 tokens, so an oversized chunk loses its tail from the embedding while BM25 still sees all of it —
  which would quietly bias the central BM25-vs-dense comparison. Chunk size is also a latency lever:
  measured prefill is ~31–38 tok/s, so each 300-token chunk in a prompt costs ~10 s of silence before
  the first output token. At `k=3` that is ~35 s of prefill per answer (see Status).
- **Each chunk stores `text` and `embed_text` separately.** Dense retrieval embeds
  `"{title} | {section}\n\n{text}"`; BM25 indexes bare `text`. Prepending titles to the BM25 index would
  inflate each title's term frequency across every chunk of that paper. This asymmetry is deliberate.
- **The references section is cut during extraction.** It is ~15% of a paper and pure author-name and
  journal-title noise that otherwise dominates BM25 term statistics.
- **Citations are never trusted to the model.** A Sources block is always appended from chunk metadata
  by code, so provenance survives even if the model cites nothing. Inline `[n]` markers are validated
  against the blocks actually sent; out-of-range markers are counted as a hallucination signal.
- **`pipeline.py` treats generation as a swappable strategy.** If the local model turns out to be too
  slow on this CPU, the fallback is extractive QA — return the 2–3 most query-relevant sentences
  verbatim with citations. That is fast, cannot hallucinate, and leaves the retrieval layer (where this
  project's value lives) completely intact. Do not hard-wire generation into the pipeline.

## Working with the user

- A student, new to ML and RAG. **Explains in Vietnamese** when they are learning: define every new term
  the first time it appears, use one concrete running example, show the actual file and line. A
  technical answer without examples gets "không hiểu gì" — rewrite it simpler, don't add more detail.
- **Asks "why" before accepting a choice** and pushes back when terms are used loosely (they caught that
  a transformer over sensor numbers is not an LLM). Be precise about what a thing actually is.
- Anything written **for their professor is in English**, short, and plain: activity logs (often one
  paragraph per day, no dashes in summaries when asked), the one-page brief in `docs/project_brief.docx`.
  Never claim progress that isn't on disk — they will be asked to show it.
- Wants to understand each step, not just have it done: after running something, say which file did it
  and what the output means.

## Running things on this machine

- **Always use the venv interpreter explicitly:** `.venv\Scripts\python.exe scripts\<name>.py` from the
  repo root. Bare `python` resolves to a system Python 3.13.2 that has none of the dependencies and fails
  with `ModuleNotFoundError: fastembed`. The venv is Python **3.13.0** — the version line printed by
  `00_check_env.py` tells you which one actually ran.
- **VSCode's ▶ button is the Code Runner extension**, which ignores the selected interpreter and calls
  bare `python`. Run from the integrated terminal instead.
- **`Activate.ps1` is blocked** by the PowerShell execution policy ("running scripts is disabled"). Don't
  fight it — call `.venv\Scripts\python.exe` directly.
- **In `cmd.exe`, `cd` does not change drives.** Use `cd /d "D:\FALL 2026\COMP 560\aeration-qa"`.
- **Console encoding:** printing paper text raises `UnicodeEncodeError` (cp1252). Set
  `PYTHONIOENCODING=utf-8` for scripts that print extracted text.
- Tests: `.venv\Scripts\python.exe -m pytest -q` (`pytest.ini` puts `src/` on the path).
- Ollama runs as a separate server on `127.0.0.1:11434`; models live in `D:\ollama-models`
  (`OLLAMA_MODELS`, `OLLAMA_MAX_LOADED_MODELS=1` are set as user env vars). If the tray app starts but the
  server doesn't answer, start it with `ollama serve`.
- **Ollama auto-updates, and 0.40.11 ignored `OLLAMA_MODELS`** (2026-10-27): the tray app's server read
  `C:\Users\ACER\.ollama\models`, so `ollama list` was empty and every request 404'd. Symptom: server
  answers, model "missing". Fix used: quit the tray app, then in PowerShell
  `$env:OLLAMA_MODELS='D:\ollama-models'; ollama serve`. Permanent fix: set the model location to
  `D:\ollama-models` in the tray app's Settings. Never re-pull the model to "fix" it (that writes 1.9 GB to C:).

## Repo and git state

- Local repo on branch `main`. One commit so far (`b90f76e`, Phase 0); the extraction work, manifest,
  tests and brief are **not yet committed**.
- **No GitHub remote yet.** `gh` is not installed, so the user creates an empty repo on github.com
  (private recommended — see the AGPL note above; no README/.gitignore/licence so the first push doesn't
  conflict) and pastes the URL; then `git remote add origin <url>` and `git push -u origin main`.
- Commit or push only when the user asks. This repo is the user's own; the separate `onboarding` repo
  next to it is a **shared lab repo where no git operations are allowed at all**.

## Status

**Phase 0 complete (2026-09-29). Gate 0 passed at k=3.**

**What exists, and what is only planned.** `00_check_env.py` tests the embedding model (BGE-small) on two
sentences; `00_smoke_test.py` times the LLM (Qwen2.5-3B) on **hand-written fake passages** — it does no
retrieval. (Written at Phase 0; retrieval and generation now exist, see the Phase 3 and Phase 4 sections below.)

Built and verified: Python 3.13 venv with every dependency resolving to a working wheel (and `torch`
confirmed absent), Ollama 0.35.0 with `OLLAMA_MODELS` redirected to `D:\ollama-models`,
`qwen2.5:3b-instruct-q4_K_M` pulled (1.84 GB). `scripts/00_check_env.py` and `scripts/00_smoke_test.py`
both pass; measurements are in `results/smoke_test.json`.

### What the benchmark found

| Passages per prompt | Prompt tokens | End to end |
|---|---|---|
| 4 (the original design) | 1,398 | 48.3 s |
| **3 (chosen)** | **1,081** | **43.5 s** |
| 2 | 762 | 30.3 s |

Measured throughput: **prefill ~31-38 tok/s, decode ~8-10 tok/s.**

**Prefill is ~80% of the cost.** The model spends ~8 s writing an answer and ~35 s merely *reading* the
retrieved passages. This is why `k` was reduced from 4 to 3 rather than dropping to a smaller model:
the bottleneck is prompt length, not model size. Anyone tempted to "fix" latency by shrinking the model
should re-read that sentence first.

**Honest caveat on the margin.** 43.5 s clears the 45 s gate, but run-to-run variance is roughly 15%
(k=4 measured 56.1 s in one session and 48.3 s in another; decode rate swung between 4 and 10 tok/s,
presumably thermal or background load). k=3 is a *marginal* pass, not a comfortable one. If latency
becomes painful in practice, k=2 has real headroom — and `k` lives in `config.yaml` precisely so this
stays a measured tradeoff rather than a constant someone guessed.

### Benchmark pitfall — do not reintroduce

The first version of `00_smoke_test.py` sent an **identical prompt** on every run and reported
**11,540 tok/s** prefill, yielding a confident and completely wrong "PASS". Ollama was replaying the
cached KV state of the prompt prefix rather than computing anything. Every benchmark here must vary the
prompt between runs — `build_prompt()` injects a unique marker at the head for exactly this reason.
Real questions retrieve different passages, so prefill genuinely runs every time.

### Phase 1–2 progress (2026-10-06)

- **Corpus: 3 of 10–12 papers** in `data/papers/` (gitignored), recorded in `data/MANIFEST.csv` with
  DOI, licence and sha256.
- **Extraction done, chunking deliberately not started.** `src/aeration_qa/extract.py` +
  `scripts/01_extract.py` write `cache/extracted/<paper_id>.jsonl` and a `.txt` for reading by eye:
  292 blocks, ~15,600 words. Two-column order, header/footer/watermark removal, ligature repair and the
  references cutoff were all verified by reading the output. `tests/test_extract.py` (9 tests) pins
  the cleaning rules — one of them caught `NH4-\nN` becoming `NH4- N`, which would have split a key
  BM25 token.
- **Known, left for later:** superscript charges detach slightly (`NO3 −`) — the glossary, not the
  extractor, should handle that. Publisher boilerplate (DOI lines, received/accepted dates, copyright)
  is now dropped at extraction.

### Chunking done (2026-10-13) — Gate 2 passed

`src/aeration_qa/chunking.py` + `scripts/02_chunk.py` → `cache/chunks.jsonl`: **125 chunks, median 257
BGE tokens, 0 over the 512 limit, 0 starting mid-sentence**, page numbers spot-checked against the PDFs.
`tests/test_chunking.py` (6 tests; 15 in total). Rules worth knowing before changing it:

- Length is counted with **BGE's own tokenizer** (`load_bge_tokenizer` reads it from the fastembed
  cache). This domain's text runs ~2 tokens per word (units, numbers, `NH4-N`), so word-count estimates
  undercount exactly the chunks that risk truncation.
- The 512 cap applies to **`embed_text` = title + section + text**, so the per-chunk budget shrinks by
  the header's size. A long section title previously pushed two chunks to 519.
- **Tables and figure captions are pulled out of the prose** into their own chunks labelled
  `<section> [tables/figures]` — inline, a table landed in the middle of a sentence. Aside chunks under
  40 tokens (a lone caption, a bare table header) are dropped: they match queries but answer nothing.
- A block that starts lowercase is **rejoined to the previous one** (a sentence broken by a column or
  page break). Chunks never cross a section boundary; overlap is whole sentences (~60 tokens).
- `docs/project_brief.docx` — one-page brief for the professor (dataset + two hypothetical outputs,
  both drawn from what the three papers actually report). Edit it in Word; the script that generated it
  was a throwaway and is not in the repo. The name line still reads "[Your name]".
- Explainer page for presenting the project (RAG in three steps, build plan, retrieval mechanics from
  scratch, Gate 0 results): https://claude.ai/artifact/L6yfhnCBxSGLXXNuVedik1 (private to the user).

### Phase 3 retrieval done (2026-10-13) — Gate 3 passed

`embed.py`, `index.py`, `retrieve.py`; `scripts/03_build_index.py` (125 × 384 matrix, 188 KB, ~28 s) and
`scripts/04_retrieval_check.py` → `results/retrieval_check.md` / `.json`. 19 tests pass. Published results
page for activity logs: https://claude.ai/artifact/NNV8h3CiifuPnUzodfsVry (private until the user shares it).

**Gate 3 result (6 questions, smoke test — no statistical claim):** dense 5/6 in top-3 (MRR@10 0.639),
BM25 3/6 (0.542), hybrid 3/6 (0.582). The plan predicted BM25 would tie or win; here dense leads.

- **The pattern behind BM25's misses is abbreviations,** not general vocabulary mismatch: the question
  says "solids retention time", the paper only says "SRT"; "biofilter / low-strength" vs "BAF / LSPDW".
  That makes `data/glossary.yaml` query expansion a concrete, measurable next experiment.
- **Hybrid is not free:** RRF fusion dragged dense's good ranks down to BM25's level on the abbreviation
  questions. Do not assume hybrid ≥ both.
- **The plan was wrong about fastembed:** in 0.8.1, `query_embed()` does **not** add the BGE prefix — it
  equals `embed()`. `Embedder.embed_query` prepends `config.yaml`'s `query_prefix` itself;
  `tests/test_retrieval.py` guards it.
- **"do" removed from BM25 stopwords** — lowercased it is DO (dissolved oxygen).
- **Disclosed answer-key change:** after the first run, the ammonia-valley question's key was widened to
  accept a p.5 chunk that also states the answer (first run: BM25 2/6, dense 4/6, hybrid 2/6). Recorded
  in the script docstring and the report. The Phase 5 key must be fixed before any run — no such edits.

### 28-question development comparison (2026-10-13)

`data/questions_dev.yaml` (28 questions: 11 shared wording, 10 abbreviation, 7 paraphrase; answer keys
fixed and validated before the first run) → `scripts/05_retrieval_eval.py` → `results/retrieval_eval.md`,
`.json` and two charts in `results/figures/` (matplotlib, Agg backend). The user asked for
**first-glance charts in matplotlib's plain default style** — default font and colors, no custom
styling; chart 1 shows question **counts** on a 0–28 axis, not percentages. A per-question rank grid and a
bootstrap forest plot were tried and dropped as too complex; the bootstrap result lives in the report
text and the JSON instead. Keep new charts at that level of simplicity. `metrics.py` (hit@k, MRR, paired
bootstrap) and `figures.py` are in the library; 24 tests pass.

- Top-3: **dense 64%, hybrid 54%, BM25 43%**. MRR@10 0.577 / 0.487 / 0.450.
- **By question type is the real finding:** BM25 wins on shared wording (73% vs dense 55%) and collapses
  on abbreviations (30% vs 60%) and paraphrases (14% vs 86%). Hybrid tracks BM25, not the best of both.
- **Paired bootstrap 95% intervals all include zero** (dense − BM25 +0.127 [−0.026, +0.285]). Dense being
  better overall is suggestive, not established. Do not report it as a win.
- Two questions defeat every method (q09 hydraulic retention time, q25 "less air help or hurt").
- This is a **development** set — the developer wrote system and questions. Not the Phase 5 evaluation.
- Results page (charts included) for logs: https://claude.ai/artifact/NNV8h3CiifuPnUzodfsVry.

### Glossary experiment (2026-10-27) — no measurable effect

`data/glossary.yaml` (36 terms) + `src/aeration_qa/glossary.py` (`Glossary.expand`, `ExpandedRetriever`)
+ `scripts/06_glossary_eval.py` → `results/glossary_eval.md` / `.json`, `results/figures/3_glossary_effect.png`.
31 tests pass (`tests/test_glossary.py`, 7). Glossary rule, fixed before the first run and stated at the
top of the YAML: every process/chemistry abbreviation a paper defines as "long form (ABBR)", plus SRT
(defined only by a section heading); non-domain labels excluded. Abbreviations match case-sensitively so
the verb "do" never expands. Expansion only appends the missing form.

- Expansion changed 13 of 28 questions. Top-3: **BM25 12→13, dense 18→19, hybrid 15→16**; MRR moved by
  < 0.01 for every method, all bootstrap intervals at or across zero (dense's [+0.000, +0.009] is a single
  question moving rank 4→3). **Headline: the glossary does not fix BM25.**
- **Why:** the remaining abbreviation misses are not parenthetical abbreviations. q08 is *ammonium* vs
  the paper's *NH4+* (chemical symbol, excluded by the rule); q02/q09 say *biofilter* where the paper says
  *BAF*. q04 (SRT) and q09 (HRT) went from not found to rank 5–6: closer, still outside top-3.
- **Expansion has a cost:** q03 dropped BM25 rank 1→3 because SBR's three spelling variants added
  "batch reactor" three times and outweighed the question's own words.
- Do **not** now add chemical symbols or "biofilter" to the glossary and rerun: that is tuning on the
  development set after seeing it. If wanted, pre-register it as a Phase 5 condition.

### Phase 4 generation built (2026-10-27) — Gate 4 run, correctness not yet hand-checked

`src/aeration_qa/generate.py` (prompt, Ollama call, citation check), `pipeline.py` (`Pipeline.ask` →
`Answer`; `LLMStrategy` and `ExtractiveStrategy` are swappable), `ask.py` at the repo root
(`--retriever`, `--extractive`, `--show-prompt`), `scripts/07_answer_check.py` →
`results/answer_check.md` / `.json`. 36 tests pass. Default path: dense retrieval, k=3.

- **Gate 4 (12 questions: every third dev question + 2 unanswerable):** genuine timings median
  **31.5 s**, worst 40.9 s (6 of 12; the rest were cache-served, see below). Gold passage sent 7/10.
  Both unanswerable questions → `NOT_IN_CORPUS`; q07 also abstained, correctly, because retrieval missed
  its passage. 0 invalid citations. **3 answers carry no citation at all** (q04, q25, q28) — the Sources
  block still shows where they came from. The *Correct?* column in the report is for the user to fill in.
- **Bug found and fixed: the papers' own reference markers.** 31 of 125 chunks contain `[1]`, `[11–14]`…
  The model copied "chloramines [1]" from excerpt [3], and the validator accepted [1] as a valid citation.
  `number_sources` now strips them from prompt text (`strip_paper_references`); the retrieval index is
  untouched, so all retrieval results stand. `test_papers_own_reference_markers_never_reach_the_prompt`.
- **Prefix caching is partial, not all-or-nothing.** Beyond exact repeats (5,000+ tok/s), Ollama 0.40
  reuses a shared prompt *prefix*: runs at 63–75 tok/s were partial hits. `07_` flags any prefill above
  55 tok/s as cached and excludes it from timing. Real prefill here: ~30–50 tok/s.
- Extractive mode (`--extractive`) answers in < 1 s; neighbouring-chunk overlap made it quote one
  sentence twice, so sentences are de-duplicated.

### Next

1. **User hand-checks `results/answer_check.md`** (fill in *Correct?*), then decide whether uncited
   answers need a prompt change. Any prompt change must be judged on new questions, not these 12.
2. In parallel, the user collects the remaining 7–9 papers → `MANIFEST.csv` → rerun `01_extract.py`,
   `02_chunk.py`, `03_build_index.py`, then `05_` and `06_` to see if the findings hold on more papers.
3. The evaluation harness is **deliberately deferred** — a working pipeline comes first, then its scope
   gets sized against something real. Note that at ~43 s per answer, a ~300-call evaluation sweep is a
   ~3.5 hour overnight run; the retrieval half needs no model at all and runs in seconds.

The full technical plan lives in this repo at [docs/PLAN.md](docs/PLAN.md) — it covers the annotation
protocol and pre-registration discipline for phase 5, and a phase 6 extension that folds the paused
bending-point detector back in as a second information source (with the rule that application code,
never the model, decides when to run it — small local models are unreliable at function calling).
