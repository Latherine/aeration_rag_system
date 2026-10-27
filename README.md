# Aeration QA

A small retrieval-augmented question-answering system for wastewater-treatment aeration control. Ask a
question in plain English; it retrieves passages from a corpus of research papers and answers with
citations to paper, page and section. Everything runs locally on a CPU-only laptop: no paid API, no GPU,
no model training.

```
python ask.py "What does the ammonia valley in the pH profile indicate?"

The ammonia valley in the pH profile indicates the completion of a nitrification process. [1]

Sources (attached by code, not written by the model):
 *[1] Recognition of Relevant ORP, pH, and DO Bending Points ... (2010), pp.5-6, 3. Results and Discussion
```

## How it works

1. **Extract** text from two-column PDFs with PyMuPDF, drop headers, footers and the references section.
2. **Chunk** into ~300-token, sentence-aligned passages (counted with the embedding model's own tokenizer).
3. **Retrieve** with BM25 keyword search, BGE-small dense embeddings, or a reciprocal-rank-fusion hybrid,
   all written by hand (no LangChain or LlamaIndex) so the retrieval mechanics can be measured.
4. **Generate** with `qwen2.5:3b-instruct` (Q4) through Ollama. The Sources list is built by code from chunk
   metadata, and every `[n]` the model writes is checked against the excerpts actually sent.
   `--extractive` quotes the most relevant sentences instead, without any language model.

## Results so far (development set, 3 papers, 28 questions)

| | Correct passage in top 3 |
|---|---|
| BM25 | 12 / 28 |
| Dense (BGE-small) | 18 / 28 |
| Hybrid (RRF) | 15 / 28 |

All paired bootstrap 95% intervals on the differences include zero, so no method is established as better.
BM25 wins when questions use the paper's own wording and loses on abbreviations and paraphrases. Glossary
query expansion had no measurable effect. End-to-end answers take ~30 s on an i5-1235U. Details:
[`results/retrieval_eval.md`](results/retrieval_eval.md), [`results/glossary_eval.md`](results/glossary_eval.md),
[`results/answer_check.md`](results/answer_check.md). The design and evaluation plan is in
[`docs/PLAN.md`](docs/PLAN.md).

## Running it

Windows, Python 3.13, [Ollama](https://ollama.com) with `ollama pull qwen2.5:3b-instruct-q4_K_M`.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\01_extract.py
.venv\Scripts\python.exe scripts\02_chunk.py
.venv\Scripts\python.exe scripts\03_build_index.py
.venv\Scripts\python.exe ask.py "your question"
.venv\Scripts\python.exe -m pytest -q
```

## The papers are not included

The PDFs are copyrighted and are not in this repository. [`data/MANIFEST.csv`](data/MANIFEST.csv) lists
each paper's DOI and a sha256 of the file, so the same corpus can be obtained and verified. Place the PDFs
in `data/papers/` under the filenames given there.

## Licence: AGPL-3.0, because of PyMuPDF

This code is released under the [GNU AGPL-3.0](LICENSE). PDF extraction uses
[PyMuPDF](https://pymupdf.readthedocs.io), which is itself AGPL-3.0, and publishing code built on it under
the same licence is what the AGPL requires. PyMuPDF was chosen deliberately: it is the only extractor tried
that gives block bounding boxes and font sizes, which are needed to put two-column papers back in reading
order. Anyone building on this code inherits the AGPL's obligations, including sharing source when the
software is offered to others over a network.

The licence covers the code only. The papers remain under their publishers' copyright (see above).
