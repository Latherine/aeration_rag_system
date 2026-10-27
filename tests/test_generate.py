from aeration_qa.generate import NOT_IN_CORPUS, build_prompt, check_citations, number_sources
from aeration_qa.pipeline import Draft, Pipeline
from aeration_qa.retrieve import Hit


def _chunk(i: int) -> dict:
    return {"chunk_id": f"p:{i:03d}", "paper_id": "p", "title": "A paper", "section": "2. Methods",
            "page_start": i, "page_end": i, "text": f"Passage number {i}."}


def test_best_chunk_is_numbered_last_and_sits_next_to_question():
    sources = number_sources([_chunk(0), _chunk(1), _chunk(2)], {"p": "2010"})  # best first
    assert [s.number for s in sources] == [1, 2, 3]
    assert [s.retrieval_rank for s in sources] == [3, 2, 1]
    prompt = build_prompt("Why?", sources)
    assert prompt.index("Passage number 0.") > prompt.index("Passage number 2.")
    assert prompt.startswith("[1] A paper (2010), p.2\n") and prompt.endswith("Question: Why?")


def test_papers_own_reference_markers_never_reach_the_prompt():
    """A copied "[1]" from the paper's text would pass as a valid citation of excerpt [1]."""
    chunk = _chunk(0) | {"text": "Ammonia yields chloramines [1], nerve damage [2, 3] and odour [11–14]."}
    assert number_sources([chunk], {})[0].text == "Ammonia yields chloramines, nerve damage and odour."


def test_citation_formats_and_out_of_range_markers():
    assert check_citations("A [1]. B [2, 3]. C [1][3].", 3) == ([1, 2, 3], [])
    assert check_citations("Range [1-3].", 3) == ([1, 2, 3], [])
    assert check_citations("Invented [4] and [7].", 3) == ([], [4, 7])
    assert check_citations("No citations, see Table 2 or [a].", 3) == ([], [])


class _FakeRetriever:
    name = "fake"

    def search(self, query, k):
        return [Hit(f"p:{i:03d}", 1.0, i + 1) for i in range(k)]


class _FakeStrategy:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def answer(self, question, sources):
        return Draft(self.text)


def test_pipeline_attaches_sources_even_when_model_cites_nothing():
    chunks = [_chunk(i) for i in range(5)]
    answer = Pipeline(_FakeRetriever(), _FakeStrategy("No citation here."), chunks, {"p": "2010"}, k=3).ask("Q?")
    assert len(answer.sources) == 3 and answer.cited == [] and not answer.abstained
    assert "<p:000>" in answer.render()


def test_pipeline_flags_abstention_and_invented_citations():
    chunks = [_chunk(i) for i in range(5)]
    abstain = Pipeline(_FakeRetriever(), _FakeStrategy(NOT_IN_CORPUS), chunks, {}, k=2).ask("Q?")
    assert abstain.abstained
    invented = Pipeline(_FakeRetriever(), _FakeStrategy("Claim [5]."), chunks, {}, k=2).ask("Q?")
    assert invented.invalid_citations == [5] and "WARNING" in invented.render()
