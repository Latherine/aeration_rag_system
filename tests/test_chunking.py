from aeration_qa.chunking import chunk_paper, is_aside, split_sentences


def words(text: str) -> int:
    """Stand-in token counter: one token per word keeps the tests readable."""
    return len(text.split())


def sentence(i: int, n_words: int = 10) -> str:
    return f"Sentence {i} " + " ".join(["word"] * (n_words - 3)) + " end."


def test_split_keeps_abbreviations_and_decimals_inside_sentences():
    text = "Removal reached 87.1 % (Fig. 3). Smith et al. (2019) agreed, e.g. at pH 7.0. A new sentence."
    assert split_sentences(text) == [
        "Removal reached 87.1 % (Fig. 3).",
        "Smith et al. (2019) agreed, e.g. at pH 7.0.",
        "A new sentence.",
    ]


def test_chunks_respect_target_and_never_cross_sections():
    blocks = [{"section": "Intro", "page": 1, "text": " ".join(sentence(i) for i in range(12))},
              {"section": "Methods", "page": 2, "text": " ".join(sentence(i) for i in range(12, 24))}]
    chunks = chunk_paper(blocks, "p", "T", words, target=40, overlap=10, max_tokens=60)
    assert all(c.n_tokens <= 60 for c in chunks)
    assert {c.section for c in chunks} == {"Intro", "Methods"}
    for c in chunks:
        numbers = [int(w) for w in c.text.split() if w.isdigit()]
        assert all((n < 12) == (c.section == "Intro") for n in numbers)


def test_consecutive_chunks_overlap_by_whole_sentences():
    blocks = [{"section": "S", "page": 1, "text": " ".join(sentence(i) for i in range(12))}]
    first, second, *_ = chunk_paper(blocks, "p", "T", words, target=40, overlap=10, max_tokens=60)
    last_sentence_of_first = split_sentences(first.text)[-1]
    assert second.text.startswith(last_sentence_of_first)


def test_block_continuing_across_a_column_break_is_rejoined():
    blocks = [{"section": "S", "page": 1, "text": "The DO concentration rises sharply once"},
              {"section": "S", "page": 1, "text": "ammonium is depleted, producing an elbow."}]
    [chunk] = chunk_paper(blocks, "p", "T", words, target=300, overlap=60, max_tokens=480)
    assert "rises sharply once ammonium is depleted" in chunk.text


def test_tables_are_kept_out_of_prose():
    table = "C1 58.8 14.6 89.0 C2 74.8 80.7 91.0 C3 64.5 87.1 93.0 C4 42.8 8.7 97.5 C5 77.2 14.5 94.0"
    assert is_aside(table)
    assert is_aside("Fig. 2. Effect of SRT on ammonium.")
    assert not is_aside("Aeration energy fell by 21% while removal stayed above 75% in most runs.")

    blocks = [{"section": "S", "page": 1, "text": "Several configurations with dedicated"},
              {"section": "S", "page": 1, "text": table * 3},
              {"section": "S", "page": 1, "text": "anoxic stages achieved strong removal."}]
    chunks = chunk_paper(blocks, "p", "T", words, target=300, overlap=60, max_tokens=480)
    prose = [c for c in chunks if c.section == "S"]
    assert prose[0].text == "Several configurations with dedicated anoxic stages achieved strong removal."


def test_header_counts_toward_the_512_limit():
    long_section = " ".join(["Heading"] * 200)
    blocks = [{"section": long_section, "page": 1, "text": " ".join(sentence(i) for i in range(60))}]
    chunks = chunk_paper(blocks, "p", "T", words, target=300, overlap=60, max_tokens=480)
    assert all(words(c.embed_text) <= 512 for c in chunks)
