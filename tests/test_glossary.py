from aeration_qa.config import load_config
from aeration_qa.glossary import ExpandedRetriever, Glossary
from aeration_qa.retrieve import BM25Retriever

TERMS = [
    {"abbr": "SRT", "long": ["solids retention time"]},
    {"abbr": "DO", "long": ["dissolved oxygen"]},
    {"abbr": "PAO", "long": ["phosphate-accumulating organism"]},
]


def test_long_form_adds_abbreviation_and_keeps_query():
    query = "What solids retention time is recommended?"
    assert Glossary(TERMS).expand(query) == query + " SRT"


def test_abbreviation_adds_long_form():
    assert Glossary(TERMS).expand("Target SRT?") == "Target SRT? solids retention time"


def test_verb_do_is_not_dissolved_oxygen():
    assert Glossary(TERMS).expand("How do operators set the DO?") == "How do operators set the DO? dissolved oxygen"
    assert Glossary(TERMS).expand("What do plants do?") == "What do plants do?"


def test_plural_and_hyphen_variants_match():
    g = Glossary(TERMS)
    assert g.expand("Role of phosphate accumulating organisms").endswith(" PAO")
    assert g.expand("Two SRTs compared") == "Two SRTs compared solids retention time"


def test_nothing_added_when_both_forms_present():
    assert Glossary(TERMS).expand("dissolved oxygen (DO)") == "dissolved oxygen (DO)"


def test_expansion_lets_bm25_find_abbreviation_only_chunk():
    bm25 = BM25Retriever(["a", "b"], ["Target an SRT of 12 days.", "Retention of solids in the clarifier."])
    expanded = ExpandedRetriever(bm25, Glossary(TERMS))
    assert expanded.name == "bm25+glossary"
    assert expanded.search("recommended solids retention time", 1)[0].chunk_id == "a"


def test_project_glossary_loads_and_is_unambiguous():
    g = Glossary.load(load_config().glossary)
    abbrs = [t.abbr for t in g.terms]
    assert len(abbrs) == len(set(abbrs))
    assert "SRT" in abbrs and "DO" in abbrs
