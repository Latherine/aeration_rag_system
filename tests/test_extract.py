from aeration_qa.extract import clean_text, is_heading, order_blocks


def test_ligatures_become_plain_letters():
    assert clean_text("eﬄuent and inﬂuent") == "effluent and influent"


def test_soft_hyphen_at_line_end_rejoins_word():
    assert clean_text("operation\xad\nalizes") == "operationalizes"


def test_line_break_hyphen_rejoins_lowercase_word():
    assert clean_text("denitri-\nfication") == "denitrification"


def test_real_compound_hyphen_survives_line_break():
    assert clean_text("NH4-\nN removal") == "NH4-N removal"


def test_control_characters_removed():
    assert clean_text("ISSN: 2307-549X\x08") == "ISSN: 2307-549X"


def test_numbered_and_roman_headings_detected():
    assert is_heading("2.1. Study sites and SBR configuration")
    assert is_heading("III. Results and discussion")
    assert is_heading("A. Dynamic Simulation of Duhok SBRs with no EQ Basin Aeration (First Run)")


def test_body_text_not_mistaken_for_heading():
    body = (
        "3.1. Track Study 1 (TS1). Figure 3 shows the profiles of TS1 and indicates that, "
        "for the 50 mg/L NH4+ fed at constant aeration, complete removal occurred."
    )
    assert not is_heading(body)


def test_two_column_page_reads_left_column_before_right():
    # (x0, y0, x1, y1, text): right column starts higher on the page than the
    # left column's second block, so naive y-sorting would interleave them.
    width = 600
    blocks = [
        (40, 100, 290, 150, "left-1"),
        (320, 90, 560, 140, "right-1"),
        (40, 200, 290, 250, "left-2"),
        (320, 210, 560, 260, "right-2"),
    ]
    assert [b[4] for b in order_blocks(blocks, width)] == ["left-1", "left-2", "right-1", "right-2"]


def test_full_width_block_splits_columns_into_bands():
    width = 600
    blocks = [
        (40, 100, 290, 150, "left-above"),
        (320, 100, 560, 150, "right-above"),
        (40, 300, 560, 340, "full-width-table"),
        (40, 400, 290, 450, "left-below"),
        (320, 400, 560, 450, "right-below"),
    ]
    assert [b[4] for b in order_blocks(blocks, width)] == [
        "left-above", "right-above", "full-width-table", "left-below", "right-below",
    ]
