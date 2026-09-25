from src.data.chunker import ChunkAnswerRecord, ChunkRecord
from src.data.window_builder import QAWindowBuilder
from src.data.window_config import WindowConfig


def _create_chunk(text, ans_text, local_start):
    return ChunkRecord(
        document_id="doc_1",
        chunk_id="chunk_0",
        chunk_start=0,
        chunk_end=len(text),
        chunk_text=text,
        answers=[
            ChunkAnswerRecord(
                annotation_id="ann_1",
                category="Term",
                answer_text=ans_text,
                original_start=local_start,
                original_end=local_start + len(ans_text),
                local_start=local_start,
                local_end=local_start + len(ans_text),
            )
        ],
    )


def test_answer_fully_inside_window():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=512, doc_stride=128))
    chunk = _create_chunk(
        "This is a short contract. The term is 5 years.", "5 years", 38
    )
    windows = builder.build_windows_for_chunk(chunk, "train")
    assert len(windows) == 1
    assert windows[0].answer.text == "5 years"


def test_answer_overlaps_whitespace_boundaries():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=512, doc_stride=128))
    # Answer text has leading space but tokenizer drops it
    text = "This contract   is for 5 years."
    ans_text = "  is for 5 years"
    local_start = text.index(ans_text)
    chunk = _create_chunk(text, ans_text, local_start)
    windows = builder.build_windows_for_chunk(chunk, "train")
    assert len(windows) == 1
    assert "is for 5 years" in windows[0].answer.text


def test_partial_answer_must_never_count_as_coverage():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=60, doc_stride=10))
    text = "A " * 60
    ans_text = (
        "A " * 40
    )  # 40 tokens is too big for the budget context if not centered properly, wait actually it might be recoverable if we center it. But if we make it 60 tokens, it's impossible.
    # Actually this tests #5: partial answer must never count as coverage.
    chunk = _create_chunk(text, ans_text.strip(), 0)
    windows = builder.build_windows_for_chunk(chunk, "train")
    # Must NOT count as covered if it's partially in window
    # Wait, it WILL be recovered if it fits! 40 tokens fits in 60 - ~26 = 34? No, 40 > 34. So it will NOT be covered.
    assert len(windows) == 0


def test_answer_crossing_existing_boundary_recoverable_by_shifting():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=60, doc_stride=1))
    # Standard window is 60 tokens, stride 1 means 1 token overlap.
    # A 15-token answer that straddles the mark.
    text = "word " * 60
    ans_text = "word " * 15
    local_start = text.index("word ") + 5 * 30  # approx 30 tokens in
    chunk = _create_chunk(text, ans_text.strip(), local_start)
    windows = builder.build_windows_for_chunk(chunk, "train")
    # With shifted recovery, we should get exactly 1 recovered window
    # Actually it might just be covered in one of the standard windows if we're not careful.
    # Let's verify it is covered.
    assert len(windows) >= 1
    assert any(ans_text.strip() in w.answer.text for w in windows)
    assert any("recovered" in w.window_id for w in windows)


def test_answer_exactly_at_shifted_window_boundary():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=40, doc_stride=5))
    text = "word " * 60
    ans_text = "word " * 10
    local_start = 0
    chunk = _create_chunk(text, ans_text.strip(), local_start)
    windows = builder.build_windows_for_chunk(chunk, "train")
    assert any(ans_text.strip() in w.answer.text for w in windows)


def test_answer_requiring_maximum_allowable_context_budget():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=40, doc_stride=10))
    text = "word " * 50
    # The question is ~23 tokens, special tokens ~3. Budget ~14 tokens.
    ans_text = "word " * 12  # 12 tokens should barely fit
    chunk = _create_chunk(text, ans_text.strip(), 20)
    windows = builder.build_windows_for_chunk(chunk, "train")
    assert len(windows) >= 1


def test_answer_still_impossible_because_budget():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=40, doc_stride=10))
    text = "word " * 50
    # Budget is ~14, answer is 20 tokens.
    ans_text = "word " * 20
    chunk = _create_chunk(text, ans_text.strip(), 10)
    windows = builder.build_windows_for_chunk(chunk, "train")
    assert len(windows) == 0


def test_duplicate_recovery_windows_do_not_inflate():
    builder = QAWindowBuilder(WindowConfig(max_seq_length=40, doc_stride=10))
    text = "word " * 60
    ans_text = "word " * 15
    local_start = text.index("word ") + 5 * 30
    chunk = _create_chunk(text, ans_text.strip(), local_start)
    windows1 = builder.build_windows_for_chunk(chunk, "train")
    windows2 = builder.build_windows_for_chunk(chunk, "train")
    # Determinism
    assert [w.window_id for w in windows1] == [w.window_id for w in windows2]
