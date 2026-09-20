import pytest

from src.data.chunker import ChunkAnswerRecord, ChunkRecord
from src.data.window_builder import QAWindowBuilder
from src.data.window_config import WindowConfig


def test_window_builder_short_sequence():
    config = WindowConfig(max_seq_length=512, doc_stride=128)
    builder = QAWindowBuilder(config)
    
    chunk = ChunkRecord(
        document_id="doc_1",
        chunk_id="chunk_0",
        chunk_start=0,
        chunk_end=100,
        chunk_text="This is a short contract. The term is 5 years.",
        answers=[
            ChunkAnswerRecord(
                annotation_id="ann_1",
                category="Term",
                answer_text="5 years",
                original_start=38,
                original_end=45,
                local_start=38,
                local_end=45
            )
        ]
    )
    
    windows = builder.build_windows_for_chunk(chunk, "train")
    
    assert len(windows) == 1
    assert windows[0].answer.text == "5 years"
    assert windows[0].split == "train"


def test_window_builder_long_sequence():
    config = WindowConfig(max_seq_length=100, doc_stride=10) # short max length to force overflow
    builder = QAWindowBuilder(config)
    
    # Text is long enough to exceed 20 tokens
    text = "word " * 50
    # Answer at the end of the text
    ans_text = "word word"
    local_start = text.rfind(ans_text)
    local_end = local_start + len(ans_text)
    
    chunk = ChunkRecord(
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
                original_end=local_end,
                local_start=local_start,
                local_end=local_end
            )
        ]
    )
    
    windows = builder.build_windows_for_chunk(chunk, "train")
    
    # Should produce multiple windows but only one valid positive window
    assert len(windows) == 1
    
    win = windows[0]
    assert win.answer.text == ans_text
    assert len(win.input_ids) <= 100
