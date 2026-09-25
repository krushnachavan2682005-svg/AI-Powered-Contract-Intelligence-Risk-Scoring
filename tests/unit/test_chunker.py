import pytest

from src.data.chunker import AnswerAwareChunker, ChunkingConfig
from src.data.legal_preprocessor import ProcessedContractRecord
from src.data.schemas import AnswerRecord, ClauseRecord


def test_short_document_creates_one_chunk():
    config = ChunkingConfig(chunk_size=100, overlap=20)
    chunker = AnswerAwareChunker(config)
    record = ProcessedContractRecord(
        document_id="doc1",
        title="Title",
        original_context="This is a short document.",
        clauses=[],
    )
    chunks = chunker.chunk_contract(record)
    assert len(chunks) == 1
    assert chunks[0].chunk_end == len("This is a short document.")


def test_long_document_creates_multiple_chunks():
    config = ChunkingConfig(chunk_size=10, overlap=5)
    chunker = AnswerAwareChunker(config)
    record = ProcessedContractRecord(
        document_id="doc2",
        title="Title",
        original_context="012345678901234567890123456789",  # 30 chars
        clauses=[],
    )
    chunks = chunker.chunk_contract(record)
    assert len(chunks) > 1
    assert chunks[-1].chunk_end == 30


def test_overlap_is_correct():
    config = ChunkingConfig(chunk_size=10, overlap=5)
    chunker = AnswerAwareChunker(config)
    record = ProcessedContractRecord(
        document_id="doc3",
        title="Title",
        original_context="01234567890123456789",  # 20 chars
        clauses=[],
    )
    chunks = chunker.chunk_contract(record)
    assert chunks[0].chunk_start == 0
    assert chunks[0].chunk_end == 10
    assert chunks[1].chunk_start == 5
    assert chunks[1].chunk_end == 15
    assert chunks[2].chunk_start == 10
    assert chunks[2].chunk_end == 20


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, overlap=10)
        AnswerAwareChunker(ChunkingConfig(chunk_size=10, overlap=15))


def test_answer_fully_inside_chunk_maps_correctly():
    config = ChunkingConfig(chunk_size=100, overlap=20)
    chunker = AnswerAwareChunker(config)
    ans = AnswerRecord(text="answer", start=10, end=16)
    clause = ClauseRecord(
        annotation_id="a1", category="Cat", is_present=True, answers=[ans]
    )
    record = ProcessedContractRecord(
        document_id="doc5",
        title="Title",
        original_context="0123456789answer0123456789",
        clauses=[clause],
    )
    chunks = chunker.chunk_contract(record)
    assert len(chunks[0].answers) == 1
    ca = chunks[0].answers[0]
    assert ca.answer_text == "answer"
    assert ca.local_start == 10
    assert ca.local_end == 16


def test_answer_crossing_boundary_recovered_by_overlap():
    config = ChunkingConfig(chunk_size=10, overlap=6)
    chunker = AnswerAwareChunker(config)
    # 0123456789 (chunk 1) ends at 10.
    # answer is from 8 to 14: "89abcd" -> 6 chars
    # Chunk 1 won't contain it because it ends at 14.
    # Wait, dynamic expansion might expand it!
    # Because answer starts at 8, which is < 10, dynamic expansion
    # will expand chunk 1 to end at 14!
    ans = AnswerRecord(text="89abcd", start=8, end=14)
    clause = ClauseRecord(
        annotation_id="a2", category="Cat", is_present=True, answers=[ans]
    )
    record = ProcessedContractRecord(
        document_id="doc6",
        title="Title",
        original_context="0123456789abcdefghij",
        clauses=[clause],
    )
    chunks = chunker.chunk_contract(record)

    # Due to dynamic expansion, chunk 0 will expand to end at 14
    assert chunks[0].chunk_end == 14
    assert len(chunks[0].answers) == 1
    assert chunks[0].answers[0].answer_text == "89abcd"


def test_answer_larger_than_chunk_size_handled_safely():
    config = ChunkingConfig(chunk_size=10, overlap=5)
    chunker = AnswerAwareChunker(config)
    ans = AnswerRecord(text="012345678901234", start=0, end=15)
    clause = ClauseRecord(
        annotation_id="a3", category="Cat", is_present=True, answers=[ans]
    )
    record = ProcessedContractRecord(
        document_id="doc7",
        title="Title",
        original_context="01234567890123456789",
        clauses=[clause],
    )
    chunks = chunker.chunk_contract(record)
    # The first chunk will expand to 15 to include the oversized answer
    assert chunks[0].chunk_end == 15
    assert len(chunks[0].answers) == 1
    assert chunks[0].answers[0].answer_text == "012345678901234"


def test_empty_context_fails():
    chunker = AnswerAwareChunker()
    record = ProcessedContractRecord(
        document_id="d8", title="T", original_context="", clauses=[]
    )
    with pytest.raises(ValueError):
        chunker.chunk_contract(record)


def test_impossible_clauses_no_duplication():
    config = ChunkingConfig(chunk_size=100, overlap=10)
    chunker = AnswerAwareChunker(config)
    clause = ClauseRecord(
        annotation_id="a4", category="Cat", is_present=False, answers=[]
    )
    record = ProcessedContractRecord(
        document_id="d9", title="T", original_context="0123456789", clauses=[clause]
    )
    chunks = chunker.chunk_contract(record)
    assert len(chunks) == 1
    assert len(chunks[0].answers) == 0
