import pytest
from src.model.model_input import ModelInputGenerator, ModelInputError
from src.data.window_builder import QAWindowRecord, QAWindowAnswer

@pytest.fixture
def valid_window():
    return QAWindowRecord(
        example_id="doc1_chunk1_ans1_win0",
        document_id="doc1",
        chunk_id="doc1_chunk1",
        window_id="doc1_chunk1_win0",
        split="train",
        annotation_id="anno1",
        category="Governing Law",
        question="What is the governing law?",
        context="This agreement is governed by the laws of New York.",
        input_ids=[101, 2054, 2003, 1996, 7352, 2112, 1029, 102, 2023, 4475, 2003, 3134, 2011, 1996, 3112, 1997, 2047, 2259, 1012, 102],
        attention_mask=[1] * 20,
        offset_mapping=[
            [0, 0], [0, 4], [5, 7], [8, 11], [12, 21], [22, 25], [25, 26], [0, 0], 
            [0, 4], [5, 14], [15, 17], [18, 26], [27, 29], [30, 33], [34, 38], [39, 41], [42, 45], [46, 50], [50, 51], [0, 0]
        ],
        answer=QAWindowAnswer(
            text="New York",
            char_start=42,
            char_end=50,
            original_char_start=42,
            original_char_end=50,
            token_start=16,
            token_end=17
        )
    )

def test_valid_start_end_positions(valid_window):
    generator = ModelInputGenerator()
    example = generator.generate_and_validate(valid_window)
    assert example.start_positions == 16
    assert example.end_positions == 17
    assert example.document_id == "doc1"
    assert example.split == "train"

def test_invalid_start_position(valid_window):
    valid_window.answer.token_start = -1
    generator = ModelInputGenerator()
    with pytest.raises(ModelInputError, match="start_positions.*< 0"):
        generator.generate_and_validate(valid_window)

def test_invalid_end_position(valid_window):
    valid_window.answer.token_end = len(valid_window.input_ids)
    generator = ModelInputGenerator()
    with pytest.raises(ModelInputError, match="end_positions.*>= sequence length"):
        generator.generate_and_validate(valid_window)

def test_start_greater_than_end(valid_window):
    valid_window.answer.token_start = 17
    valid_window.answer.token_end = 16
    generator = ModelInputGenerator()
    with pytest.raises(ModelInputError, match="start_positions.*> end_positions"):
        generator.generate_and_validate(valid_window)

def test_special_token_rejection(valid_window):
    valid_window.answer.token_start = 0  # CLS token with mapping [0, 0]
    # To trigger rejection, we must make sure char_start != char_end
    # so the generator knows it wasn't INTENTIONALLY a zero-span (e.g. for impossible answers)
    generator = ModelInputGenerator()
    with pytest.raises(ModelInputError, match="special token"):
        generator.generate_and_validate(valid_window)

def test_deterministic_generation(valid_window):
    generator = ModelInputGenerator()
    ex1 = generator.generate_and_validate(valid_window)
    ex2 = generator.generate_and_validate(valid_window)
    assert ex1.model_dump() == ex2.model_dump()
