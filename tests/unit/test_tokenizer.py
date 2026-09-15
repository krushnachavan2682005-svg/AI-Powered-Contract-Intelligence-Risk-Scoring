import pytest

from src.data.tokenizer import TokenizerConfig, TokenizerWrapper


@pytest.fixture(scope="module")
def tokenizer_wrapper():
    return TokenizerWrapper(TokenizerConfig(model_name="distilbert-base-uncased"))


def test_tokenizer_initialization(tokenizer_wrapper):
    assert tokenizer_wrapper.is_fast is True
    assert tokenizer_wrapper.vocab_size > 0


def test_tokenize_chunk_no_truncation(tokenizer_wrapper):
    text = "This is a simple contract clause."
    encoding = tokenizer_wrapper.tokenize_chunk(text)

    assert "input_ids" in encoding
    assert "attention_mask" in encoding
    assert "offset_mapping" in encoding
    assert len(encoding["input_ids"]) > 0
    assert len(encoding["input_ids"]) == len(encoding["offset_mapping"])


def test_map_character_to_token(tokenizer_wrapper):
    text = "The effective date of this agreement is January 1, 2023."
    # Let's say we want to map "January 1, 2023"
    char_start = text.find("January")
    char_end = char_start + len("January 1, 2023")

    encoding = tokenizer_wrapper.tokenize_chunk(text)
    offset_mapping = encoding["offset_mapping"]

    token_start, token_end = tokenizer_wrapper.map_character_to_token(
        char_start, char_end, offset_mapping
    )

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end

    # Reconstruct
    recon_start = offset_mapping[token_start][0]
    recon_end = offset_mapping[token_end][1]

    # The reconstructed span might include some whitespace depending on tokenizer,
    # but the offsets must encompass the answer
    assert recon_start <= char_start
    assert recon_end >= char_end
