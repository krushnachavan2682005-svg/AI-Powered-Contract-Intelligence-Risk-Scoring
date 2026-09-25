import pytest
import torch
from src.model.model_factory import ModelFactory
from src.model.model_config import ModelConfig
from transformers import PreTrainedTokenizerFast

@pytest.fixture
def factory():
    config = ModelConfig(model_name="distilbert-base-uncased", tokenizer_name="distilbert-base-uncased")
    return ModelFactory(config)

def test_tokenizer_model_compatibility(factory):
    model, tokenizer = factory.load()
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert model.config.name_or_path == factory.config.model_name
    
def test_smoke_test(factory):
    model = factory.load_model()
    # Create fake inputs
    input_ids = torch.tensor([[101, 2023, 2003, 1037, 3350, 102]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    
    success = factory.smoke_test(model, input_ids, attention_mask)
    assert success is True

def test_load_invalid_tokenizer():
    config = ModelConfig(tokenizer_name="invalid-tokenizer-name-xyz123")
    factory = ModelFactory(config)
    with pytest.raises(RuntimeError):
        factory.load_tokenizer()
