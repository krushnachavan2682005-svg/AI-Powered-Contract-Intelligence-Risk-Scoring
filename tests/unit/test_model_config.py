import pytest
from src.model.model_config import ModelConfig

def test_model_config_defaults():
    config = ModelConfig()
    assert config.model_name == "distilbert-base-uncased"
    assert config.tokenizer_name == "distilbert-base-uncased"
    assert config.max_seq_length == 512
    assert config.learning_rate == 3e-5
    assert config.batch_size == 8
    assert config.eval_batch_size == 8
    assert config.num_epochs == 3
    assert config.weight_decay == 0.01
    assert config.warmup_ratio == 0.1
    assert config.gradient_accumulation_steps == 1
    assert config.seed == 42
    assert config.output_dir == "artifacts/model_output"

def test_model_config_validation():
    # Test valid overrides
    config = ModelConfig(batch_size=16, learning_rate=5e-5)
    assert config.batch_size == 16
    assert config.learning_rate == 5e-5
    
    # Validation errors for invalid types
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ModelConfig(batch_size="invalid_size")
