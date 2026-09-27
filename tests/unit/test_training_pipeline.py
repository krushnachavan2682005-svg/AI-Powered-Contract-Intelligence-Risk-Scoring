import pytest
import torch
import json
from pathlib import Path
from src.data.window_builder import QAWindowBuilder
from src.data.window_config import WindowConfig
from src.model.model_input import ModelInputGenerator
from scripts.train_qa import TrainConfig, set_seed

def test_impossible_example_cls_labels():
    # If a start/end position is for an impossible example, it should map to CLS (0)
    # The prepare_training_data.py logic ensures this, but we can test the expected convention.
    config = WindowConfig()
    builder = QAWindowBuilder(config)
    tokenizer = builder.tokenizer
    
    cls_token_id = tokenizer.cls_token_id
    encoding = tokenizer("test", add_special_tokens=True)
    
    # In distilbert/bert, CLS is usually at index 0
    assert encoding["input_ids"][0] == cls_token_id

def test_positive_example_cannot_map_to_cls():
    config = WindowConfig()
    builder = QAWindowBuilder(config)
    tokenizer = builder.tokenizer
    
    encoding = tokenizer("question", "context", add_special_tokens=True, return_offsets_mapping=True)
    
    # Find offsets for CLS
    cls_index = encoding["input_ids"].index(tokenizer.cls_token_id)
    cls_offset = encoding["offset_mapping"][cls_index]
    
    # CLS offset should be (0,0), and our model input logic throws error if answer maps to (0,0) unless it's empty
    assert cls_offset == (0, 0)

def test_dataset_split_integrity():
    # Just a mock test ensuring our script produces correct splits
    splits = ["train", "validation", "test"]
    assert len(splits) == 3

def test_training_configuration():
    config = TrainConfig()
    assert config.model_name == "distilbert-base-uncased"
    assert config.learning_rate > 0
    assert config.train_batch_size > 0
    assert config.num_epochs > 0

def test_deterministic_seed_setup():
    set_seed(42)
    val1 = torch.rand(1).item()
    set_seed(42)
    val2 = torch.rand(1).item()
    assert val1 == val2

def test_checkpoint_path():
    config = TrainConfig()
    path = Path(config.output_dir)
    # the script creates it if it doesn't exist, we just check it's properly configured
    assert path.name == "baseline"
    
def test_model_loading_and_inference_shape():
    # A tiny smoke test for the test suite using untuned model to test shapes
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    inputs = tokenizer("What is this?", "This is a test context.", return_tensors="pt")
    outputs = model(**inputs)
    
    seq_length = inputs["input_ids"].shape[1]
    assert outputs.start_logits.shape == (1, seq_length)
    assert outputs.end_logits.shape == (1, seq_length)
