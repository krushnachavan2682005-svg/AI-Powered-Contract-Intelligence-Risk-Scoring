from typing import Tuple, Union
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, PreTrainedTokenizerFast
import torch

from src.model.model_config import ModelConfig

class ModelFactory:
    """Factory to load models and tokenizers for extractive QA."""
    
    def __init__(self, config: Union[ModelConfig, None] = None):
        self.config = config or ModelConfig()
        
    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        """Loads and returns the fast tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, use_fast=True)
            if not isinstance(tokenizer, PreTrainedTokenizerFast):
                raise ValueError(f"Tokenizer {self.config.tokenizer_name} is not a fast tokenizer. Fast tokenizer is required.")
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer {self.config.tokenizer_name}: {e}") from e
            
    def load_model(self):
        """Loads and returns the model for question answering."""
        try:
            model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.config.model_name}: {e}") from e

    def load(self) -> Tuple[AutoModelForQuestionAnswering, PreTrainedTokenizerFast]:
        """Loads both model and tokenizer, ensuring basic compatibility."""
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        return model, tokenizer

    def smoke_test(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> bool:
        """Runs a tiny forward pass to verify shapes and outputs."""
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        assert hasattr(outputs, "start_logits"), "Model missing start_logits"
        assert hasattr(outputs, "end_logits"), "Model missing end_logits"
        
        batch_size, seq_len = input_ids.shape
        assert outputs.start_logits.shape == (batch_size, seq_len), "start_logits shape mismatch"
        assert outputs.end_logits.shape == (batch_size, seq_len), "end_logits shape mismatch"
        
        return True
