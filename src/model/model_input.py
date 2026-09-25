from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel
from src.data.window_builder import QAWindowRecord

class ModelInputExample(BaseModel):
    document_id: str
    annotation_id: str
    category: str
    input_ids: List[int]
    attention_mask: List[int]
    start_positions: int
    end_positions: int
    original_start: int
    original_end: int
    window_id: str
    split: str

class ModelInputError(Exception):
    pass

class ModelInputGenerator:
    """Transforms QAWindowRecords into training-ready ModelInputExamples and validates them."""
    
    def generate_and_validate(self, window: QAWindowRecord) -> ModelInputExample:
        """
        Creates a ModelInputExample from a QAWindowRecord and strictly validates it.
        Raises ModelInputError if the example is invalid.
        """
        seq_len = len(window.input_ids)
        start_pos = window.answer.token_start
        end_pos = window.answer.token_end
        
        # Validate positions
        if start_pos < 0:
            raise ModelInputError(f"start_positions ({start_pos}) < 0")
        if end_pos < 0:
            raise ModelInputError(f"end_positions ({end_pos}) < 0")
        if start_pos > end_pos:
            raise ModelInputError(f"start_positions ({start_pos}) > end_positions ({end_pos})")
        if end_pos >= seq_len:
            raise ModelInputError(f"end_positions ({end_pos}) >= sequence length ({seq_len})")
            
        # Check special tokens mappings
        # Special tokens usually have offset [0, 0] in fast tokenizers, except the actual first char might be 0,0 for a real word? No, typically (0,0) is purely special.
        start_offset = window.offset_mapping[start_pos]
        end_offset = window.offset_mapping[end_pos]
        
        if start_offset == [0, 0] and window.answer.char_start != window.answer.char_end:
             # If it's [0,0] and not a purposely empty answer
             raise ModelInputError("start_position maps to a special token [0, 0]")
        if end_offset == [0, 0] and window.answer.char_start != window.answer.char_end:
             raise ModelInputError("end_position maps to a special token [0, 0]")
             
        # Create valid example
        return ModelInputExample(
            document_id=window.document_id,
            annotation_id=window.annotation_id,
            category=window.category,
            input_ids=window.input_ids,
            attention_mask=window.attention_mask,
            start_positions=start_pos,
            end_positions=end_pos,
            original_start=window.answer.original_char_start,
            original_end=window.answer.original_char_end,
            window_id=window.window_id,
            split=window.split
        )
