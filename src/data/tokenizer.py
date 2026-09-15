from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class TokenizerConfig(BaseModel):
    model_name: str = "distilbert-base-uncased"
    max_length: Optional[int] = None


class TokenizerWrapper:
    """
    Wraps a Hugging Face fast tokenizer for deterministic QA preprocessing.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        # Initialize fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)  # type: ignore
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                f"Tokenizer {self.config.model_name} is not a fast tokenizer. Fast tokenizer is required for offset mapping."
            )
        self.tokenizer = tokenizer

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def tokenize_chunk(self, text: str) -> Dict[str, Any]:
        """
        Tokenizes the chunk and returns offsets without truncation unless max_length is set.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=False if self.config.max_length is None else True,
            max_length=self.config.max_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": encoding["offset_mapping"],
        }

    def map_character_to_token(
        self, char_start: int, char_end: int, offset_mapping: List[Tuple[int, int]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Maps a character-level span to a token-level span.
        Returns (token_start, token_end) or (None, None) if mapping fails.
        """
        token_start = None
        token_end = None

        for idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens which have (0, 0)
            if start == 0 and end == 0:
                continue

            if token_start is None and start <= char_start < end:
                token_start = idx

            if token_start is not None and start < char_end <= end:
                token_end = idx
                break

        # Edge case: Answer might end exactly at the token start, which we might miss above
        if token_start is not None and token_end is None:
            # Check if char_end aligns exactly with a token end or we just reached it
            for idx in range(token_start, len(offset_mapping)):
                s, e = offset_mapping[idx]
                if s == 0 and e == 0:
                    continue
                if e >= char_end:
                    token_end = idx
                    break

        if token_start is None or token_end is None:
            return None, None

        return token_start, token_end
