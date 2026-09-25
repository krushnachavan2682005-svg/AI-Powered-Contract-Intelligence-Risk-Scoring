from pydantic import BaseModel, Field


class WindowConfig(BaseModel):
    """Configuration for token-aware QA window construction."""

    max_seq_length: int = Field(default=512, gt=0)
    doc_stride: int = Field(default=128, ge=0)
    tokenizer_name: str = Field(default="distilbert-base-uncased")
