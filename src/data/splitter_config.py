from pydantic import BaseModel, Field

class SplitConfig(BaseModel):
    """Configuration for document-level dataset splitting."""
    train_percent: float = Field(default=0.70, ge=0.0, le=1.0)
    validation_percent: float = Field(default=0.15, ge=0.0, le=1.0)
    test_percent: float = Field(default=0.15, ge=0.0, le=1.0)
    seed: int = Field(default=42)

    def model_post_init(self, __context) -> None:
        total = self.train_percent + self.validation_percent + self.test_percent
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split percentages must sum to 1.0, got {total}")
