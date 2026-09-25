from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Configuration for the extractive QA transformer model."""
    
    model_name: str = Field(default="distilbert-base-uncased", description="Base model name or path.")
    tokenizer_name: str = Field(default="distilbert-base-uncased", description="Tokenizer name or path.")
    
    max_seq_length: int = Field(default=512, description="Maximum sequence length for tokenization.")
    
    learning_rate: float = Field(default=3e-5, description="Initial learning rate for AdamW.")
    batch_size: int = Field(default=8, description="Training batch size per device.")
    eval_batch_size: int = Field(default=8, description="Evaluation batch size per device.")
    num_epochs: int = Field(default=3, description="Total number of training epochs to perform.")
    weight_decay: float = Field(default=0.01, description="Weight decay for AdamW if we apply some.")
    warmup_ratio: float = Field(default=0.1, description="Linear warmup over warmup_ratio fraction of total steps.")
    gradient_accumulation_steps: int = Field(default=1, description="Number of updates steps to accumulate before performing a backward/update pass.")
    
    seed: int = Field(default=42, description="Random seed for initialization.")
    output_dir: str = Field(default="artifacts/model_output", description="The output directory where the model predictions and checkpoints will be written.")
