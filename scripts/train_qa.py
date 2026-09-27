import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 3e-5
    train_batch_size: int = 4
    eval_batch_size: int = 8
    num_epochs: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = "artifacts/checkpoints/baseline"
    metrics_dir: str = "artifacts/metrics"
    reports_dir: str = "reports/training"
    logging_steps: int = 50
    smoke_test_mode: bool = False  # Default to true to prevent massive laptop heating

def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class QADataset(Dataset):
    def __init__(self, data_path: Path, split: str, limit: int = None):
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ex["split"] == split:
                    self.examples.append(ex)
                    
        if limit and limit < len(self.examples):
            self.examples = self.examples[:limit]
            
        logger.info(f"Loaded {len(self.examples)} examples for split '{split}'")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(ex["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(ex["end_positions"], dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    start_positions = torch.stack([item["start_positions"] for item in batch])
    end_positions = torch.stack([item["end_positions"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions
    }

def train():
    config = TrainConfig()
    set_seed(config.seed)
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(config.reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    data_path = Path("data/processed/cuad/final_training_dataset.jsonl")
    
    train_limit = 100 if config.smoke_test_mode else None
    eval_limit = 20 if config.smoke_test_mode else None
    
    train_dataset = QADataset(data_path, "train", limit=train_limit)
    val_dataset = QADataset(data_path, "validation", limit=eval_limit)
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    logger.info("Starting training...")
    start_time = time.time()
    
    model.train()
    total_loss = 0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    logger.info(f"Epoch {epoch+1}, Step {global_step}, Loss: {total_loss / config.logging_steps:.4f}")
                    total_loss = 0
                    
    train_runtime = time.time() - start_time
    logger.info(f"Training completed in {train_runtime:.2f}s")
    
    # Validation
    logger.info("Starting validation...")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                start_positions=batch["start_positions"].to(device),
                end_positions=batch["end_positions"].to(device)
            )
            val_loss += outputs.loss.item()
            
    val_loss /= len(val_loader) if len(val_loader) > 0 else 1
    logger.info(f"Validation Loss: {val_loss:.4f}")
    
    # Save checkpoint
    logger.info(f"Saving checkpoint to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save config
    with open(f"{config.output_dir}/train_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
        
    metrics = {
        "train_runtime": train_runtime,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": epoch_loss / len(train_loader) if len(train_loader) > 0 else 0,
        "val_loss": val_loss,
        "smoke_test_mode": config.smoke_test_mode
    }
    
    with open(Path(config.metrics_dir) / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    report = f"""# Baseline Training Report

## Configuration
- Model: {config.model_name}
- Batch Size: {config.train_batch_size} (Train), {config.eval_batch_size} (Eval)
- Gradient Accumulation: {config.gradient_accumulation_steps}
- Learning Rate: {config.learning_rate}
- Epochs: {config.num_epochs}
- Smoke Test Mode: {config.smoke_test_mode}

## Metrics
- Training Runtime: {train_runtime:.2f}s
- Training Loss: {metrics['train_loss']:.4f}
- Validation Loss: {metrics['val_loss']:.4f}

## Checkpoint Location
{config.output_dir}
"""
    with open(Path(config.reports_dir) / "baseline_training_report.md", "w") as f:
        f.write(report)
        
    logger.info("Done.")

if __name__ == "__main__":
    train()
