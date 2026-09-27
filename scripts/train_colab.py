import argparse
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 3e-5
    train_batch_size: int = 16  # Good for T4 with mixed precision
    eval_batch_size: int = 16
    num_epochs: int = 3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    output_dir: str = "artifacts/colab_training"
    metrics_dir: str = "artifacts/colab_training/metrics"
    reports_dir: str = "artifacts/colab_training/reports"
    logging_steps: int = 50
    smoke_test_mode: bool = False
    use_fp16: bool = True
    project_root: str = "."

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
        logger.info(f"Loading {split} dataset from {data_path}...")
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

def parse_args():
    parser = argparse.ArgumentParser(description="Colab-optimized Training Script")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--project-root", type=str, default=".", help="Root directory of the project")
    parser.add_argument("--copy-to-local", action="store_true", help="Copy dataset to /content/temp_data for faster IO in Colab")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = TrainConfig(
        smoke_test_mode=args.smoke_test,
        use_fp16=not args.no_fp16,
        project_root=args.project_root
    )
    
    if config.smoke_test_mode:
        logger.info("SMOKE TEST MODE ENABLED. Using minimal dataset and 1 epoch.")
        config.num_epochs = 1
        config.logging_steps = 5

    set_seed(config.seed)
    
    project_root = Path(config.project_root)
    output_dir = project_root / config.output_dir
    metrics_dir = project_root / config.metrics_dir
    reports_dir = project_root / config.reports_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"Loading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
    model.to(device)
    
    original_data_path = project_root / "data/processed/cuad/final_training_dataset.jsonl"
    if not original_data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {original_data_path}")

    # Copy to local /content for faster I/O in Colab if requested
    data_path = original_data_path
    if args.copy_to_local and str(project_root).startswith("/content/drive"):
        local_dir = Path("/content/temp_data")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_data_path = local_dir / "final_training_dataset.jsonl"
        if not local_data_path.exists():
            logger.info(f"Copying dataset from Google Drive to {local_data_path} for faster I/O...")
            shutil.copy2(original_data_path, local_data_path)
        data_path = local_data_path
    
    train_limit = 100 if config.smoke_test_mode else None
    eval_limit = 20 if config.smoke_test_mode else None
    
    train_dataset = QADataset(data_path, "train", limit=train_limit)
    val_dataset = QADataset(data_path, "validation", limit=eval_limit)
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    scaler = GradScaler(enabled=config.use_fp16)
    
    logger.info("Starting training...")
    start_time = time.time()
    
    total_loss = 0
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            with autocast(enabled=config.use_fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Step {global_step}, Loss: {total_loss / config.logging_steps:.4f}")
                    total_loss = 0
        
        # Validation at end of epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                
                with autocast(enabled=config.use_fp16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions
                    )
                val_loss += outputs.loss.item()
                
        val_loss /= max(len(val_loader), 1)
        train_loss = epoch_loss / max(len(train_loader), 1)
        logger.info(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        ckpt_dir = output_dir / "checkpoints" / f"epoch_{epoch+1}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dir = output_dir / "best_model"
            logger.info(f"New best validation loss ({best_val_loss:.4f}). Saving to {best_model_dir}")
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
                    
    train_runtime = time.time() - start_time
    logger.info(f"Training completed in {train_runtime:.2f}s")
    
    # Save final model
    final_model_dir = output_dir / "final_model"
    logger.info(f"Saving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
        
    metrics = {
        "train_runtime": train_runtime,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "smoke_test_mode": config.smoke_test_mode
    }
    
    with open(metrics_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    report = f"""# Colab Training Report

## Configuration
- Model: {config.model_name}
- Batch Size: {config.train_batch_size} (Train), {config.eval_batch_size} (Eval)
- Gradient Accumulation: {config.gradient_accumulation_steps}
- Mixed Precision (FP16): {config.use_fp16}
- Learning Rate: {config.learning_rate}
- Epochs: {config.num_epochs}
- Smoke Test Mode: {config.smoke_test_mode}

## Metrics
- Training Runtime: {train_runtime:.2f}s
- Best Validation Loss: {best_val_loss:.4f}
- Final Train Loss: {train_loss:.4f}
- Final Validation Loss: {val_loss:.4f}

## Saved Models Location
- Best Model: {output_dir}/best_model
- Final Model: {output_dir}/final_model
"""
    with open(reports_dir / "colab_training_report.md", "w") as f:
        f.write(report)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()
