import json
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def run_smoke_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    checkpoint_dir = Path("artifacts/checkpoints/baseline")
    input_file = Path("data/processed/cuad/final_training_dataset.jsonl")
    
    if not checkpoint_dir.exists():
        logging.error(f"Checkpoint not found at {checkpoint_dir}. Please run train_qa.py first.")
        return
        
    logging.info(f"Loading model and tokenizer from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_dir)
    model.eval()
    
    logging.info("Loading first validation example for inference...")
    val_example = None
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["split"] == "validation":
                val_example = data
                break
                
    if not val_example:
        logging.error("No validation examples found.")
        return
        
    input_ids = torch.tensor([val_example["input_ids"]])
    attention_mask = torch.tensor([val_example["attention_mask"]])
    seq_length = input_ids.shape[1]
    
    logging.info(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    logging.info(f"Start logits shape: {start_logits.shape}")
    logging.info(f"End logits shape: {end_logits.shape}")
    
    assert start_logits.shape == (1, seq_length), "Invalid start logits shape"
    assert end_logits.shape == (1, seq_length), "Invalid end logits shape"
    
    pred_start = torch.argmax(start_logits, dim=1).item()
    pred_end = torch.argmax(end_logits, dim=1).item()
    
    logging.info(f"Predicted start position: {pred_start}")
    logging.info(f"Predicted end position: {pred_end}")
    
    assert pred_start <= seq_length, "Predicted start > sequence length"
    assert pred_end <= seq_length, "Predicted end > sequence length"
    
    # Check if context token or CLS
    token_id_start = input_ids[0, pred_start].item()
    token_id_end = input_ids[0, pred_end].item()
    
    start_token = tokenizer.decode([token_id_start])
    end_token = tokenizer.decode([token_id_end])
    
    logging.info(f"Predicted start token: '{start_token}' (ID: {token_id_start})")
    logging.info(f"Predicted end token: '{end_token}' (ID: {token_id_end})")
    
    logging.info("Inference smoke test passed successfully!")
    
if __name__ == "__main__":
    run_smoke_test()
