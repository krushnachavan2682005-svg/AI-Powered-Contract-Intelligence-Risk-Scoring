import json
import logging
from pathlib import Path
import torch

from src.model.model_input import ModelInputExample
from src.model.model_factory import ModelFactory

def run_smoke_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    input_file = Path("data/processed/cuad/cuad_model_inputs.jsonl")
    
    if not input_file.exists():
        logging.error(f"Dataset not found at {input_file}. Please run prepare_model_dataset.py first.")
        return
        
    logging.info("Loading first example for smoke test...")
    with open(input_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            logging.error("No examples found in dataset.")
            return
            
        example_data = json.loads(first_line)
        example = ModelInputExample(**example_data)
        
    input_ids = torch.tensor([example.input_ids])
    attention_mask = torch.tensor([example.attention_mask])
    
    logging.info(f"Loaded example {example.window_id}")
    logging.info(f"input_ids shape: {input_ids.shape}")
    logging.info(f"attention_mask shape: {attention_mask.shape}")
    logging.info(f"start_positions: {example.start_positions}")
    logging.info(f"end_positions: {example.end_positions}")
    
    logging.info("Initializing ModelFactory...")
    factory = ModelFactory()
    model = factory.load_model()
    
    logging.info("Running forward pass smoke test...")
    try:
        success = factory.smoke_test(model, input_ids, attention_mask)
        if success:
            logging.info("Smoke test passed successfully! Model produced expected start/end logits.")
    except Exception as e:
        logging.error(f"Smoke test failed: {e}")
        
if __name__ == "__main__":
    run_smoke_test()
