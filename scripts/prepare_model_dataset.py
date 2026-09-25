import json
import logging
from collections import defaultdict
from pathlib import Path

from src.data.window_builder import QAWindowRecord
from src.model.model_input import ModelInputGenerator, ModelInputError
from src.model.model_factory import ModelFactory

def prepare_dataset():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    input_file = Path("data/interim/windows/cuad_qa_windows.jsonl")
    output_file = Path("data/processed/cuad/cuad_model_inputs.jsonl")
    audit_json = Path("reports/model/model_input_audit.json")
    audit_md = Path("reports/model/model_input_audit.md")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    
    generator = ModelInputGenerator()
    
    stats = {
        "total_windows": 0,
        "valid_examples": 0,
        "excluded_examples": 0,
        "exclusions_by_reason": defaultdict(int),
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "document_overlap": {"train": set(), "val": set(), "test": set()},
    }
    
    logging.info("Starting dataset preparation...")
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in fin:
            stats["total_windows"] += 1
            data = json.loads(line)
            
            try:
                window = QAWindowRecord(**data)
                example = generator.generate_and_validate(window)
                
                # Write to output
                fout.write(example.model_dump_json() + "\n")
                
                # Update stats
                stats["valid_examples"] += 1
                stats["split_counts"][example.split] += 1
                stats["document_overlap"][example.split].add(example.document_id)
                
            except ModelInputError as e:
                stats["excluded_examples"] += 1
                stats["exclusions_by_reason"][str(e)] += 1
            except Exception as e:
                stats["excluded_examples"] += 1
                stats["exclusions_by_reason"][f"Unexpected error: {type(e).__name__}"] += 1

    # Verify overlap
    overlap = {
        "train_val": len(stats["document_overlap"]["train"].intersection(stats["document_overlap"]["val"])),
        "train_test": len(stats["document_overlap"]["train"].intersection(stats["document_overlap"]["test"])),
        "val_test": len(stats["document_overlap"]["val"].intersection(stats["document_overlap"]["test"])),
    }
    
    logging.info(f"Overlap verification: {overlap}")
    assert sum(overlap.values()) == 0, f"Found overlapping documents across splits! Overlap: {overlap}"
    
    # Save audit reports
    audit_data = {
        "total_original_spans_processed": stats["total_windows"],  # This is total windows actually
        "valid_examples": stats["valid_examples"],
        "excluded_examples": stats["excluded_examples"],
        "exclusion_reasons": dict(stats["exclusions_by_reason"]),
        "train_count": stats["split_counts"]["train"],
        "val_count": stats["split_counts"]["val"],
        "test_count": stats["split_counts"]["test"],
        "document_overlap_verification": overlap,
    }
    
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    # Write MD report
    with open(audit_md, "w", encoding="utf-8") as f:
        f.write("# Model Input Audit Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **Total Windows Processed:** {stats['total_windows']}\n")
        f.write(f"- **Valid Model Inputs:** {stats['valid_examples']}\n")
        f.write(f"- **Excluded Windows:** {stats['excluded_examples']}\n\n")
        
        f.write("## Data Splits\n")
        f.write(f"- **Train Examples:** {stats['split_counts']['train']}\n")
        f.write(f"- **Validation Examples:** {stats['split_counts']['val']}\n")
        f.write(f"- **Test Examples:** {stats['split_counts']['test']}\n\n")
        
        f.write("## Document Split Integrity\n")
        f.write("Zero document overlap across splits:\n")
        f.write(f"- Train-Val Overlap: {overlap['train_val']}\n")
        f.write(f"- Train-Test Overlap: {overlap['train_test']}\n")
        f.write(f"- Val-Test Overlap: {overlap['val_test']}\n\n")
        
        f.write("## Exclusion Reasons\n")
        for reason, count in stats["exclusions_by_reason"].items():
            f.write(f"- {reason}: {count}\n")
            
    logging.info("Dataset preparation complete.")
    logging.info(f"Valid examples: {stats['valid_examples']}, Excluded: {stats['excluded_examples']}")

if __name__ == "__main__":
    prepare_dataset()
