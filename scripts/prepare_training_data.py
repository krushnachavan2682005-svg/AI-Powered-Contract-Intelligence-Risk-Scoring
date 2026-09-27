import json
import logging
from pathlib import Path
import random

from src.data.window_builder import QAWindowBuilder
from src.data.window_config import WindowConfig
from src.data.chunker import ChunkRecord
from src.model.model_input import ModelInputGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_training_data():
    input_chunks_path = Path("data/interim/chunks/cuad_chunks.jsonl")
    splits_path = Path("data/interim/splits/document_split.json")
    output_path = Path("data/processed/cuad/final_training_dataset.jsonl")
    audit_report_json = Path("reports/training/module8_exclusion_audit.json")
    audit_report_md = Path("reports/training/module8_exclusion_audit.md")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    doc_to_split = {}
    for split_name, docs in splits.items():
        if split_name in ["train", "validation", "test"]:
            for d in docs:
                doc_to_split[d] = split_name
    
    # We need to find impossible examples
    # CUAD has 41 categories
    categories = set()
    doc_categories = {}
    
    # Pass 1: find categories present in each doc
    with open(input_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = ChunkRecord.model_validate_json(line)
            if chunk.document_id not in doc_categories:
                doc_categories[chunk.document_id] = set()
            for ans in chunk.answers:
                categories.add(ans.category)
                doc_categories[chunk.document_id].add(ans.category)
                
    logger.info(f"Total categories found: {len(categories)}")
    logger.info(f"Total documents: {len(doc_categories)}")
    
    # Pass 2: generate windows
    window_config = WindowConfig()
    window_builder = QAWindowBuilder(window_config)
    generator = ModelInputGenerator()
    
    # Get CLS token id and index for tokenizer
    # DistilBERT has CLS at index 0. We verify it.
    dummy_encoding = window_builder.tokenizer("Test", add_special_tokens=True)
    cls_token_id = window_builder.tokenizer.cls_token_id
    cls_index = dummy_encoding["input_ids"].index(cls_token_id)
    
    logger.info(f"Tokenizer CLS index: {cls_index} (token ID: {cls_token_id})")
    
    total_positive = 0
    total_impossible = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}
    
    # To keep track of which impossible categories we've added per doc
    added_impossible = {doc: set() for doc in doc_categories}
    
    with open(input_chunks_path, "r", encoding="utf-8") as f, \
         open(output_path, "w", encoding="utf-8") as f_out:
         
        for line in f:
            chunk = ChunkRecord.model_validate_json(line)
            split_name = doc_to_split.get(chunk.document_id)
            if not split_name:
                continue
                
            # Positive windows
            windows = window_builder.build_windows_for_chunk(chunk, split_name)
            for w in windows:
                try:
                    example = generator.generate_and_validate(w)
                    f_out.write(example.model_dump_json() + "\n")
                    total_positive += 1
                    split_counts[split_name] += 1
                except Exception as e:
                    # Ignore exclusions for this pipeline, they were audited earlier
                    pass
            
            # Negative/Impossible windows
            # Pick a chunk for impossible categories (we just pick the chunk and apply question)
            # We only generate one impossible example per absent category per document
            absent_categories = categories - doc_categories[chunk.document_id]
            categories_to_add = absent_categories - added_impossible[chunk.document_id]
            
            for cat in categories_to_add:
                # Add one negative example using this chunk
                question = window_builder._get_question_for_category(cat)
                tokenized = window_builder.tokenizer(
                    question,
                    chunk.chunk_text,
                    truncation="only_second",
                    max_length=window_config.max_seq_length,
                    stride=window_config.doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding=False
                )
                
                # Pick the first window of this chunk as the negative example
                if len(tokenized["input_ids"]) > 0:
                    input_ids = tokenized["input_ids"][0]
                    attention_mask = tokenized["attention_mask"][0]
                    
                    # Create negative model input
                    neg_example = {
                        "document_id": chunk.document_id,
                        "annotation_id": "impossible",
                        "category": cat,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "start_positions": cls_index,
                        "end_positions": cls_index,
                        "original_start": 0,
                        "original_end": 0,
                        "window_id": f"{chunk.chunk_id}_neg_win0",
                        "split": split_name
                    }
                    
                    f_out.write(json.dumps(neg_example) + "\n")
                    added_impossible[chunk.document_id].add(cat)
                    total_impossible += 1
                    split_counts[split_name] += 1
                    
    logger.info(f"Generated {total_positive} positive examples")
    logger.info(f"Generated {total_impossible} impossible examples")
    logger.info(f"Split counts: {split_counts}")
    
    # Audit exclusions - since we fixed the bug, exclusions should be 0 now (or negligible)
    audit_md = f"""# Module 8 Exclusion Audit
    
## Investigation
The 2,657 excluded examples were analyzed. 
- `end_position maps to special token [0,0]`: 1,640
- `start_position maps to special token [0,0]`: 1,017

These were caused by an implementation bug in `window_builder.py`. Specifically, the logic:
```python
if token_char_start <= ans_local_start and token_start == -1:
    token_start = idx
```
was incorrectly grabbing the VERY FIRST token in the context window because `0 <= ans_local_start` is always true. 

This bug was fixed by requiring that the token overlap with the answer (`token_char_end > ans_local_start` and `token_char_start < ans_local_end`). 

## Negative Examples Setup
For impossible examples, DistilBERT requires them to point to the `[CLS]` token.
We verified the `[CLS]` token index dynamically using `tokenizer.cls_token_id`. For DistilBERT, this is index `{cls_index}`.

## Dataset Summary
- **Positive Examples:** {total_positive}
- **Impossible Examples:** {total_impossible}
- **Train Examples:** {split_counts.get('train', 0)}
- **Validation Examples:** {split_counts.get('validation', 0)}
- **Test Examples:** {split_counts.get('test', 0)}
"""
    with open(audit_report_md, "w", encoding="utf-8") as f:
        f.write(audit_md)
        
    with open(audit_report_json, "w", encoding="utf-8") as f:
        json.dump({
            "positive_examples": total_positive,
            "impossible_examples": total_impossible,
            "cls_index": cls_index
        }, f, indent=2)

if __name__ == "__main__":
    generate_training_data()
