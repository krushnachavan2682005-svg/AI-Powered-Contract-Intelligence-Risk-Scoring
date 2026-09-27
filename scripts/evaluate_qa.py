import argparse
import json
import logging
import re
import string
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_text(s):
    """Standard SQuAD text normalization."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(a_gold, a_pred):
    return int(normalize_text(a_gold) == normalize_text(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_text(a_gold).split()
    pred_toks = normalize_text(a_pred).split()
    common = set(gold_toks) & set(pred_toks)
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class QATestDataset(Dataset):
    def __init__(self, data_path: Path, limit: int = None):
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ex["split"] == "test":
                    self.examples.append(ex)
        if limit:
            self.examples = self.examples[:limit]
        logger.info(f"Loaded {len(self.examples)} test examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def get_best_valid_span(start_logits, end_logits, max_answer_len=100):
    """Finds the best valid start and end indices."""
    max_score = float('-inf')
    best_start = 0
    best_end = 0
    
    # Exclude [CLS] for span search, we will compare with [CLS] later if it's best
    for i in range(1, len(start_logits)):
        for j in range(i, min(len(end_logits), i + max_answer_len)):
            score = start_logits[i] + end_logits[j]
            if score > max_score:
                max_score = score
                best_start = i
                best_end = j
                
    cls_score = start_logits[0] + end_logits[0]
    if cls_score > max_score:
        return 0, 0, cls_score
    else:
        return best_start, best_end, max_score

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--model-path", type=str, default="artifacts/colab_training/best_model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    model_path = project_root / args.model_path
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load original documents for gold text
    doc_text_map = {}
    docs_path = project_root / "data/processed/cuad/contracts_normalized.jsonl"
    logger.info("Loading original documents...")
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("normalized_context") or record.get("original_context")
            doc_text_map[record["document_id"]] = text

    dataset_path = project_root / "data/processed/cuad/final_training_dataset.jsonl"
    limit = 500 if args.smoke_test else None
    dataset = QATestDataset(dataset_path, limit=limit)
    
    def collate_fn(batch):
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "raw_batch": batch
        }

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Dictionary to aggregate predictions per (document_id, annotation_id)
    aggregated_preds = defaultdict(lambda: {"max_score": float('-inf'), "pred_text": "", "gold_text": "", "category": ""})

    logger.info("Starting inference...")
    with torch.no_grad():
        for batch_dict in dataloader:
            input_ids_tensor = batch_dict["input_ids"].to(device)
            attention_mask_tensor = batch_dict["attention_mask"].to(device)
            raw_batch = batch_dict["raw_batch"]
            
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            
            for i, ex in enumerate(raw_batch):
                doc_id = ex["document_id"]
                ann_id = ex["annotation_id"]
                cat = ex["category"]
                orig_start = ex["original_start"]
                orig_end = ex["original_end"]
                input_ids = ex["input_ids"]
                
                group_key = (doc_id, ann_id)
                
                # Get gold text
                if ann_id == "impossible":
                    gold_text = ""
                else:
                    doc_text = doc_text_map.get(doc_id, "")
                    gold_text = doc_text[orig_start:orig_end]
                
                # Find best span for this window
                best_s, best_e, score = get_best_valid_span(start_logits[i], end_logits[i])
                
                # Decode predicted span
                if best_s == 0 and best_e == 0:
                    pred_text = ""
                else:
                    pred_text = tokenizer.decode(input_ids[best_s:best_e+1], skip_special_tokens=True)
                
                # Update aggregated prediction if this window has a higher score
                if score > aggregated_preds[group_key]["max_score"]:
                    aggregated_preds[group_key]["max_score"] = score
                    aggregated_preds[group_key]["pred_text"] = pred_text
                    aggregated_preds[group_key]["gold_text"] = gold_text
                    aggregated_preds[group_key]["category"] = cat

    # Calculate metrics
    logger.info("Calculating metrics...")
    total_em = 0.0
    total_f1 = 0.0
    num_answers = len(aggregated_preds)
    
    for key, data in aggregated_preds.items():
        gold = data["gold_text"]
        pred = data["pred_text"]
        
        em = compute_em(gold, pred)
        f1 = compute_f1(gold, pred)
        
        total_em += em
        total_f1 += f1
        
    avg_em = (total_em / num_answers) * 100 if num_answers > 0 else 0
    avg_f1 = (total_f1 / num_answers) * 100 if num_answers > 0 else 0

    logger.info("=== Evaluation Summary ===")
    logger.info(f"Test Examples (windows) processed: {len(dataset)}")
    logger.info(f"Unique Answers Evaluated (groups): {num_answers}")
    logger.info(f"Exact Match (EM): {avg_em:.2f}%")
    logger.info(f"Token F1: {avg_f1:.2f}%")
    
    report_data = {
        "num_windows": len(dataset),
        "num_answers": num_answers,
        "exact_match_percentage": avg_em,
        "average_f1_percentage": avg_f1,
        "model_path": str(model_path)
    }

    report_dir = project_root / "artifacts/colab_training/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "test_evaluation_report.json"
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()
