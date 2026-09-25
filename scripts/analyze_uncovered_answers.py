import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def run_analysis():
    chunks_path = Path("data/interim/chunks/cuad_chunks.jsonl")
    windows_path = Path("data/interim/windows/cuad_qa_windows.jsonl")
    splits_path = Path("data/interim/splits/document_split.json")
    out_dir = Path("reports/window_coverage")
    out_dir.mkdir(parents=True, exist_ok=True)

    # load splits
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    doc_to_split = {}
    for sp_name in ["train", "validation", "test"]:
        if sp_name in splits:
            for d in splits[sp_name]:
                doc_to_split[d] = sp_name

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)

    all_answers = {}

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            doc_id = chunk["document_id"]
            for ans in chunk.get("answers", []):
                span_id = f"{doc_id}_{ans['annotation_id']}_{ans['original_start']}_{ans['original_end']}"
                if span_id not in all_answers:
                    all_answers[span_id] = {
                        "document_id": doc_id,
                        "annotation_id": ans["annotation_id"],
                        "category": ans["category"],
                        "original_start": ans["original_start"],
                        "original_end": ans["original_end"],
                        "answer_text": ans["answer_text"],
                        "chunks": [],
                    }
                all_answers[span_id]["chunks"].append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["chunk_text"],
                        "local_start": ans["local_start"],
                        "local_end": ans["local_end"],
                    }
                )

    covered_spans = set()
    with open(windows_path, "r", encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            span_id = f"{w['document_id']}_{w['annotation_id']}_{w['answer']['original_char_start']}_{w['answer']['original_char_end']}"
            covered_spans.add(span_id)

    uncovered_ids = set(all_answers.keys()) - covered_spans
    uncovered_answers = [all_answers[sid] for sid in uncovered_ids]

    # We want exact cause tracking as instructed:
    # 1. Answer token length > available context capacity
    # 2. Answer lies across every generated window boundary
    # 3. Incorrect sequence_ids handling / logic issue
    # We will verify if it's completely missing or partially covered

    cause_distribution = defaultdict(int)
    examples_by_cause = defaultdict(list)
    stats = {
        "answer_char_lengths": [],
        "answer_token_lengths": [],
        "split_distribution": defaultdict(int),
        "category_distribution": defaultdict(int),
    }

    def get_question_for_category(category):
        return f'Highlight the parts (if any) of this contract related to "{category}" that should be reviewed by a lawyer. Details:'

    # To answer:
    # A. How many uncovered answers are fundamentally impossible to fit in a 512-token context?
    # B. How many should be recoverable by fixing window generation?
    # C. How many are caused by answer length exceeding model context?
    # D. How many are caused by implementation/mapping errors?

    num_impossible = 0
    num_recoverable_window = 0
    num_mapping_error = 0

    for u_ans in uncovered_answers:
        category = u_ans["category"]
        doc_id = u_ans["document_id"]
        split = doc_to_split.get(doc_id, "unknown")
        stats["split_distribution"][split] += 1
        stats["category_distribution"][category] += 1

        ans_text = u_ans["answer_text"]
        char_len = len(ans_text)
        stats["answer_char_lengths"].append(char_len)

        ans_tokens = tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        token_len = len(ans_tokens)
        stats["answer_token_lengths"].append(token_len)

        q_text = get_question_for_category(category)
        q_tokens = tokenizer(q_text, add_special_tokens=False)["input_ids"]

        max_context_capacity = 512 - len(q_tokens) - 3

        if token_len > max_context_capacity:
            cause = "Answer length > max context capacity (Impossible to fit)"
            num_impossible += 1
        else:
            # Answer fits, why was it missed?
            crosses_boundary = True
            for ch in u_ans["chunks"]:
                tokenized = tokenizer(
                    q_text,
                    ch["chunk_text"],
                    truncation="only_second",
                    max_length=512,
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding=False,
                )

                num_windows = len(tokenized["input_ids"])
                for window_idx in range(num_windows):
                    sequence_ids = tokenized.sequence_ids(window_idx)
                    offset_mapping = tokenized["offset_mapping"][window_idx]
                    context_token_indices = [
                        i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
                    ]
                    if not context_token_indices:
                        continue
                    window_char_start = offset_mapping[context_token_indices[0]][0]
                    window_char_end = offset_mapping[context_token_indices[-1]][1]

                    if (
                        ch["local_start"] >= window_char_start
                        and ch["local_end"] <= window_char_end
                    ):
                        crosses_boundary = False
                        break
                if not crosses_boundary:
                    break

            if crosses_boundary:
                cause = "Answer lies across every generated window boundary (Stride/Overlap issue)"
                num_recoverable_window += 1
            else:
                cause = "Logic/Mapping error (Window exists but logic missed it)"
                num_mapping_error += 1

        cause_distribution[cause] += 1
        if len(examples_by_cause[cause]) < 5:
            examples_by_cause[cause].append(u_ans)

    report_json = {
        "summary": {
            "total_original_answer_spans": len(all_answers),
            "covered_spans": len(covered_spans),
            "uncovered_spans": len(uncovered_answers),
            "coverage_percentage": (
                (len(covered_spans) / len(all_answers)) * 100 if all_answers else 0
            ),
        },
        "causes": dict(cause_distribution),
        "categorization": {
            "A_impossible_context_exceeded": num_impossible,
            "B_recoverable_window_generation": num_recoverable_window,
            "C_context_exceeded": num_impossible,
            "D_implementation_mapping_error": num_mapping_error,
        },
        "answer_length_stats": {
            "char_mean": (
                sum(stats["answer_char_lengths"]) / len(stats["answer_char_lengths"])
                if stats["answer_char_lengths"]
                else 0
            ),
            "char_max": (
                max(stats["answer_char_lengths"]) if stats["answer_char_lengths"] else 0
            ),
            "char_min": (
                min(stats["answer_char_lengths"]) if stats["answer_char_lengths"] else 0
            ),
            "token_mean": (
                sum(stats["answer_token_lengths"]) / len(stats["answer_token_lengths"])
                if stats["answer_token_lengths"]
                else 0
            ),
            "token_max": (
                max(stats["answer_token_lengths"])
                if stats["answer_token_lengths"]
                else 0
            ),
            "token_min": (
                min(stats["answer_token_lengths"])
                if stats["answer_token_lengths"]
                else 0
            ),
        },
        "split_distribution": dict(stats["split_distribution"]),
        "category_distribution": dict(stats["category_distribution"]),
        "examples_by_cause": {
            k: [
                {
                    "document_id": ex["document_id"],
                    "annotation_id": ex["annotation_id"],
                    "length_chars": len(ex["answer_text"]),
                    "category": ex["category"],
                }
                for ex in v
            ]
            for k, v in examples_by_cause.items()
        },
    }

    with open(out_dir / "window_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    with open(out_dir / "window_coverage_report.md", "w", encoding="utf-8") as f:
        f.write("# Window Coverage Report\n\n")
        f.write(f"- Total original answer spans: {len(all_answers)}\n")
        f.write(f"- Covered spans: {len(covered_spans)}\n")
        f.write(f"- Uncovered spans: {len(uncovered_answers)}\n\n")

        f.write("## Root Cause Breakdown\n")
        for k, v in cause_distribution.items():
            pct = (v / len(uncovered_answers)) * 100 if uncovered_answers else 0
            f.write(f"- **{k}**: {v} ({pct:.2f}%)\n")

        f.write("\n## High-Level Categorization\n")
        f.write(f"A. Impossible to fit in a 512-token context: {num_impossible}\n")
        f.write(
            f"B. Recoverable by fixing window generation (stride): {num_recoverable_window}\n"
        )
        f.write(
            f"C. Caused by answer length exceeding model context: {num_impossible}\n"
        )
        f.write(f"D. Caused by implementation/mapping errors: {num_mapping_error}\n")

        f.write("\n## Answer Length Statistics (Uncovered)\n")
        f.write(
            f"- Characters: Mean = {report_json['answer_length_stats']['char_mean']:.1f}, Max = {report_json['answer_length_stats']['char_max']}, Min = {report_json['answer_length_stats']['char_min']}\n"
        )
        f.write(
            f"- Tokens: Mean = {report_json['answer_length_stats']['token_mean']:.1f}, Max = {report_json['answer_length_stats']['token_max']}, Min = {report_json['answer_length_stats']['token_min']}\n"
        )

        f.write("\n## Distribution by Split\n")
        for k, v in stats["split_distribution"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Distribution by Category (Top 10)\n")
        top_cats = sorted(
            stats["category_distribution"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        for k, v in top_cats:
            f.write(f"- {k}: {v}\n")

        f.write("\n## Sample Examples\n")
        for k, v in examples_by_cause.items():
            f.write(f"### {k}\n")
            for ex in v:
                f.write(
                    f"- Document: `{ex['document_id']}`, Annotation: `{ex['annotation_id']}`, Category: `{ex['category']}`, Length: {len(ex['answer_text'])} chars\n"
                )


if __name__ == "__main__":
    run_analysis()
