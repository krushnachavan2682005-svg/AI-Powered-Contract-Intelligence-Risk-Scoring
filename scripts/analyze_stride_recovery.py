import json
from pathlib import Path

from transformers import AutoTokenizer


def run_analysis():
    chunks_path = Path("data/interim/chunks/cuad_chunks.jsonl")
    windows_path = Path("data/interim/windows/cuad_qa_windows.jsonl")
    out_dir = Path("reports/window_coverage")
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    }

    covered_spans = set()
    with open(windows_path, "r", encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            span_id = f"{w['document_id']}_{w['annotation_id']}_{w['answer']['original_char_start']}_{w['answer']['original_char_end']}"
            covered_spans.add(span_id)

    uncovered_ids = set(all_answers.keys()) - covered_spans
    uncovered_answers = [all_answers[sid] for sid in uncovered_ids]

    num_impossible = 0
    num_other = 0

    def get_question_for_category(category):
        return f'Highlight the parts (if any) of this contract related to "{category}" that should be reviewed by a lawyer. Details:'

    impossible_examples = []

    for u_ans in uncovered_answers:
        category = u_ans["category"]
        ans_text = u_ans["answer_text"]

        ans_tokens = tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        token_len = len(ans_tokens)

        q_text = category # Category already has the full text
        q_tokens = tokenizer(q_text, add_special_tokens=False)["input_ids"]

        max_context_capacity = 512 - len(q_tokens) - 3

        if token_len > max_context_capacity:
            num_impossible += 1
            if len(impossible_examples) < 5:
                impossible_examples.append(u_ans)
        else:
            num_other += 1

    total_spans = len(all_answers)
    covered = len(covered_spans)
    uncovered = len(uncovered_answers)

    original_uncovered = 232
    recovered = original_uncovered - uncovered

    report_json = {
        "coverage_before": (13591 / 13823) * 100 if total_spans == 13823 else 0,
        "coverage_after": (covered / total_spans) * 100 if total_spans else 0,
        "total_spans": total_spans,
        "covered": covered,
        "uncovered": uncovered,
        "original_uncovered": original_uncovered,
        "recovered": recovered,
        "still_uncovered": uncovered,
        "remaining_impossible_spans": num_impossible,
        "remaining_other_spans": num_other,
        "impossible_examples": impossible_examples,
    }

    with open(out_dir / "stride_recovery_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    with open(out_dir / "stride_recovery_report.md", "w", encoding="utf-8") as f:
        f.write("# Stride Recovery Report\n\n")
        f.write(
            f"- Coverage before Module 7.4: {report_json['coverage_before']:.2f}%\n"
        )
        f.write(
            f"- Coverage after Module 7.4: {report_json['coverage_after']:.2f}%\n\n"
        )

        f.write(f"- Total original answer spans: {total_spans}\n")
        f.write(f"- Covered spans: {covered}\n")
        f.write(f"- Original uncovered: {original_uncovered}\n")
        f.write(f"- Recovered: {recovered}\n")
        f.write(f"- Still uncovered: {uncovered}\n")
        f.write(f"  - Remaining impossible spans (>512): {num_impossible}\n")
        f.write(f"  - Remaining other (mapping errors): {num_other}\n\n")

        if impossible_examples:
            f.write("## Examples of Remaining Impossible Spans\n")
            for ex in impossible_examples:
                f.write(
                    f"- Document: `{ex['document_id']}`, Category: `{ex['category']}`, Length: {len(ex['answer_text'])} chars\n"
                )


if __name__ == "__main__":
    run_analysis()
