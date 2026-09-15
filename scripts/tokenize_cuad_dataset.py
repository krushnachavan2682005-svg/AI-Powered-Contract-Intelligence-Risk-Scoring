import json
import logging
from pathlib import Path

from src.data.tokenizer import TokenizerConfig, TokenizerWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_chunks(
    input_file: Path,
    output_file: Path,
    report_file: Path,
    summary_file: Path,
    tokenizer_model: str = "distilbert-base-uncased",
):
    wrapper = TokenizerWrapper(TokenizerConfig(model_name=tokenizer_model))

    total_contracts = set()
    total_chunks = 0
    total_examples = 0
    total_spans = 0
    mapped_spans = 0
    failed_spans = 0

    token_lengths = []
    char_lengths = []

    thresholds = {128: 0, 256: 0, 384: 0, 512: 0}

    with (
        open(input_file, "r", encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            chunk = json.loads(line)
            total_contracts.add(chunk["document_id"])
            total_chunks += 1

            text = chunk["chunk_text"]
            char_lengths.append(len(text))

            encoding = wrapper.tokenize_chunk(text)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            offset_mapping = encoding["offset_mapping"]

            token_len = len(input_ids)
            token_lengths.append(token_len)

            for threshold in thresholds:
                if token_len > threshold:
                    thresholds[threshold] += 1

            answers = chunk.get("answers", [])
            for ans in answers:
                total_spans += 1
                char_start = ans["local_start"]
                char_end = ans["local_end"]

                token_start, token_end = wrapper.map_character_to_token(
                    char_start, char_end, offset_mapping
                )

                if (
                    token_start is not None
                    and token_end is not None
                    and token_start <= token_end
                ):
                    mapped_spans += 1

                    # Validate reconstructing string
                    # Distilbert uses WordPiece, we check char offsets directly
                    recon_start = offset_mapping[token_start][0]
                    recon_end = offset_mapping[token_end][1]

                    example = {
                        "example_id": f"{chunk['chunk_id']}::{ans['annotation_id']}",
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "annotation_id": ans["annotation_id"],
                        "category": ans["category"],
                        "question": ans["category"],
                        "context": text,
                        "answer": {
                            "text": ans["answer_text"],
                            "char_start": char_start,
                            "char_end": char_end,
                            "token_start": token_start,
                            "token_end": token_end,
                            "recon_char_start": recon_start,
                            "recon_char_end": recon_end,
                        },
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "offset_mapping": offset_mapping,
                    }
                    fout.write(json.dumps(example) + "\n")
                    total_examples += 1
                else:
                    failed_spans += 1
                    logger.warning(f"Failed to map span: {ans['annotation_id']}")

    if total_spans == 0:
        coverage = 0.0
    else:
        coverage = (mapped_spans / total_spans) * 100

    token_lengths.sort()
    count = len(token_lengths)

    stats = {
        "dataset": {
            "contracts": len(total_contracts),
            "chunks": total_chunks,
            "examples": total_examples,
            "answer_spans": total_spans,
        },
        "tokenizer": {
            "model_name": tokenizer_model,
            "is_fast": wrapper.is_fast,
            "vocab_size": wrapper.vocab_size,
        },
        "token_statistics": {
            "min": token_lengths[0] if count > 0 else 0,
            "max": token_lengths[-1] if count > 0 else 0,
            "mean": sum(token_lengths) / count if count > 0 else 0,
            "median": token_lengths[count // 2] if count > 0 else 0,
            "p90": token_lengths[int(count * 0.9)] if count > 0 else 0,
            "p95": token_lengths[int(count * 0.95)] if count > 0 else 0,
            "p99": token_lengths[int(count * 0.99)] if count > 0 else 0,
        },
        "threshold_counts": thresholds,
        "answer_mapping": {
            "total_spans": total_spans,
            "mapped_spans": mapped_spans,
            "failed_spans": failed_spans,
            "coverage_percentage": coverage,
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(report_file.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(report_file.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write("# Tokenization Report\n\n")
        f.write(f"Model: {tokenizer_model}\n")
        f.write(f"Fast Tokenizer: {wrapper.is_fast}\n")
        f.write(f"Coverage: {coverage:.2f}%\n")
        f.write(
            f"Spans: {total_spans} total, {mapped_spans} mapped, {failed_spans} failed\n"
        )
        f.write("\n## Thresholds\n")
        for k, v in thresholds.items():
            f.write(f"> {k}: {v}\n")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    input_file = base / "data" / "interim" / "chunks" / "cuad_chunks.jsonl"
    output_dir = base / "data" / "interim" / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = base / "reports" / "tokenization"
    report_dir.mkdir(parents=True, exist_ok=True)

    process_chunks(
        input_file=input_file,
        output_file=output_dir / "cuad_qa_examples.jsonl",
        summary_file=output_dir / "tokenization_summary.json",
        report_file=report_dir / "tokenization_report",
    )
