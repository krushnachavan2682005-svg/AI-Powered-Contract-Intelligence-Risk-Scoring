import json
import logging
from collections import defaultdict
from pathlib import Path

from src.data.chunker import ChunkRecord
from src.data.splitter import DocumentSplitter
from src.data.splitter_config import SplitConfig
from src.data.window_builder import QAWindowBuilder
from src.data.window_config import WindowConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run():
    input_chunks_path = Path("data/interim/chunks/cuad_chunks.jsonl")
    if not input_chunks_path.exists():
        logger.error(f"Input file not found: {input_chunks_path}")
        return

    splits_dir = Path("data/interim/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    windows_dir = Path("data/interim/windows")
    windows_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path("reports/splitting")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Extract all document IDs
    document_ids = set()
    with open(input_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = ChunkRecord.model_validate_json(line)
            document_ids.add(chunk.document_id)

    # Split documents
    logger.info(f"Found {len(document_ids)} unique documents.")
    splitter_config = SplitConfig()
    splitter = DocumentSplitter(splitter_config)
    splits = splitter.split_documents(list(document_ids))

    # Save splits
    splits_out_path = splits_dir / "document_split.json"
    with open(splits_out_path, "w", encoding="utf-8") as f:
        json.main = {
            "seed": splitter_config.seed,
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"],
        }
        json.dump(json.main, f, indent=2)

    doc_to_split = {}
    for s_name, s_docs in splits.items():
        for d in s_docs:
            doc_to_split[d] = s_name

    # Pass 2: Build QA windows
    window_config = WindowConfig()
    window_builder = QAWindowBuilder(window_config)

    windows_out_path = windows_dir / "cuad_qa_windows.jsonl"

    stats = {
        "total_documents": len(document_ids),
        "train_documents": len(splits["train"]),
        "validation_documents": len(splits["validation"]),
        "test_documents": len(splits["test"]),
        "percentages": {
            "train": splitter_config.train_percent,
            "validation": splitter_config.validation_percent,
            "test": splitter_config.test_percent,
        },
        "seed": splitter_config.seed,
        "overlap_count": 0,
        "missing_count": 0,
        "total_windows": 0,
        "positive_windows": 0,
        "total_original_answer_spans": 0,
        "covered_answer_spans": 0,
        "uncovered_answer_spans": 0,
        "multiple_window_answer_count": 0,
        "spans_represented": set(),
        "spans_total_set": set(),
        "span_occurrences": defaultdict(int),
        "split_stats": defaultdict(
            lambda: {
                "chunks": 0,
                "qa_examples": 0,
                "positive_answer_occurrences": 0,
                "unique_original_answer_spans": set(),
                "covered_spans": set(),
            }
        ),
    }

    with (
        open(input_chunks_path, "r", encoding="utf-8") as f_in,
        open(windows_out_path, "w", encoding="utf-8") as f_out,
    ):

        for line in f_in:
            chunk = ChunkRecord.model_validate_json(line)
            split_name = doc_to_split.get(chunk.document_id)
            if not split_name:
                continue

            stats["split_stats"][split_name]["chunks"] += 1

            for ans in chunk.answers:
                span_id = f"{chunk.document_id}_{ans.annotation_id}_{ans.original_start}_{ans.original_end}"
                stats["spans_total_set"].add(span_id)
                stats["split_stats"][split_name]["unique_original_answer_spans"].add(
                    span_id
                )

            windows = window_builder.build_windows_for_chunk(chunk, split_name)
            stats["total_windows"] += len(windows)

            for w in windows:
                # We only yield positive windows currently
                stats["positive_windows"] += 1
                stats["split_stats"][split_name]["qa_examples"] += 1
                stats["split_stats"][split_name]["positive_answer_occurrences"] += 1

                # Deduplicate answer span using original char offsets
                span_id = f"{w.document_id}_{w.annotation_id}_{w.answer.original_char_start}_{w.answer.original_char_end}"
                stats["spans_represented"].add(span_id)
                stats["span_occurrences"][span_id] += 1
                stats["split_stats"][split_name]["covered_spans"].add(span_id)

                f_out.write(w.model_dump_json() + "\n")

    # Clean up stats for JSON
    stats["total_original_answer_spans"] = len(stats["spans_total_set"])
    stats["covered_answer_spans"] = len(stats["spans_represented"])
    stats["uncovered_answer_spans"] = (
        stats["total_original_answer_spans"] - stats["covered_answer_spans"]
    )
    stats["multiple_window_answer_count"] = sum(
        1 for count in stats["span_occurrences"].values() if count > 1
    )

    for k, v in stats["split_stats"].items():
        v["unique_original_answer_spans"] = len(v["unique_original_answer_spans"])
        v["covered_spans"] = len(v["covered_spans"])

    del stats["spans_represented"]
    del stats["spans_total_set"]
    del stats["span_occurrences"]
    with open(reports_dir / "split_report.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Module 7 complete. Reports generated.")


if __name__ == "__main__":
    run()
