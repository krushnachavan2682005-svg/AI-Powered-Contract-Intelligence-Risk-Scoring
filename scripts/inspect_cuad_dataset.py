"""
Script to inspect the CUAD dataset and generate reports.
"""

import json
import logging
from pathlib import Path

from src.core.logging import setup_logging
from src.data.cuad_inspector import (
    extract_clause_categories,
    inspect_contract_statistics,
    inspect_csv_structure,
    inspect_json_structure,
    load_json,
)


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    json_path = Path("data/raw/CUAD_v1.json")
    csv_path = Path("data/raw/master_clauses.csv")
    output_dir = Path("reports/dataset_understanding")

    if not json_path.exists() or not csv_path.exists():
        logger.error("Dataset files not found in data/raw/")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dataset inspection...")
    data = load_json(str(json_path))

    structure = inspect_json_structure(data)
    stats = inspect_contract_statistics(data)
    categories = extract_clause_categories(data)
    csv_stats = inspect_csv_structure(str(csv_path))

    # Save dataset_summary.json
    summary = {
        "json_structure": structure,
        "statistics": stats,
        "csv_statistics": {
            "columns_count": len(csv_stats.get("columns", [])),
            "row_count": csv_stats.get("row_count"),
            "duplicates": csv_stats.get("duplicates"),
            "missing_values": csv_stats.get("missing_values"),
        },
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Save clause_categories.json
    with open(output_dir / "clause_categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=4)

    # Generate Markdown report
    # noqa: E501
    report_content = f"""# CUAD Dataset Understanding Report

## Dataset Files
- **CUAD_v1.json**: Contains the contracts, paragraphs, and annotations in SQuAD format.
- **master_clauses.csv**: Contains {csv_stats.get('row_count')} rows and {len(csv_stats.get('columns', []))} columns detailing the clauses. 

## JSON Schema
The dataset follows a SQuAD v2 format:
- Root is a `{structure.get('root_type')}` with keys: {structure.get('root_keys')}
- Contracts contain keys: {structure.get('contract_keys')}
- Paragraphs contain keys: {structure.get('paragraph_keys')}
- QAs contain keys: {structure.get('qa_keys')}
- Answers contain keys: {structure.get('answer_keys')}

## Dataset Statistics
- Total Contracts: {stats.get('total_contracts')}
- Total Paragraphs: {stats.get('total_paragraphs')}
- Total Annotations: {stats.get('total_annotations')}

## Clause Categories
- Total Unique Clause Categories: {categories.get('total_unique_categories')}
(See `clause_categories.json` for details)

## Annotation Format
Annotations are encoded as questions (clause categories). If a clause exists in the text, it has an answer with `text` and `answer_start`. Otherwise, it is marked with `is_impossible: true` and has an empty answer list.

## Data Quality Observations
- Duplicate rows in CSV: {csv_stats.get('duplicates')}
- Missing values exist heavily in CSV columns indicating sparsity.

## Contract Length Analysis
Context length limits are a major consideration for transformer models.
- **Character Lengths**: Min: {stats.get('context_length_chars', {}).get('min')}, Max: {stats.get('context_length_chars', {}).get('max')}, Mean: {stats.get('context_length_chars', {}).get('mean'):.2f}, Median: {stats.get('context_length_chars', {}).get('median')}
- **Word Lengths**: Min: {stats.get('context_length_words', {}).get('min')}, Max: {stats.get('context_length_words', {}).get('max')}, Mean: {stats.get('context_length_words', {}).get('mean'):.2f}, Median: {stats.get('context_length_words', {}).get('median')}

### Implication for Transformers:
The maximum word length is {stats.get('context_length_words', {}).get('max')}, which translates to even more subword tokens. This far exceeds the typical 512-token limit of models like BERT or RoBERTa. We will need long-document strategies like chunking (sliding window) or Longformer.

## Relationship Between JSON and CSV
The JSON file is tailored for Question Answering and Span Extraction tasks, pairing contract text with specific clauses. The CSV file acts as a document-level metadata summary, offering a broad overview of which clauses exist in which document without the precise span locations.
"""
    with open(output_dir / "dataset_inspection_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(
        "Dataset inspection complete. Reports generated in reports/dataset_understanding/"
    )


if __name__ == "__main__":
    main()
