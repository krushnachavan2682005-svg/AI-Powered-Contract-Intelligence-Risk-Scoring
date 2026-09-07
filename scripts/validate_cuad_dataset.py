"""
Script to validate the CUAD dataset and generate quality reports.
"""

import csv
import json
import logging
from pathlib import Path

from src.core.logging import setup_logging
from src.data.cuad_inspector import load_json
from src.data.cuad_validator import DatasetValidator


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    json_path = Path("data/raw/CUAD_v1.json")
    csv_path = Path("data/raw/master_clauses.csv")
    output_dir = Path("reports/dataset_validation")

    if not json_path.exists() or not csv_path.exists():
        logger.error("Dataset files not found in data/raw/")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dataset validation...")
    data = load_json(str(json_path))

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        csv_data = list(reader)

    validator = DatasetValidator()
    logger.info("Validating JSON structure and annotations...")
    validator.validate_json(data)

    logger.info("Validating CSV structure...")
    validator.validate_csv(csv_data)

    logger.info("Validating JSON <-> CSV cross-consistency...")
    validator.validate_cross_consistency(data, csv_data)

    report = validator.get_report()

    # Save validation_report.json
    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # Save validation_errors.json
    with open(output_dir / "validation_errors.json", "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in validator.errors], f, indent=4)

    # Save validation_warnings.json
    with open(output_dir / "validation_warnings.json", "w", encoding="utf-8") as f:
        json.dump([w.to_dict() for w in validator.warnings], f, indent=4)

    # Generate Markdown report
    # noqa: E501
    status = "VALID"
    if validator.errors:
        status = "INVALID"
    elif validator.warnings:
        status = "VALID WITH WARNINGS"

    report_content = f"""# CUAD Dataset Validation Report

## Overall Status
{status}

## Files Validated
- CUAD_v1.json
- master_clauses.csv

## Validation Summary
- Contracts checked: {report["summary"]["contracts_checked"]}
- Paragraphs checked: {report["summary"]["paragraphs_checked"]}
- Annotations checked: {report["summary"]["annotations_checked"]}
- Answers checked: {report["summary"]["answers_checked"]}
- Global clause categories discovered: {report["summary"].get("total_global_categories")}

## JSON Structural Integrity
- Valid JSON schema: {'Yes' if not any(e.validation_type == 'root_structure' for e in validator.errors) else 'No'}

## Annotation Integrity
- Malformed annotations: {sum(1 for e in validator.errors if e.validation_type in ['missing_qa_id', 'missing_question'])}
- Impossible answers inconsistency: {sum(1 for e in validator.errors if e.validation_type == 'impossible_with_answers')}
- Possible answers inconsistency: {sum(1 for e in validator.errors if e.validation_type == 'possible_without_answers')}
- Duplicate IDs: {sum(1 for e in validator.errors if e.validation_type == 'duplicate_qa_id')}

## Answer Span Integrity
- Total answer spans checked: {report["summary"]["answers_checked"]}
- Invalid answer spans: {sum(1 for e in validator.errors if e.validation_type == 'invalid_answer_start')}
- Span mismatches: {sum(1 for e in validator.errors if e.validation_type == 'span_mismatch')}

## Category Consistency
- Expected 41 categories per contract.
- Contracts with missing categories: {sum(1 for w in validator.warnings if w.validation_type == 'missing_categories')}
- Contracts with duplicate categories: {sum(1 for e in validator.errors if e.validation_type == 'duplicate_category')}

## Duplicate Analysis
- Duplicate CSV Rows (by filename): {sum(1 for e in validator.errors if e.validation_type == 'csv_duplicate_row')}

## CSV Validation
- Rows checked: {report["summary"].get("csv_rows")}
- Empty rows: {sum(1 for w in validator.warnings if w.validation_type == 'empty_csv_row')}

## JSON <-> CSV Consistency
- JSON Titles: {report["summary"].get("cross_consistency", {}).get("json_titles_count")}
- CSV Titles: {report["summary"].get("cross_consistency", {}).get("csv_titles_count")}
- Matched exactly: {report["summary"].get("cross_consistency", {}).get("matched_count")}
- JSON only: {report["summary"].get("cross_consistency", {}).get("json_only_count")}
- CSV only: {report["summary"].get("cross_consistency", {}).get("csv_only_count")}

## Errors
Total: {len(validator.errors)}
(See `validation_errors.json` for details)

## Warnings
Total: {len(validator.warnings)}
(See `validation_warnings.json` for details)

## Final Dataset Readiness Assessment
Is this dataset ready for downstream parsing and transformation?
{"Yes, the dataset structure is well-formed with no critical structural errors preventing downstream parsing." if not validator.errors else "No, there are critical validation errors that must be investigated."}
"""
    with open(output_dir / "validation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\nCUAD Dataset Validation Complete")
    print(f"Contracts checked: {report['summary']['contracts_checked']}")
    print(f"Annotations checked: {report['summary']['annotations_checked']}")
    print(f"Answer spans checked: {report['summary']['answers_checked']}")
    print()
    print(f"Errors: {len(validator.errors)}")
    print(f"Warnings: {len(validator.warnings)}")
    print()
    print(f"Final Status: {status}")
    print("Report location: reports/dataset_validation/\n")


if __name__ == "__main__":
    main()
