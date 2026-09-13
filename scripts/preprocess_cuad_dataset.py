import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.data.schemas import ContractRecord
from src.data.legal_preprocessor import LegalPreprocessor, NormalizationPolicy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    input_path = Path("data/interim/cuad/contracts.jsonl")
    output_path = Path("data/processed/cuad/contracts_normalized.jsonl")
    summary_path = Path("data/processed/cuad/preprocessing_summary.json")
    report_json_path = Path("reports/preprocessing/normalization_report.json")
    report_md_path = Path("reports/preprocessing/normalization_report.md")

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    policy = NormalizationPolicy(replace_control_characters=False)
    preprocessor = LegalPreprocessor(policy)

    stats = {
        "total_contracts": 0,
        "modified_contracts": 0,
        "unchanged_contracts": 0,
        "contracts_with_crlf": 0,
        "contracts_with_cr": 0,
        "contracts_with_lf": 0,
        "contracts_with_control_chars": 0,
        "contracts_with_non_ascii": 0,
        "contracts_with_tabs": 0,
        "contracts_with_repeated_whitespace": 0,
        "answer_spans_verified": 0,
        "answer_offset_failures": 0,
        "total_clauses": 0
    }

    processed_records = []

    logging.info(f"Starting legal-safe preprocessing from {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            data = json.loads(line)
            record = ContractRecord(**data)
            stats["total_contracts"] += 1
            
            # Analyze text
            analysis = preprocessor.analyze_text(record.context)
            if analysis.has_crlf: stats["contracts_with_crlf"] += 1
            if analysis.has_cr: stats["contracts_with_cr"] += 1
            if analysis.has_lf: stats["contracts_with_lf"] += 1
            if analysis.has_control_chars: stats["contracts_with_control_chars"] += 1
            if analysis.has_non_ascii: stats["contracts_with_non_ascii"] += 1
            if analysis.has_tabs: stats["contracts_with_tabs"] += 1
            if analysis.has_repeated_whitespace: stats["contracts_with_repeated_whitespace"] += 1

            # Count answers and clauses before
            for clause in record.clauses:
                stats["total_clauses"] += 1
                stats["answer_spans_verified"] += len(clause.answers)
            
            try:
                processed_record = preprocessor.normalize_contract(record)
                processed_records.append(processed_record)
                
                if processed_record.normalized_context is not None:
                    stats["modified_contracts"] += 1
                else:
                    stats["unchanged_contracts"] += 1
            except ValueError as e:
                logging.error(str(e))
                stats["answer_offset_failures"] += 1
                exit(1)

    # Write normalized jsonl
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in processed_records:
            f.write(rec.model_dump_json() + "\n")

    summary = {
        "metrics": stats,
        "policy": policy.model_dump()
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Generate MD report
    md_content = f"""# Legal Text Normalization Report

## Objective
Legal language is highly sensitive to negations, punctuation, capitalization, dates, monetary values, section references, and numbering. Therefore, transformations must preserve legal semantics. This pipeline ensures strict conservative preprocessing without corrupting the canonical dataset.

## Input Dataset
`{input_path}`

## Text Artifact Analysis
- Line Endings (CRLF): {stats['contracts_with_crlf']} documents
- Line Endings (LF only): {stats['contracts_with_lf']} documents
- Control Characters: {stats['contracts_with_control_chars']} documents
- Non-ASCII Characters: {stats['contracts_with_non_ascii']} documents
- Tabs: {stats['contracts_with_tabs']} documents
- Repeated Whitespace: {stats['contracts_with_repeated_whitespace']} documents

## Normalization Policy

### Applied
- Verification of answer offsets mapping against the original boundaries.

### Intentionally Not Applied
- Lowercasing: Destroys legal terms (e.g. "Company" vs "company").
- Punctuation removal: Alters clause meaning (e.g. "shall not, terminate").
- Whitespace collapsing: Changes string character length and invalidates ground truth label offsets.
- Control character replacement: Opted out for this pass to guarantee offset preservation.

## Offset Integrity
- Answer spans checked: {stats['answer_spans_verified']}
- Valid answer spans: {stats['answer_spans_verified']}
- Invalid answer spans: {stats['answer_offset_failures']}

## Output Dataset
Processed schema with identical structural fidelity. Output location: `{output_path}`

## Final Readiness
Dataset is fully canonical, checked, and safe for long-document chunking and model-specific tokenization steps.
"""

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("\nLegal-Safe Preprocessing Complete\n")
    print(f"Contracts processed: {stats['total_contracts']}")
    print(f"Contracts modified: {stats['modified_contracts']}")
    print(f"Contracts unchanged: {stats['unchanged_contracts']}\n")
    print(f"Answer spans verified: {stats['answer_spans_verified']}")
    print(f"Offset failures: {stats['answer_offset_failures']}\n")
    print(f"Output:\n{output_path}")

if __name__ == "__main__":
    main()
