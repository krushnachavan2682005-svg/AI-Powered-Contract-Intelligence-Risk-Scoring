import json
from pathlib import Path

from src.data.cuad_parser import parse_cuad_dataset


def main() -> None:
    raw_dataset_path = Path("data/raw/CUAD_v1.json")
    interim_dir = Path("data/interim/cuad")
    output_jsonl_path = interim_dir / "contracts.jsonl"
    summary_path = interim_dir / "transformation_summary.json"

    print("Parsing CUAD dataset...")
    try:
        contracts = parse_cuad_dataset(raw_dataset_path)
    except Exception as e:
        print(f"Error parsing dataset: {e}")
        return

    # Ensure output directory exists
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Invariants checks tracking
    total_clauses = 0
    total_present_clauses = 0
    total_absent_clauses = 0
    total_answer_spans = 0
    unique_categories = set()

    # Write JSONL
    print(f"Writing canonical records to {output_jsonl_path}...")
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for contract in contracts:
            f.write(contract.model_dump_json() + "\n")

            for clause in contract.clauses:
                total_clauses += 1
                unique_categories.add(clause.category)
                if clause.is_present:
                    total_present_clauses += 1
                else:
                    total_absent_clauses += 1

                total_answer_spans += len(clause.answers)

    # Prepare and write summary
    summary = {
        "total_parsed_contracts": len(contracts),
        "total_clauses": total_clauses,
        "total_present_clauses": total_present_clauses,
        "total_absent_clauses": total_absent_clauses,
        "total_answer_spans": total_answer_spans,
        "unique_categories": len(unique_categories),
        "output_format": "JSON Lines (.jsonl)",
        "document_id_strategy": "Raw contract title",
        "source_dataset_path": str(raw_dataset_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nCUAD Dataset Transformation Complete")
    print("-" * 40)
    print(f"Contracts parsed: {len(contracts)}")
    print(f"Clauses parsed: {total_clauses:,}")
    print(f"Present clauses: {total_present_clauses:,}")
    print(f"Absent clauses: {total_absent_clauses:,}")
    print(f"Answer spans: {total_answer_spans:,}")
    print(f"\nOutput:\n{output_jsonl_path}")
    print(f"{summary_path}")


if __name__ == "__main__":
    main()
