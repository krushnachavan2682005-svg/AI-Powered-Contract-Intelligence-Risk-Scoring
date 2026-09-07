"""
CUAD Dataset Inspector.

Provides functions to analyze CUAD_v1.json and master_clauses.csv
without modifying the raw data.
"""

import csv
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON data from a filepath."""
    logger.info(f"Loading JSON from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, dict)
        return data


def inspect_json_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect the top-level structure of the JSON dataset."""
    structure: Dict[str, Any] = {
        "root_type": type(data).__name__,
        "root_keys": list(data.keys()),
    }
    if "version" in data:
        structure["version"] = data["version"]

    if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
        structure["total_contracts"] = len(data["data"])
        first_contract = data["data"][0]
        structure["contract_keys"] = list(first_contract.keys())
        if "paragraphs" in first_contract and len(first_contract["paragraphs"]) > 0:
            first_paragraph = first_contract["paragraphs"][0]
            structure["paragraph_keys"] = list(first_paragraph.keys())
            if "qas" in first_paragraph and len(first_paragraph["qas"]) > 0:
                first_qa = first_paragraph["qas"][0]
                structure["qa_keys"] = list(first_qa.keys())
                if "answers" in first_qa and len(first_qa["answers"]) > 0:
                    first_answer = first_qa["answers"][0]
                    structure["answer_keys"] = list(first_answer.keys())
    return structure


def inspect_contract_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics such as lengths and counts."""
    total_contracts = 0
    total_paragraphs = 0
    total_qas = 0

    context_lengths_chars: List[int] = []
    context_lengths_words: List[int] = []

    for contract in data.get("data", []):
        total_contracts += 1
        for paragraph in contract.get("paragraphs", []):
            total_paragraphs += 1
            context = paragraph.get("context", "")
            context_lengths_chars.append(len(context))
            context_lengths_words.append(len(context.split()))
            total_qas += len(paragraph.get("qas", []))

    stats: Dict[str, Any] = {
        "total_contracts": total_contracts,
        "total_paragraphs": total_paragraphs,
        "total_annotations": total_qas,
    }

    if context_lengths_chars:
        context_lengths_chars.sort()
        stats["context_length_chars"] = {
            "min": min(context_lengths_chars),
            "max": max(context_lengths_chars),
            "mean": sum(context_lengths_chars) / len(context_lengths_chars),
            "median": context_lengths_chars[len(context_lengths_chars) // 2],
        }
    if context_lengths_words:
        context_lengths_words.sort()
        stats["context_length_words"] = {
            "min": min(context_lengths_words),
            "max": max(context_lengths_words),
            "mean": sum(context_lengths_words) / len(context_lengths_words),
            "median": context_lengths_words[len(context_lengths_words) // 2],
        }

    return stats


def extract_clause_categories(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and analyze the clause categories encoded as questions."""
    categories: Dict[str, Dict[str, int]] = {}
    for contract in data.get("data", []):
        for paragraph in contract.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                is_impossible = qa.get("is_impossible", False)
                if question not in categories:
                    categories[question] = {"total": 0, "positive": 0, "empty": 0}
                categories[question]["total"] += 1
                if is_impossible or len(qa.get("answers", [])) == 0:
                    categories[question]["empty"] += 1
                else:
                    categories[question]["positive"] += 1

    return {
        "total_unique_categories": len(categories),
        "categories": categories,
    }


def inspect_csv_structure(filepath: str) -> Dict[str, Any]:
    """Inspect the master_clauses.csv structure."""
    logger.info(f"Loading CSV from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {}

        row_count = 0
        missing_values = {col: 0 for col in header}
        filenames = set()
        duplicates = 0

        for row in reader:
            row_count += 1
            row = row + [""] * (len(header) - len(row))
            if row[0] in filenames:
                duplicates += 1
            filenames.add(row[0])
            for i, val in enumerate(row):
                if not val.strip():
                    if i < len(header):
                        missing_values[header[i]] += 1

    return {
        "columns": header,
        "row_count": row_count,
        "missing_values": missing_values,
        "duplicates": duplicates,
    }
