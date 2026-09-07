"""
Unit tests for the CUAD inspector module.
"""

from typing import Any, Dict

import pytest

from src.data.cuad_inspector import (
    extract_clause_categories,
    inspect_contract_statistics,
    inspect_json_structure,
)


@pytest.fixture
def mock_cuad_data() -> Dict[str, Any]:
    """Fixture providing a minimal mock CUAD dataset."""
    return {
        "version": "1.0",
        "data": [
            {
                "title": "Contract 1",
                "paragraphs": [
                    {
                        "context": "This is a short contract text.",
                        "qas": [
                            {
                                "id": "1",
                                "question": "termination",
                                "is_impossible": False,
                                "answers": [
                                    {"text": "short contract", "answer_start": 10}
                                ],
                            },
                            {
                                "id": "2",
                                "question": "confidentiality",
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_inspect_json_structure(mock_cuad_data: Dict[str, Any]) -> None:
    """Test top-level structure inspection."""
    structure = inspect_json_structure(mock_cuad_data)
    assert structure["root_type"] == "dict"
    assert "version" in structure["root_keys"]
    assert "data" in structure["root_keys"]
    assert structure["total_contracts"] == 1
    assert "paragraphs" in structure["contract_keys"]


def test_inspect_contract_statistics(mock_cuad_data: Dict[str, Any]) -> None:
    """Test calculation of contract statistics."""
    stats = inspect_contract_statistics(mock_cuad_data)
    assert stats["total_contracts"] == 1
    assert stats["total_paragraphs"] == 1
    assert stats["total_annotations"] == 2

    # "This is a short contract text." -> 30 chars, 6 words
    assert stats["context_length_chars"]["min"] == 30
    assert stats["context_length_words"]["min"] == 6


def test_extract_clause_categories(mock_cuad_data: Dict[str, Any]) -> None:
    """Test extraction of clause categories."""
    categories_res = extract_clause_categories(mock_cuad_data)
    assert categories_res["total_unique_categories"] == 2
    cats = categories_res["categories"]
    assert cats["termination"]["total"] == 1
    assert cats["termination"]["positive"] == 1
    assert cats["termination"]["empty"] == 0

    assert cats["confidentiality"]["total"] == 1
    assert cats["confidentiality"]["positive"] == 0
    assert cats["confidentiality"]["empty"] == 1
