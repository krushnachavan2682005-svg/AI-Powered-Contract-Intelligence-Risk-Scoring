"""
Unit tests for the CUAD dataset validator.
"""

from typing import Any, Dict

import pytest

from src.data.cuad_validator import DatasetValidator


@pytest.fixture
def validator() -> DatasetValidator:
    """Return a fresh DatasetValidator."""
    return DatasetValidator()


@pytest.fixture
def valid_dataset() -> Dict[str, Any]:
    """Return a valid minimal synthetic CUAD dataset."""
    return {
        "data": [
            {
                "title": "Contract 1",
                "paragraphs": [
                    {
                        "context": "The quick brown fox.",
                        "qas": [
                            {
                                "id": "1",
                                "question": "termination",
                                "is_impossible": False,
                                "answers": [{"text": "quick brown", "answer_start": 4}],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_valid_answer_span_passes(
    validator: DatasetValidator, valid_dataset: Dict[str, Any]
) -> None:
    """Test that a valid answer span passes validation."""
    validator.validate_json(valid_dataset)
    assert len(validator.errors) == 0


def test_invalid_answer_span_is_detected(
    validator: DatasetValidator, valid_dataset: Dict[str, Any]
) -> None:
    """Test that a span mismatch is detected."""
    valid_dataset["data"][0]["paragraphs"][0]["qas"][0]["answers"][0][
        "text"
    ] = "wrong text"
    validator.validate_json(valid_dataset)
    assert len(validator.errors) == 1
    assert validator.errors[0].validation_type == "span_mismatch"


def test_impossible_with_answers_is_detected(
    validator: DatasetValidator, valid_dataset: Dict[str, Any]
) -> None:
    """Test that impossible annotations with answers are detected."""
    valid_dataset["data"][0]["paragraphs"][0]["qas"][0]["is_impossible"] = True
    validator.validate_json(valid_dataset)
    assert len(validator.errors) == 1
    assert validator.errors[0].validation_type == "impossible_with_answers"


def test_possible_without_answers_is_detected(
    validator: DatasetValidator, valid_dataset: Dict[str, Any]
) -> None:
    """Test that possible annotations without answers are detected."""
    valid_dataset["data"][0]["paragraphs"][0]["qas"][0]["answers"] = []
    validator.validate_json(valid_dataset)
    assert len(validator.errors) == 1
    assert validator.errors[0].validation_type == "possible_without_answers"


def test_duplicate_qa_id(
    validator: DatasetValidator, valid_dataset: Dict[str, Any]
) -> None:
    """Test that duplicate QA IDs are detected."""
    qa_dup = valid_dataset["data"][0]["paragraphs"][0]["qas"][0].copy()
    valid_dataset["data"][0]["paragraphs"][0]["qas"].append(qa_dup)
    validator.validate_json(valid_dataset)
    assert len(validator.errors) > 0
    assert any(e.validation_type == "duplicate_qa_id" for e in validator.errors)


def test_category_inconsistency(
    validator: DatasetValidator, valid_dataset: Dict[str, Any]
) -> None:
    """Test that missing or duplicate categories generate warnings/errors."""
    qa_dup = valid_dataset["data"][0]["paragraphs"][0]["qas"][0].copy()
    qa_dup["id"] = "2"
    valid_dataset["data"][0]["paragraphs"][0]["qas"].append(qa_dup)
    validator.validate_json(valid_dataset)

    assert any(e.validation_type == "duplicate_category" for e in validator.errors)
    assert any(w.validation_type == "missing_categories" for w in validator.warnings)
