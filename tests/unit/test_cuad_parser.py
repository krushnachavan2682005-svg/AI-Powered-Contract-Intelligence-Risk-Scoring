import json
from pathlib import Path

import pytest

from src.data.cuad_parser import (
    ParserError,
    parse_answer,
    parse_clause,
    parse_contract,
    parse_cuad_dataset,
)


def test_parse_answer_valid() -> None:
    raw_answer = {"text": "State of Delaware", "answer_start": 10}
    context = "0123456789State of Delaware - more text"

    answer = parse_answer(raw_answer, context)

    assert answer.text == "State of Delaware"
    assert answer.start == 10
    assert answer.end == 27


def test_parse_answer_invalid_offset() -> None:
    raw_answer = {"text": "State of Delaware", "answer_start": 9}  # wrong start
    context = "0123456789State of Delaware - more text"

    with pytest.raises(ParserError, match="Answer offset invariant broken"):
        parse_answer(raw_answer, context)


def test_parse_clause_present() -> None:
    raw_qa = {
        "id": "123",
        "question": "Governing Law",
        "is_impossible": False,
        "answers": [{"text": "Delaware", "answer_start": 0}],
    }
    context = "Delaware is the state."

    clause = parse_clause(raw_qa, context)

    assert clause.annotation_id == "123"
    assert clause.category == "Governing Law"
    assert clause.is_present is True
    assert len(clause.answers) == 1
    assert clause.answers[0].text == "Delaware"


def test_parse_clause_absent() -> None:
    raw_qa = {
        "id": "124",
        "question": "Non-Compete",
        "is_impossible": True,
        "answers": [],
    }
    context = "No non-compete here."

    clause = parse_clause(raw_qa, context)

    assert clause.annotation_id == "124"
    assert clause.category == "Non-Compete"
    assert clause.is_present is False
    assert len(clause.answers) == 0


def test_parse_contract_single_paragraph() -> None:
    raw_contract = {
        "title": "Contract A",
        "paragraphs": [
            {
                "context": "This is a contract.",
                "qas": [
                    {
                        "id": "1",
                        "question": "Type",
                        "is_impossible": False,
                        "answers": [{"text": "contract", "answer_start": 10}],
                    }
                ],
            }
        ],
    }

    contract = parse_contract(raw_contract)

    assert contract.document_id == "Contract A"
    assert contract.title == "Contract A"
    assert contract.context == "This is a contract."
    assert len(contract.clauses) == 1


def test_parse_contract_multiple_paragraphs() -> None:
    raw_contract = {
        "title": "Contract B",
        "paragraphs": [{"context": "P1", "qas": []}, {"context": "P2", "qas": []}],
    }

    with pytest.raises(ParserError, match="multiple paragraphs"):
        parse_contract(raw_contract)


def test_parse_cuad_dataset_duplicate_titles(tmp_path: Path) -> None:
    data = {
        "data": [
            {"title": "Same Title", "paragraphs": [{"context": "A", "qas": []}]},
            {"title": "Same Title", "paragraphs": [{"context": "B", "qas": []}]},
        ]
    }

    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ParserError, match="Duplicate generated document ID"):
        parse_cuad_dataset(file_path)


def test_parse_cuad_dataset_valid(tmp_path: Path) -> None:
    data = {
        "data": [
            {"title": "Doc 1", "paragraphs": [{"context": "A", "qas": []}]},
            {"title": "Doc 2", "paragraphs": [{"context": "B", "qas": []}]},
        ]
    }

    file_path = tmp_path / "test.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    contracts = parse_cuad_dataset(file_path)
    assert len(contracts) == 2
    assert contracts[0].document_id == "Doc 1"
    assert contracts[1].document_id == "Doc 2"
