import json
from pathlib import Path
from typing import Any

from src.core.exceptions import ApplicationError
from src.data.schemas import AnswerRecord, ClauseRecord, ContractRecord


class ParserError(ApplicationError):
    """Exception raised for errors during dataset parsing and transformation."""

    pass


def parse_answer(raw_answer: dict[str, Any], context: str) -> AnswerRecord:
    """Parses a single raw answer, calculating the end offset and validating."""
    text = raw_answer["text"]
    start = raw_answer["answer_start"]
    end = start + len(text)

    if context[start:end] != text:
        raise ParserError(
            f"Answer offset invariant broken. Expected '{text}', "
            f"found '{context[start:end]}' at [{start}:{end}]"
        )

    return AnswerRecord(text=text, start=start, end=end)


def parse_clause(raw_qa: dict[str, Any], context: str) -> ClauseRecord:
    """Parses a QA dictionary into a ClauseRecord."""
    is_impossible = raw_qa.get("is_impossible", False)

    answers = []
    for raw_answer in raw_qa.get("answers", []):
        answers.append(parse_answer(raw_answer, context))

    return ClauseRecord(
        annotation_id=raw_qa["id"],
        category=raw_qa["question"],
        is_present=not is_impossible,
        answers=answers,
    )


def parse_contract(raw_contract: dict[str, Any]) -> ContractRecord:
    """Parses a single contract, iterating through its paragraphs and QAs."""
    title = raw_contract["title"]
    document_id = title  # Using title directly as the document ID

    paragraphs = raw_contract.get("paragraphs", [])
    if not paragraphs:
        raise ParserError(f"Contract '{title}' has no paragraphs.")

    # CUAD normally has one paragraph per contract
    # We will join them if there are multiple, but adjust offsets carefully.
    # However, CUAD v1 is strictly one paragraph per contract document context.
    # If there are multiple paragraphs, we will fail loudly because offset
    # recalculation across joined contexts is complex and unnecessary for CUAD v1.
    if len(paragraphs) > 1:
        raise ParserError(
            f"Contract '{title}' has multiple paragraphs. "
            "Multi-paragraph offset adjustment is not supported."
        )

    paragraph = paragraphs[0]
    context = paragraph.get("context", "")

    clauses = []
    for raw_qa in paragraph.get("qas", []):
        clauses.append(parse_clause(raw_qa, context))

    return ContractRecord(
        document_id=document_id,
        title=title,
        context=context,
        clauses=clauses,
    )


def parse_cuad_dataset(filepath: str | Path) -> list[ContractRecord]:
    """Loads and parses the entire raw CUAD dataset."""
    path = Path(filepath)
    if not path.exists():
        raise ParserError(f"Raw dataset file missing: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ParserError(f"Malformed JSON in dataset: {e}")

    raw_contracts = raw_data.get("data", [])
    if not isinstance(raw_contracts, list):
        raise ParserError("Malformed root structure: 'data' is not a list.")

    contracts = []
    seen_ids = set()

    for raw_contract in raw_contracts:
        contract = parse_contract(raw_contract)

        if contract.document_id in seen_ids:
            raise ParserError(
                f"Duplicate generated document ID: {contract.document_id}"
            )

        seen_ids.add(contract.document_id)
        contracts.append(contract)

    return contracts
