"""
Module to validate the integrity and quality of the raw CUAD dataset.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ValidationIssue:
    def __init__(
        self, severity: str, validation_type: str, message: str, **kwargs: Any
    ) -> None:
        self.severity = severity
        self.validation_type = validation_type
        self.message = message
        self.details = kwargs

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "severity": self.severity,
            "validation_type": self.validation_type,
            "message": self.message,
        }
        d.update(self.details)
        return d


class DatasetValidator:
    def __init__(self) -> None:
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []
        self.stats: Dict[str, Any] = {
            "contracts_checked": 0,
            "paragraphs_checked": 0,
            "annotations_checked": 0,
            "answers_checked": 0,
        }

    def add_error(self, validation_type: str, message: str, **kwargs: Any) -> None:
        self.errors.append(ValidationIssue("error", validation_type, message, **kwargs))

    def add_warning(self, validation_type: str, message: str, **kwargs: Any) -> None:
        self.warnings.append(
            ValidationIssue("warning", validation_type, message, **kwargs)
        )

    def validate_json(self, data: Any) -> bool:
        if not isinstance(data, dict):
            self.add_error("root_structure", "JSON root is not a dictionary.")
            return False
        if "data" not in data:
            self.add_error("root_structure", "Missing 'data' key in JSON root.")
            return False
        if not isinstance(data["data"], list):
            self.add_error("root_structure", "'data' field is not a list.")
            return False

        contracts = data["data"]
        global_categories: set[str] = set()

        for contract_idx, contract in enumerate(contracts):
            self.stats["contracts_checked"] += 1
            if not isinstance(contract, dict):
                self.add_error(
                    "contract_structure",
                    "Contract is not a dictionary.",
                    contract_index=contract_idx,
                )
                continue

            title = contract.get("title")
            if title is None:
                self.add_error(
                    "missing_title",
                    "Contract missing title.",
                    contract_index=contract_idx,
                )
            elif not isinstance(title, str) or not title.strip():
                self.add_error(
                    "empty_title",
                    "Contract has empty or invalid title.",
                    contract_index=contract_idx,
                )

            paragraphs = contract.get("paragraphs")
            if paragraphs is None:
                self.add_error(
                    "missing_paragraphs",
                    "Contract missing paragraphs.",
                    contract_index=contract_idx,
                    title=title,
                )
                continue
            if not isinstance(paragraphs, list):
                self.add_error(
                    "invalid_paragraphs",
                    "Paragraphs field is not a list.",
                    contract_index=contract_idx,
                    title=title,
                )
                continue
            if len(paragraphs) == 0:
                self.add_error(
                    "zero_paragraphs",
                    "Contract has zero paragraphs.",
                    contract_index=contract_idx,
                    title=title,
                )
                continue

            contract_categories: set[str] = set()

            for para_idx, para in enumerate(paragraphs):
                self.stats["paragraphs_checked"] += 1
                context = para.get("context")
                if context is None:
                    self.add_error(
                        "missing_context",
                        "Paragraph missing context.",
                        contract_index=contract_idx,
                        title=title,
                        paragraph_index=para_idx,
                    )
                    continue
                if not isinstance(context, str):
                    self.add_error(
                        "invalid_context",
                        "Context is not a string.",
                        contract_index=contract_idx,
                        title=title,
                        paragraph_index=para_idx,
                    )
                    continue
                if not context.strip():
                    self.add_warning(
                        "empty_context",
                        "Context is empty or whitespace-only.",
                        contract_index=contract_idx,
                        title=title,
                        paragraph_index=para_idx,
                    )

                qas = para.get("qas")
                if qas is None:
                    self.add_error(
                        "missing_qas",
                        "Paragraph missing qas field.",
                        contract_index=contract_idx,
                        title=title,
                        paragraph_index=para_idx,
                    )
                    continue
                if not isinstance(qas, list):
                    self.add_error(
                        "invalid_qas",
                        "qas field is not a list.",
                        contract_index=contract_idx,
                        title=title,
                        paragraph_index=para_idx,
                    )
                    continue
                if len(qas) == 0:
                    self.add_warning(
                        "empty_qas",
                        "Paragraph has zero annotations.",
                        contract_index=contract_idx,
                        title=title,
                        paragraph_index=para_idx,
                    )

                qa_ids: set[str] = set()
                for qa_idx, qa in enumerate(qas):
                    self.stats["annotations_checked"] += 1
                    qa_id = qa.get("id")
                    if not qa_id or not str(qa_id).strip():
                        self.add_error(
                            "missing_qa_id",
                            "QA missing ID.",
                            contract_index=contract_idx,
                            title=title,
                            paragraph_index=para_idx,
                            qa_index=qa_idx,
                        )
                    else:
                        if qa_id in qa_ids:
                            self.add_error(
                                "duplicate_qa_id",
                                f"Duplicate QA ID found: {qa_id}",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )
                        qa_ids.add(str(qa_id))

                    question = qa.get("question")
                    if not question or not str(question).strip():
                        self.add_error(
                            "missing_question",
                            "QA missing question.",
                            contract_index=contract_idx,
                            title=title,
                            qa_id=qa_id,
                        )
                    else:
                        global_categories.add(str(question))
                        if question in contract_categories:
                            self.add_error(
                                "duplicate_category",
                                f"Duplicate category in contract: {question}",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )
                        contract_categories.add(str(question))

                    is_impossible = qa.get("is_impossible")
                    if not isinstance(is_impossible, bool):
                        self.add_error(
                            "invalid_is_impossible",
                            "is_impossible must be a boolean.",
                            contract_index=contract_idx,
                            title=title,
                            qa_id=qa_id,
                        )

                    answers = qa.get("answers")
                    if not isinstance(answers, list):
                        self.add_error(
                            "invalid_answers",
                            "answers must be a list.",
                            contract_index=contract_idx,
                            title=title,
                            qa_id=qa_id,
                        )
                        continue

                    if is_impossible:
                        if len(answers) > 0:
                            self.add_error(
                                "impossible_with_answers",
                                "is_impossible is True but answers exist.",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )
                    else:
                        if len(answers) == 0:
                            self.add_error(
                                "possible_without_answers",
                                "is_impossible is False but no answers exist.",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )

                    for ans_idx, ans in enumerate(answers):
                        self.stats["answers_checked"] += 1
                        ans_text = ans.get("text")
                        ans_start = ans.get("answer_start")

                        if ans_text is None or not isinstance(ans_text, str):
                            self.add_error(
                                "invalid_answer_text",
                                "Answer text is missing or invalid.",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )
                            continue
                        if not ans_text.strip():
                            self.add_warning(
                                "empty_answer_text",
                                "Answer text is empty or whitespace.",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )

                        if (
                            ans_start is None
                            or not isinstance(ans_start, int)
                            or ans_start < 0
                        ):
                            self.add_error(
                                "invalid_answer_start",
                                "Answer start is invalid or missing.",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                            )
                            continue

                        extracted = context[ans_start : ans_start + len(ans_text)]
                        if extracted != ans_text:
                            self.add_error(
                                "span_mismatch",
                                "Answer text does not match context span.",
                                contract_index=contract_idx,
                                title=title,
                                qa_id=qa_id,
                                expected=ans_text,
                                actual=extracted,
                                start=ans_start,
                            )

            if len(contract_categories) != 41:
                self.add_warning(
                    "missing_categories",
                    f"Contract has {len(contract_categories)} categories, not 41.",
                    contract_index=contract_idx,
                    title=title,
                )

        self.stats["global_categories"] = list(global_categories)
        self.stats["total_global_categories"] = len(global_categories)
        return True

    def validate_csv(self, csv_data: List[List[str]]) -> None:
        if not csv_data:
            self.add_error("csv_structure", "CSV data is empty.")
            return

        header = csv_data[0]
        if len(set(header)) != len(header):
            self.add_error("csv_structure", "CSV header has duplicate columns.")

        row_count = len(csv_data) - 1
        filenames: set[str] = set()

        for row_idx, row in enumerate(csv_data[1:], start=1):
            if not any(val.strip() for val in row):
                self.add_warning(
                    "empty_csv_row", f"CSV row {row_idx} is entirely empty."
                )

            if row:
                filename = row[0]
                if filename in filenames:
                    self.add_error(
                        "csv_duplicate_row", f"Duplicate filename in CSV: {filename}"
                    )
                filenames.add(filename)

        self.stats["csv_rows"] = row_count
        self.stats["csv_unique_filenames"] = len(filenames)

    def validate_cross_consistency(
        self, json_data: Any, csv_data: List[List[str]]
    ) -> None:
        if not isinstance(json_data, dict) or "data" not in json_data or not csv_data:
            return

        json_titles = {
            c.get("title", "") for c in json_data["data"] if isinstance(c, dict)
        }
        csv_titles = set()
        for row in csv_data[1:]:
            if row:
                csv_titles.add(row[0])

        json_only = json_titles - csv_titles
        csv_only = csv_titles - json_titles
        matched = json_titles.intersection(csv_titles)

        self.stats["cross_consistency"] = {
            "json_titles_count": len(json_titles),
            "csv_titles_count": len(csv_titles),
            "matched_count": len(matched),
            "json_only_count": len(json_only),
            "csv_only_count": len(csv_only),
        }

        if json_only or csv_only:
            self.add_warning(
                "cross_consistency",
                "Mismatch between JSON titles and CSV filenames. "
                "They do not perfectly map 1-to-1.",
            )

    def get_report(self) -> Dict[str, Any]:
        return {
            "dataset_valid": len(self.errors) == 0,
            "summary": self.stats,
            "errors": {
                "count": len(self.errors),
            },
            "warnings": {
                "count": len(self.warnings),
            },
        }
