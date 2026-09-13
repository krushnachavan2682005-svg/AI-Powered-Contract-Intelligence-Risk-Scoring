import re
from typing import List, Optional

from pydantic import BaseModel, Field

from src.data.schemas import ClauseRecord, ContractRecord


class NormalizationPolicy(BaseModel):
    """
    Defines the policy for legal text normalization.
    By default, all options that could alter text length and offset mappings are disabled.
    """

    preserve_case: bool = True
    preserve_punctuation: bool = True
    preserve_numbers: bool = True
    preserve_whitespace_offsets: bool = True
    preserve_unicode: bool = True
    replace_control_characters: bool = False


class ProcessedContractRecord(BaseModel):
    """
    Processed representation of a canonical contract document.
    To avoid duplicating massive amounts of text, `normalized_context`
    is only populated if it differs from `original_context`.
    """

    document_id: str
    title: str
    original_context: str
    normalized_context: Optional[str] = None
    clauses: List[ClauseRecord] = Field(default_factory=list)


class DocumentAnalysis(BaseModel):
    has_lf: bool = False
    has_crlf: bool = False
    has_cr: bool = False
    has_control_chars: bool = False
    has_non_ascii: bool = False
    has_tabs: bool = False
    has_repeated_whitespace: bool = False
    empty_or_whitespace_only: bool = False
    has_dates: bool = False
    has_currency: bool = False
    has_section_references: bool = False


class LegalPreprocessor:
    """
    Legal-Safe Text Normalization Preprocessor.
    Ensures absolute offset integrity for extracted answer spans.
    """

    def __init__(self, policy: Optional[NormalizationPolicy] = None):
        self.policy = policy or NormalizationPolicy()

    def analyze_text(self, text: str) -> DocumentAnalysis:
        analysis = DocumentAnalysis()
        if not text or not text.strip():
            analysis.empty_or_whitespace_only = True
            return analysis

        analysis.has_crlf = "\r\n" in text
        # Match \r not followed by \n
        analysis.has_cr = bool(re.search(r"\r(?!\n)", text))
        # Match \n not preceded by \r
        analysis.has_lf = bool(re.search(r"(?<!\r)\n", text))

        # Control chars (excluding \n, \r, \t)
        analysis.has_control_chars = bool(
            re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text)
        )

        analysis.has_non_ascii = any(ord(c) > 127 for c in text)
        analysis.has_tabs = "\t" in text

        analysis.has_repeated_whitespace = bool(re.search(r"[ \t]{2,}", text))

        # Basic legal formatting patterns
        # Dates like MM/DD/YYYY or Month DD, YYYY
        analysis.has_dates = bool(
            re.search(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{2,4}\b",
                text,
                re.IGNORECASE,
            )
        )
        # Currencies like $10,000 or £50
        analysis.has_currency = bool(
            re.search(r"[\$£€¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?", text)
        )
        # Section references like "Section 5.2" or "ARTICLE III"
        analysis.has_section_references = bool(
            re.search(
                r"\b(?:section|article|clause)\s+[A-Z0-9\.]+\b", text, re.IGNORECASE
            )
        )

        return analysis

    def normalize_contract(self, record: ContractRecord) -> ProcessedContractRecord:
        """
        Normalizes a contract using the provided policy.
        Ensures that offset integrity is maintained.
        """
        if not record.context or not record.context.strip():
            raise ValueError(f"Document {record.document_id} has empty context.")

        normalized_text = record.context

        if self.policy.replace_control_characters:
            # Replace control characters with spaces to preserve length
            normalized_text = re.sub(
                r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized_text
            )

        # We enforce offset integrity
        self.verify_answer_offsets(record, normalized_text)

        processed = ProcessedContractRecord(
            document_id=record.document_id,
            title=record.title,
            original_context=record.context,
            normalized_context=(
                normalized_text if normalized_text != record.context else None
            ),
            clauses=record.clauses,
        )
        return processed

    def verify_answer_offsets(
        self, original_record: ContractRecord, normalized_text: str
    ) -> None:
        """Verifies that all answer spans in the original record still map correctly in the normalized text."""
        for clause in original_record.clauses:
            for answer in clause.answers:
                # The extracted span from the normalized text
                span_text = normalized_text[answer.start : answer.end]
                if span_text != answer.text:
                    raise ValueError(
                        f"Offset corruption detected in document {original_record.document_id}! "
                        f"Expected '{answer.text}', but got '{span_text}' at [{answer.start}:{answer.end}]."
                    )
