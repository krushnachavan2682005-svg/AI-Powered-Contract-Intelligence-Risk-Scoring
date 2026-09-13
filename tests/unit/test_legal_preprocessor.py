import pytest

from src.data.legal_preprocessor import LegalPreprocessor, NormalizationPolicy
from src.data.schemas import AnswerRecord, ClauseRecord, ContractRecord


def test_lowercasing_is_not_applied():
    policy = NormalizationPolicy(preserve_case=True)
    preprocessor = LegalPreprocessor(policy)

    record = ContractRecord(
        document_id="doc_1",
        title="Test Doc",
        context="The Company shall NOT terminate the agreement.",
        clauses=[],
    )

    processed = preprocessor.normalize_contract(record)
    assert (
        processed.original_context == "The Company shall NOT terminate the agreement."
    )
    assert processed.normalized_context is None


def test_punctuation_and_numbers_preserved():
    preprocessor = LegalPreprocessor()

    record = ContractRecord(
        document_id="doc_2",
        title="Test Doc 2",
        context="Section 5.2: The fine is $10,000.",
        clauses=[],
    )

    processed = preprocessor.normalize_contract(record)
    assert processed.original_context == "Section 5.2: The fine is $10,000."
    assert processed.normalized_context is None


def test_answer_offsets_remain_valid():
    preprocessor = LegalPreprocessor()

    context = "The party shall not terminate the contract."
    # "shall not terminate" -> start: 10, end: 29
    answer = AnswerRecord(text="shall not terminate", start=10, end=29)
    clause = ClauseRecord(
        annotation_id="c_1", category="Termination", is_present=True, answers=[answer]
    )

    record = ContractRecord(
        document_id="doc_3", title="Test Doc 3", context=context, clauses=[clause]
    )

    processed = preprocessor.normalize_contract(record)
    # The verifier in normalize_contract will throw an error if it fails,
    # so reaching here means it passed.
    assert processed.original_context[answer.start : answer.end] == answer.text


def test_offset_corruption_detected():
    # If we somehow made a bad policy
    policy = NormalizationPolicy(replace_control_characters=True)
    preprocessor = LegalPreprocessor(policy)

    # Let's say we have a contract that gets normalized (e.g. control char stripped instead of replaced)
    # But our code replaces it with space, which preserves length.
    # We can test the verifier directly to simulate corruption.
    context = "The party shall terminate."
    answer = AnswerRecord(text="shall terminate", start=10, end=25)
    clause = ClauseRecord(
        annotation_id="c_1", category="Termination", is_present=True, answers=[answer]
    )
    record = ContractRecord(
        document_id="doc_4", title="Doc 4", context=context, clauses=[clause]
    )

    bad_normalized_text = "The party shall   terminate."  # Offset changed!

    with pytest.raises(ValueError, match="Offset corruption detected"):
        preprocessor.verify_answer_offsets(record, bad_normalized_text)


def test_empty_context_fails():
    preprocessor = LegalPreprocessor()
    record = ContractRecord(document_id="doc_5", title="Doc", context="   ", clauses=[])

    with pytest.raises(ValueError, match="empty context"):
        preprocessor.normalize_contract(record)


def test_text_analysis():
    preprocessor = LegalPreprocessor()
    analysis = preprocessor.analyze_text(
        "Hello\r\nWorld\t!\x01  Date: Jan 01, 2020 $500"
    )

    assert analysis.has_crlf is True
    assert analysis.has_tabs is True
    assert analysis.has_control_chars is True
    assert analysis.has_repeated_whitespace is True
    assert analysis.has_dates is True
    assert analysis.has_currency is True
