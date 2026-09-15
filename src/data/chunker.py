from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.data.legal_preprocessor import ProcessedContractRecord


class ChunkAnswerRecord(BaseModel):
    annotation_id: str
    category: str
    answer_text: str
    original_start: int
    original_end: int
    local_start: int
    local_end: int


class ChunkRecord(BaseModel):
    document_id: str
    chunk_id: str
    chunk_start: int
    chunk_end: int
    chunk_text: str
    answers: List[ChunkAnswerRecord] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    chunk_size: int = 4000
    overlap: int = 500


class ChunkingSummary(BaseModel):
    config: Dict[str, Any]
    total_contracts: int = 0
    total_chunks: int = 0
    total_answer_spans: int = 0
    covered_answer_spans: int = 0
    uncovered_answer_spans: int = 0
    coverage_percentage: float = 0.0
    average_chunk_length: float = 0.0
    max_chunk_length: int = 0
    min_chunk_length: int = 0
    average_chunks_per_contract: float = 0.0
    multiple_coverage_answers: int = 0


class AnswerAwareChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        if self.config.overlap >= self.config.chunk_size:
            raise ValueError(
                f"Overlap ({self.config.overlap}) must be less than chunk_size ({self.config.chunk_size})."
            )

    def chunk_contract(self, record: ProcessedContractRecord) -> List[ChunkRecord]:
        text = (
            record.normalized_context
            if record.normalized_context is not None
            else record.original_context
        )

        if not text:
            raise ValueError(f"Document {record.document_id} has empty context.")

        doc_len = len(text)
        chunks: List[ChunkRecord] = []

        flat_answers: List[Dict[str, Any]] = []
        for clause in record.clauses:
            if clause.is_present and clause.answers:
                for answer in clause.answers:
                    flat_answers.append(
                        {
                            "annotation_id": clause.annotation_id,
                            "category": clause.category,
                            "answer_text": answer.text,
                            "original_start": answer.start,
                            "original_end": answer.end,
                        }
                    )

        start = 0
        chunk_idx = 0
        step = self.config.chunk_size - self.config.overlap

        while start < doc_len:
            end = min(start + self.config.chunk_size, doc_len)

            # Dynamic expansion to avoid truncating answers that start in this chunk
            max_answer_end = end
            for ans in flat_answers:
                ans_start = int(ans["original_start"])
                ans_end = int(ans["original_end"])
                # Check if it starts in the current chunk and ends after `end`
                if ans_start >= start and ans_start < end:
                    max_answer_end = max(max_answer_end, ans_end)

            if max_answer_end > end:
                end = max_answer_end

            chunk_text = text[start:end]
            chunk_answers = []

            for ans in flat_answers:
                ans_start = int(ans["original_start"])
                ans_end = int(ans["original_end"])

                if ans_start >= start and ans_end <= end:
                    local_start = ans_start - start
                    local_end = ans_end - start

                    if chunk_text[local_start:local_end] != ans["answer_text"]:
                        raise ValueError(
                            f"Offset corruption during chunking for {ans['annotation_id']}"
                        )

                    chunk_answers.append(
                        ChunkAnswerRecord(
                            annotation_id=str(ans["annotation_id"]),
                            category=str(ans["category"]),
                            answer_text=str(ans["answer_text"]),
                            original_start=ans_start,
                            original_end=ans_end,
                            local_start=local_start,
                            local_end=local_end,
                        )
                    )

            chunks.append(
                ChunkRecord(
                    document_id=record.document_id,
                    chunk_id=f"{record.document_id}::chunk_{chunk_idx:04d}",
                    chunk_start=start,
                    chunk_end=end,
                    chunk_text=chunk_text,
                    answers=chunk_answers,
                )
            )

            if end >= doc_len:
                break

            start += step
            chunk_idx += 1

        return chunks
