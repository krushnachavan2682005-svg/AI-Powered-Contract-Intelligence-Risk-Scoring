from pydantic import BaseModel, Field


class AnswerRecord(BaseModel):
    """Represents a single extracted answer span from a contract."""

    text: str
    start: int
    end: int


class ClauseRecord(BaseModel):
    """Represents a specific clause category annotation for a contract."""

    annotation_id: str
    category: str
    is_present: bool
    answers: list[AnswerRecord] = Field(default_factory=list)


class ContractRecord(BaseModel):
    """Canonical representation of a single parsed contract document."""

    document_id: str
    title: str
    context: str
    clauses: list[ClauseRecord] = Field(default_factory=list)
