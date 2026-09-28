from pydantic import BaseModel
from typing import List, Optional

class Extraction(BaseModel):
    category: str
    question: str
    answer: str
    confidence: float
    source_text: str

class RiskIndicator(BaseModel):
    category: str
    severity: str
    reason: str
    evidence: str
    requires_manual_review: bool

class AnalysisResponse(BaseModel):
    filename: str
    status: str
    document_statistics: dict
    extractions: List[Extraction]
    risk_indicators: List[RiskIndicator]
    processing_metadata: dict
