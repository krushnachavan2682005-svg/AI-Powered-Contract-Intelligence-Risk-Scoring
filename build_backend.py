import os

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# 1. src/inference/model_service.py
model_service = """import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def load_model(self, model_path: str):
        if self.model is None:
            print(f"Loading model from {model_path} onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")

    def get_model(self):
        return self.model, self.tokenizer, self.device
"""

# 2. src/inference/qa_inference.py
qa_inference = """import torch
from src.inference.model_service import ModelService

def run_inference(question, context):
    model, tokenizer, device = ModelService().get_model()
    
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()
    
    if start_idx >= end_idx or start_idx == 0:
        return None, 0.0
        
    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx+1], skip_special_tokens=True)
    confidence = (start_logits[0][start_idx].item() + end_logits[0][end_idx].item()) / 2.0
    return answer, confidence
"""

# 3. src/inference/risk_analysis.py
risk_analysis = """def analyze_risk(extractions):
    indicators = []
    for ext in extractions:
        cat = ext["category"]
        if cat in ["Uncapped Liability", "Liquidated Damages", "Termination For Convenience"]:
            indicators.append({
                "category": cat,
                "severity": "high",
                "reason": f"A {cat} clause was detected.",
                "evidence": ext["answer"],
                "requires_manual_review": True
            })
        elif cat in ["Audit Rights", "Anti-Assignment"]:
            indicators.append({
                "category": cat,
                "severity": "medium",
                "reason": f"A {cat} clause requires review.",
                "evidence": ext["answer"],
                "requires_manual_review": True
            })
    return indicators
"""

# 4. src/api/schemas.py
api_schemas = """from pydantic import BaseModel
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
"""

# 5. src/api/main.py
api_main = """import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import pdfplumber

from src.inference.model_service import ModelService
from src.inference.qa_inference import run_inference
from src.inference.risk_analysis import analyze_risk
from src.api.schemas import AnalysisResponse, Extraction

app = FastAPI(title="AI-Powered Contract Intelligence & Risk Scoring")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_DIR", "artifacts/checkpoints/baseline")

@app.on_event("startup")
def load_model_on_startup():
    ModelService().load_model(MODEL_PATH)

@app.get("/health")
def health_check():
    model, _, _ = ModelService().get_model()
    return {
        "status": "ok",
        "model": "distilbert-base-uncased",
        "model_loaded": model is not None
    }

def extract_text(file_path, filename):
    text = ""
    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\\n"
    elif filename.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format")
    return text.strip()

categories = {
    "Uncapped Liability": "Highlight the parts (if any) of this contract related to uncapped liability.",
    "Liquidated Damages": "Highlight the parts (if any) of this contract related to liquidated damages.",
    "Audit Rights": "Highlight the parts (if any) of this contract related to audit rights."
}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
        
    try:
        text = extract_text(tmp_path, file.filename)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Error extracting text: {str(e)}")
        
    os.remove(tmp_path)
    
    if not text:
        raise HTTPException(status_code=400, detail="Empty document or unable to extract text.")

    extractions = []
    # simplify chunking: take first 2000 chars for inference to avoid timeout for now
    context = text[:2000]
    
    for cat, question in categories.items():
        ans, conf = run_inference(question, context)
        if ans:
            extractions.append(Extraction(
                category=cat,
                question=question,
                answer=ans,
                confidence=conf,
                source_text=context
            ))
            
    risk_indicators = analyze_risk([ext.dict() for ext in extractions])
    
    processing_time = time.time() - start_time
    
    return AnalysisResponse(
        filename=file.filename,
        status="success",
        document_statistics={
            "extracted_text_length": len(text)
        },
        extractions=extractions,
        risk_indicators=risk_indicators,
        processing_metadata={
            "processing_time_seconds": round(processing_time, 2)
        }
    )
"""

if __name__ == "__main__":
    write_file("src/inference/model_service.py", model_service)
    write_file("src/inference/qa_inference.py", qa_inference)
    write_file("src/inference/risk_analysis.py", risk_analysis)
    write_file("src/api/schemas.py", api_schemas)
    write_file("src/api/main.py", api_main)
    print("Backend code generated successfully.")
