import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import pdfplumber

from src.inference.model_service import ModelService
from src.inference.qa_inference import run_inference
from src.inference.risk_analysis import analyze_risk
from src.api.schemas import AnalysisResponse, Extraction
from fastapi.staticfiles import StaticFiles

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
                    text += page_text + "\n"
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

app.mount('/', StaticFiles(directory='static', html=True), name='static')
