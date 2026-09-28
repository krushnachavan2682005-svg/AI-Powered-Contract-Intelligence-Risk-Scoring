# AI-Powered Contract Intelligence & Risk Scoring

## 1. Project Overview
An NLP platform for ingesting legal contracts, extracting entities and clauses, detecting risky language, and enabling semantic search. This portfolio project demonstrates an end-to-end machine learning pipeline from data preparation to a production-ready application.

## 2. Problem Statement
Reviewing legal contracts is time-consuming and error-prone. This project aims to automate the extraction of key legal clauses and highlight potential risk indicators to assist legal professionals, improving efficiency and reducing the likelihood of missed critical terms.

## 3. Architecture
- **Backend:** FastAPI for asynchronous API endpoints.
- **Frontend:** Vanilla HTML/JS with responsive CSS.
- **Model:** Fine-tuned `distilbert-base-uncased` for extractive QA.
- **Inference:** PyTorch with single-instance model loading to optimize memory.

## 4. Dataset
The model was fine-tuned on the CUAD (Contract Understanding Atticus Dataset), a corpus of legal contracts annotated by legal experts.

## 5. CUAD Statistics
- 510 commercial legal contracts.
- 41 distinct clause categories.
- High complexity, long-document context.

## 6. Preprocessing
- Contracts were normalized, handling duplicate/invalid data.
- Special legal characters and spacing were preserved safely.

## 7. Chunking
- Long contracts were chunked using an overlapping sliding window strategy to ensure answer spans were fully captured across boundaries.

## 8. Model
- **Base Architecture:** `distilbert-base-uncased`
- **Task:** Question Answering (`DistilBertForQuestionAnswering`)

## 9. Training Configuration
- **Hardware:** Tesla T4 GPU
- **Epochs:** 3
- **Batch Size:** 16 (gradient accumulation 2 -> effective 32)
- **Precision:** FP16
- **Learning Rate:** 3e-5

## 10. Training Results
- **Epoch 1:** Train Loss = 1.5116, Val Loss = 0.9915
- **Epoch 2:** Train Loss = 0.8761, Val Loss = 0.9032
- **Epoch 3:** Train Loss = 0.7223, Val Loss = 0.8481
- **Best Model:** Epoch 3

## 11. Test Metrics
- **Test Windows:** 4,602
- **Unique Answers:** 1,043
- **Exact Match (EM):** 46.31%
- **Token F1:** 58.84%

## 12. Inference Pipeline
Upload -> Text Extraction (PDF/TXT) -> Chunking (simplified context) -> Tokenization -> Model Inference -> Answer Aggregation -> Risk Analysis -> JSON Output

## 13. Risk Analysis Design
The risk-analysis layer is a transparent heuristic layer over extracted clauses and is **not a substitute for legal advice**. It uses deterministic rules on extracted CUAD categories (e.g., Uncapped Liability implies High Risk).

## 14. API Endpoints
- `GET /health` - Check API status and model load.
- `POST /analyze` - Upload file and get extraction + risk JSON.

## 15. Frontend
A clean HTML/JS interface deployed at `/` via FastAPI's `StaticFiles`. 

## 16. Installation
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 17. Running Backend
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## 18. Running Frontend
The frontend is integrated into the backend and runs automatically at `http://127.0.0.1:8000/`.

## 19. Example Workflow
1. Run the server.
2. Visit `http://127.0.0.1:8000/`.
3. Upload a contract (e.g., `sample_contract.txt`).
4. Click "Analyze" and view risk indicators.

## 20. Limitations
- Inference on very long documents may take significant time or require GPU.
- Risk scores are heuristic-based and deterministically derived from clause presence.

## 21. Responsible/Legal Disclaimer
The model is an extractive QA model trained on CUAD. The risk-analysis layer is a transparent heuristic layer over extracted clauses and is not a substitute for legal advice.

## 22. Project Structure
- `src/api` - FastAPI endpoints
- `src/inference` - Model loading, QA inference, and risk logic
- `static` - Frontend HTML
- `artifacts` - Trained model artifacts
- `data` - Interim/Processed datasets
