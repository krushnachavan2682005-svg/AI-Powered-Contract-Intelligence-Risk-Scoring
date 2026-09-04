# High-Level Architecture

This document describes the planned future architecture of the AI-Powered Contract Intelligence & Risk Scoring platform.

## Conceptual Pipeline

The system is designed to process legal contracts through a sequence of modular components:

1. **Contract Input**: Reception of legal documents (PDF/DOCX) via API or asynchronous storage events.
2. **Document Ingestion**: Parsing file metadata, tracking progress, and validating formats.
3. **Text Extraction**: Using PyMuPDF and python-docx to extract raw text, and OCR (pytesseract) for scanned components.
4. **Preprocessing**: Normalizing text, removing artifacts, and preparing sentences/paragraphs for NLP.
5. **Contract Structure Analysis**: Identifying sections, headers, and document hierarchy.
6. **NER + Clause Classification**: Identifying entities (organizations, dates, jurisdictions) and classifying legal clauses (termination, confidentiality, liability) using fine-tuned models (e.g., on the CUAD dataset).
7. **Risk Analysis**: Evaluating extracted clauses against predefined policies to generate explainable risk scores.
8. **Embeddings / Vector Search**: Generating semantic embeddings for clauses and storing them in a vector database (e.g., Qdrant or Milvus) to support semantic search.
9. **API**: Exposing all functionality via FastAPI and managing asynchronous processing with Celery and Redis.

> **Note**: The majority of these components are future modules and are not implemented in Module 0.
