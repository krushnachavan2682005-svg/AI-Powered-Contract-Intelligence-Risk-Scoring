import os
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
