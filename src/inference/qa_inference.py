import torch
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
