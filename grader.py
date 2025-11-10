import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class RussianExamGrader:
    def __init__(self, model_path="my_trained_model_2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(self, question_text, transcription_text, question_number):
        """Предсказание для одного ответа"""
        try:
            from .utils import clean_html
            
            # Очистка и подготовка текста
            cleaned_question = clean_html(question_text)
            input_text = f"ЗАДАНИЕ: {cleaned_question} | ДИАЛОГ: {transcription_text}"
            
            # Токенизация
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                raw_score = float(outputs.logits.cpu().numpy()[0][0])
            
            # Постобработка
            max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_number, 2.0)
            clipped_score = np.clip(raw_score, 0.0, max_score)
            final_score = int(round(clipped_score))
            
            return final_score, raw_score
            
        except Exception as e:
            raise Exception(f"Prediction error: {e}")
