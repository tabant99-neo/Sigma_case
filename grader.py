import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from contextlib import contextmanager

from .utils import clean_html

class RussianExamGraderGPU:
    def __init__(self, model_path):
        # Принудительно используем GPU с оптимизацией
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Оптимизации для GPU
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        try:
            # Загрузка с оптимизацией для GPU
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Используем float16 для GPU для ускорения и экономии памяти
            torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                torch_dtype=torch_dtype
            )
            
            # Перенос модели на устройство
            self.model.to(self.device)
            self.model.eval()
            
            # Дополнительные оптимизации
            if self.device.type == 'cuda':
                try:
                    self.model = torch.compile(self.model)  # Компиляция для дополнительного ускорения
                except:
                    pass  # Если компиляция не поддерживается
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке модели: {e}")

    @contextmanager
    def inference_mode(self):
        """Контекст для оптимального inference"""
        original_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            yield
        finally:
            torch.set_grad_enabled(original_grad)

    def predict_batch_gpu_optimized(self, df, batch_size=128, max_length=384):
        """
        GPU-оптимизированная батч-обработка
        """
        try:
            with self.inference_mode():
                # Импорты здесь чтобы избежать циклических импортов
                from .utils import preprocess_data_fast, finalize_score_vectorized
                
                # Быстрая предобработка
                df_processed = preprocess_data_fast(df.copy())
                texts = df_processed['Input_Text'].tolist()
                question_numbers = df_processed['№ вопроса'].values
                
                all_predictions = []
                total_samples = len(texts)
                
                # Обработка батчами
                for i in range(0, total_samples, batch_size):
                    batch_texts = texts[i:i + batch_size]
                    current_batch_size = len(batch_texts)
                    
                    # Векторизованная токенизация батча
                    inputs = self.tokenizer(
                        batch_texts,
                        max_length=max_length,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device, non_blocking=True)
                    
                    # Пакетное предсказание на GPU
                    with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                        outputs = self.model(**inputs)
                        batch_predictions = outputs.logits.squeeze()
                    
                    # Обрабатываем разные случаи dimensionalities
                    if batch_predictions.ndim == 0:  # Один элемент
                        batch_predictions = [float(batch_predictions.cpu().numpy())]
                    elif batch_predictions.ndim == 1:  # Один батч
                        batch_predictions = batch_predictions.cpu().numpy().tolist()
                    else:  # Несколько измерений
                        batch_predictions = batch_predictions.cpu().numpy().flatten().tolist()
                    
                    all_predictions.extend(batch_predictions[:current_batch_size])
                
                # Векторизованная постобработка
                final_predictions = finalize_score_vectorized(
                    np.array(all_predictions), 
                    question_numbers
                )
                
                # Создаем результат
                df_result = df.iloc[df_processed.index].copy() if len(df_processed) < len(df) else df.copy()
                df_result['predicted_score'] = all_predictions
                df_result['Оценка экзаменатора_predicted'] = final_predictions
                
                return df_result.drop(columns=['predicted_score'], errors='ignore')
                
        except Exception as e:
            raise Exception(f"Ошибка при GPU-обработке: {e}")

    def predict_single_fast(self, question_text, transcription_text, question_number):
        """Быстрая оценка одного ответа"""
        try:
            with self.inference_mode():
                # Быстрая предобработка
                cleaned_question = clean_html(question_text)
                input_text = f"ЗАДАНИЕ: {cleaned_question} | ДИАЛОГ: {transcription_text}"
                
                # Токенизация
                inputs = self.tokenizer(
                    input_text,
                    max_length=384,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device, non_blocking=True)
                
                # Предсказание
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    outputs = self.model(**inputs)
                    raw_score = float(outputs.logits.cpu().numpy()[0][0])
                
                # Постобработка
                max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_number, 2.0)
                clipped_score = np.clip(raw_score, 0.0, max_score)
                final_score = int(round(clipped_score))
                
                return final_score, raw_score
                
        except Exception as e:
            raise Exception(f"Ошибка при предсказании: {e}")
