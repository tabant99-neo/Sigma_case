import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import re
import streamlit as st
from typing import List, Optional
import time

class OptimizedRussianExamGrader:
    def __init__(self, model_path, batch_size=32, use_fp16=True):
        """
        Оптимизированный класс для оценки экзаменационных ответов.
        
        Args:
            model_path (str): Путь к папке с моделью
            batch_size (int): Размер батча для обработки
            use_fp16 (bool): Использовать ли FP16 для ускорения (только для GPU)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        
        # Оптимизации PyTorch
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Перенос модели на устройство
        self.model.to(self.device)
        
        # Использование FP16 для ускорения на GPU
        if self.use_fp16:
            self.model.half()
            st.info("✅ Используется FP16 для ускорения на GPU")
        
        self.model.eval()
        
        st.success(f"✅ Модель загружена на устройство: {self.device}")
        st.info(f"📊 Размер батча: {batch_size}")

    def preprocess_text(self, text):
        """
        Базовая очистка текста.
        """
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def predict_single(self, text):
        """
        Предсказание оценки для одного текста.
        """
        processed_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Конвертация в fp16 если используется
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits.cpu().numpy()

        grade = float(prediction[0][0])
        grade = max(0, min(5, grade))
        return round(grade, 2)

    def predict_batch(self, texts: List[str]) -> List[float]:
        """
        Пакетное предсказание для списка текстов.
        
        Args:
            texts (List[str]): Список текстов для оценки
            
        Returns:
            List[float]: Список оценок от 0 до 5
        """
        if not texts:
            return []
        
        try:
            # Предобработка текстов
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Токенизация батча
            inputs = self.tokenizer(
                processed_texts,
                max_length=512,
                padding=True,  # Динамический паддинг
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Конвертация в fp16 если используется
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.cpu().numpy()

            # Извлечение оценок и нормализация
            grades = predictions[:, 0].tolist()
            grades = [max(0, min(5, float(grade))) for grade in grades]
            return [round(grade, 2) for grade in grades]
            
        except Exception as e:
            st.error(f"❌ Ошибка при пакетной обработке: {e}")
            # Возвращаем нулевые оценки в случае ошибки
            return [0.0] * len(texts)

    def predict_large_dataset(self, texts: List[str], 
                            progress_callback: Optional[callable] = None) -> List[float]:
        """
        Обработка большого набора текстов батчами.
        
        Args:
            texts (List[str]): Список текстов для оценки
            progress_callback (callable): Функция для отслеживания прогресса
            
        Returns:
            List[float]: Список всех оценок
        """
        if not texts:
            return []
        
        all_grades = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        st.info(f"🔢 Всего батчей для обработки: {total_batches}")
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_grades = self.predict_batch(batch_texts)
            all_grades.extend(batch_grades)
            
            # Вызов callback для обновления прогресса
            if progress_callback:
                progress = (i + len(batch_texts)) / len(texts)
                processed_count = i + len(batch_texts)
                progress_callback(progress, processed_count, len(texts))
        
        return all_grades

    def benchmark_performance(self, sample_texts: List[str], num_runs: int = 3):
        """
        Бенчмарк производительности модели.
        
        Args:
            sample_texts (List[str]): Примеры текстов для тестирования
            num_runs (int): Количество запусков для усреднения
        """
        st.header("📊 Тестирование производительности")
        
        times = []
        for run in range(num_runs):
            start_time = time.time()
            _ = self.predict_large_dataset(sample_texts)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        speed = len(sample_texts) / avg_time
        
        st.metric("Среднее время обработки", f"{avg_time:.2f} сек")
        st.metric("Скорость обработки", f"{speed:.2f} ответов/сек")
        st.metric("Размер батча", self.batch_size)
        st.metric("Устройство", str(self.device))

# Функции для обработки CSV
def process_csv_file(csv_path: str, grader: OptimizedRussianExamGrader, 
                    text_column: str = 'answer') -> pd.DataFrame:
    """
    Обработка CSV файла с ответами.
    
    Args:
        csv_path (str): Путь к CSV файлу
        grader (OptimizedRussianExamGrader): Объект оценщика
        text_column (str): Название столбца с текстами ответов
        
    Returns:
        pd.DataFrame: DataFrame с добавленными оценками
    """
    try:
        # Чтение CSV
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Проверка наличия нужного столбца
        if text_column not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Столбец '{text_column}' не найден. Доступные столбцы: {available_columns}")
        
        # Извлечение ответов
        answers = df[text_column].astype(str).tolist()
        
        # Прогресс-бар для Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        
        start
