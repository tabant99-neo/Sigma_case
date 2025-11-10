import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from .utils import clean_html_simple, get_model_path, check_model_files

class RussianExamGrader:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = get_model_path()
            
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "CPU"
        
        # Проверяем файлы модели
        existing_files, missing_files = check_model_files(model_path)
        
        if existing_files:
            st.success(f"✅ Найдены файлы модели: {', '.join(existing_files)}")
        
        if missing_files:
            st.warning(f"⚠️ Отсутствуют файлы: {', '.join(missing_files)}")
        
        # Пытаемся загрузить модель если есть основные файлы
        if any(f in existing_files for f in ['model.safetensors', 'pytorch_model.bin']) and 'config.json' in existing_files:
            try:
                self._load_model()
            except Exception as e:
                st.error(f"❌ Ошибка загрузки модели: {e}")
                st.info("💡 Используется демо-режим")
        else:
            st.info("🎯 Используется демо-режим оценки")
    
    def _load_model(self):
        """Загрузка ML модели"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            st.info("🔄 Загружаем модель...")
            
            # Определяем устройство
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                st.success(f"🎯 Используется GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                st.info("💻 Используется CPU")
            
            # Загрузка токенизатора и модели
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            st.success("✅ Модель успешно загружена!")
            
        except ImportError as e:
            st.error(f"❌ Не установлены ML зависимости: {e}")
            st.info("💡 Установите: pip install torch transformers")
            st.info("💡 Используется демо-режим")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.info("💡 Используется демо-режим")
    
    def predict_single_fast(self, question_text, transcription_text, question_number):
        """Оценка одного ответа"""
        if self.model is None:
            return self._demo_predict_single(question_text, transcription_text, question_number)
        else:
            return self._ml_predict_single(question_text, transcription_text, question_number)
    
    def _ml_predict_single(self, question_text, transcription_text, question_number):
        """ML оценка одного ответа"""
        try:
            import torch
            
            # Очистка и подготовка текста
            cleaned_question = clean_html_simple(question_text)
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
            final_score = int(round(np.clip(raw_score, 0, max_score)))
            
            return final_score, raw_score
            
        except Exception as e:
            st.error(f"Ошибка ML оценки: {e}")
            return self._demo_predict_single(question_text, transcription_text, question_number)
    
    def _demo_predict_single(self, question_text, transcription_text, question_number):
        """Демо-оценка одного ответа"""
        time.sleep(0.05)
        
        # Интеллектуальная демо-оценка
        text_length = len(transcription_text)
        word_count = len(transcription_text.split())
        base_score = min(2.0, word_count / 20)
        random_factor = np.random.normal(0, 0.3)
        raw_score = max(0, min(2.0, base_score + random_factor))
        
        max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_number, 2.0)
        final_score = int(round(np.clip(raw_score, 0, max_score)))
        
        return final_score, float(raw_score)
    
    def predict_batch_gpu_optimized(self, df, batch_size=100, max_length=384):
        """Пакетная оценка"""
        if self.model is None:
            return self._demo_predict_batch(df, batch_size)
        else:
            return self._ml_predict_batch(df, batch_size, max_length)
    
    def _ml_predict_batch(self, df, batch_size, max_length):
        """ML пакетная оценка"""
        try:
            import torch
            
            # Предобработка
            df_copy = df.copy()
            mask = ~(df_copy['Текст вопроса'].isna() | df_copy['Транскрибация ответа'].isna())
            df_copy = df_copy[mask].copy()
            
            df_copy['Текст_очищенный'] = df_copy['Текст вопроса'].apply(clean_html_simple)
            df_copy['Input_Text'] = "ЗАДАНИЕ: " + df_copy['Текст_очищенный'] + " | ДИАЛОГ: " + df_copy['Транскрибация ответа']
            
            texts = df_copy['Input_Text'].tolist()
            question_numbers = df_copy['№ вопроса'].values
            
            # Прогресс-бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_predictions = []
            total_samples = len(texts)
            
            # Обработка батчами
            for i in range(0, total_samples, batch_size):
                batch_texts = texts[i:i + batch_size]
                current_batch_size = len(batch_texts)
                
                # Токенизация батча
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Предсказание
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_predictions = outputs.logits.squeeze()
                
                if batch_predictions.ndim == 0:
                    batch_predictions = [float(batch_predictions.cpu().numpy())]
                elif batch_predictions.ndim == 1:
                    batch_predictions = batch_predictions.cpu().numpy().tolist()
                else:
                    batch_predictions = batch_predictions.cpu().numpy().flatten().tolist()
                
                all_predictions.extend(batch_predictions[:current_batch_size])
                
                # Обновление прогресса
                progress = min((i + current_batch_size) / total_samples, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Обработано: {min(i + current_batch_size, total_samples)}/{total_samples}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Постобработка
            result_df = df.iloc[df_copy.index].copy() if len(df_copy) < len(df) else df.copy()
            result_df['predicted_score'] = all_predictions
            
            def finalize_score(row):
                score = row['predicted_score']
                question_num = row['№ вопроса']
                max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_num, 2.0)
                return int(round(np.clip(score, 0, max_score)))
            
            result_df['Оценка экзаменатора_predicted'] = result_df.apply(finalize_score, axis=1)
            
            st.success(f"✅ ML оценка завершена! Обработано {total_samples} ответов")
            return result_df.drop(columns=['predicted_score'], errors='ignore')
            
        except Exception as e:
            st.error(f"Ошибка ML пакетной оценки: {e}")
            return self._demo_predict_batch(df, batch_size)
    
    def _demo_predict_batch(self, df, batch_size):
        """Демо пакетная оценка"""
        try:
            # Демо-обработка
            result_df = df.copy()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = len(result_df)
            
            for i in range(total_rows):
                # Демо-оценка для каждой строки
                transcription = str(result_df.iloc[i]['Транскрибация ответа'])
                word_count = len(transcription.split())
                base_score = min(2.0, word_count / 25)
                random_factor = np.random.normal(0, 0.2)
                raw_score = max(0, min(2.0, base_score + random_factor))
                
                question_num = result_df.iloc[i]['№ вопроса']
                max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_num, 2.0)
                final_score = int(round(np.clip(raw_score, 0, max_score)))
                
                result_df.loc[result_df.index[i], 'Оценка экзаменатора_predicted'] = final_score
                
                # Обновление прогресса
                if i % 10 == 0 or i == total_rows - 1:
                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Обработано: {i+1}/{total_rows} ответов")
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Демо-оценка завершена! Обработано {total_rows} ответов")
            return result_df
            
        except Exception as e:
            st.error(f"Ошибка демо-оценки: {e}")
            return None
