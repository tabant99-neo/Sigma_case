import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import re
import sys

# Конфигурация страницы
st.set_page_config(
    page_title="Russian Exam Grader",
    page_icon="🇷🇺",
    layout="wide"
)

# Заголовок приложения
st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
st.markdown("""
**⚡ Версия с ML моделью из Git LFS**  
Загрузите CSV-файл с транскрибациями ответов для оценки.
""")

# Вспомогательные функции
def safe_read_csv(uploaded_file):
    """Безопасное чтение CSV"""
    for encoding in ['utf-8', 'cp1251', 'windows-1251']:
        for sep in [';', ',', '\t']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                if len(df.columns) > 1:
                    return df
            except:
                continue
    
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, on_bad_lines='skip')
    except:
        raise ValueError("Не удалось прочитать файл")

def clean_html_simple(html_text):
    """Простая очистка HTML"""
    if pd.isna(html_text): 
        return ""
    text = re.sub(r'<[^>]+>', '', str(html_text))
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def check_model_files(model_path):
    """Проверяем наличие всех файлов модели"""
    required_files = [
        'pytorch_model.bin',
        'config.json', 
        'tokenizer_config.json',
        'vocab.txt'
    ]
    
    existing_files = []
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    return existing_files, missing_files

# Класс для работы с моделью
class RussianExamGrader:
    def __init__(self, model_path="my_trained_model_2"):
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
        
        # Пытаемся загрузить модель если есть все основные файлы
        if 'pytorch_model.bin' in existing_files and 'config.json' in existing_files:
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
            st.info("💡 Используется демо-режим")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.info("💡 Используется демо-режим")
    
    def predict_single_fast(self, question_text, transcription_text, question_number):
        """Оценка одного ответа"""
        if self.model is None:
            # Демо-режим
            return self._demo_predict_single(question_text, transcription_text, question_number)
        else:
            # Режим с ML моделью
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

# Инициализация градера
@st.cache_resource
def load_grader():
    return RussianExamGrader()

# Загружаем градер
grader = load_grader()

# Дальше тот же интерфейс что и в предыдущем коде...
# [ОСТАВЬТЕ ВЕСЬ ИНТЕРФЕЙС ИЗ ПРЕДЫДУЩЕГО КОДА БЕЗ ИЗМЕНЕНИЙ]

# Создаем вкладки
tab1, tab2 = st.tabs(["🎯 Оценить один ответ", "📊 Оценить файл CSV"])

with tab1:
    st.header("Оценка одного ответа")
    
    col1, col2 = st.columns(2)
    with col1:
        question_number = st.selectbox("№ вопроса:", [1, 2, 3, 4], key="question_number")
    with col2:
        max_score = {1: 1, 2: 2, 3: 1, 4: 2}.get(question_number, 2)
        st.info(f"Максимальный балл: {max_score}")
    
    question_text = st.text_area(
        "Текст вопроса:",
        height=100,
        placeholder="Введите текст вопроса...",
        key="question_text"
    )
    
    transcription_text = st.text_area(
        "Транскрибация ответа:",
        height=150,
        placeholder="Введите транскрибацию ответа...",
        key="transcription_text"
    )

    if st.button("⚡ Оценить ответ", type="primary", key="single"):
        if question_text.strip() and transcription_text.strip():
            with st.spinner("🤖 Анализируем ответ..."):
                start_time = time.time()
                try:
                    final_score, raw_score = grader.predict_single_fast(question_text, transcription_text, question_number)
                    processing_time = time.time() - start_time
                    
                    mode = "ML модель" if grader.model is not None else "Демо-режим"
                    st.success(f"**Предсказанная оценка: {final_score} / {max_score}** ({mode})")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Оценка", f"{final_score}/{max_score}")
                    with col2:
                        st.progress(final_score / max_score)
                    
                    with st.expander("🔍 Детали анализа"):
                        st.write(f"**Сырая оценка:** {raw_score:.4f}")
                        st.write(f"**Режим:** {mode}")
                        st.write(f"**Время обработки:** {processing_time:.3f} сек")
                        if grader.model is None:
                            st.info("💡 Для использования ML модели установите зависимости: torch, transformers")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при оценке: {e}")
        else:
            st.warning("⚠️ Пожалуйста, заполните все поля.")

with tab2:
    st.header("Пакетная оценка из CSV-файла")
    
    st.markdown("""
    **Особенности версии:**
    - 🚀 Автоматическое определение режима (ML/Демо)
    - 🎯 Интеллектуальная обработка данных
    - 📊 Детальная статистика результатов
    - ⚡ Поддержка больших файлов
    """)
    
    # Настройки обработки
    st.subheader("⚙️ Настройки обработки")
    
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.slider(
            "Размер батча:",
            min_value=50,
            max_value=200,
            value=100,
            help="Количество ответов в пакете"
        )
    with col2:
        mode = "ML модель" if grader.model is not None else "Демо-режим"
        st.metric("Режим", mode)
        if grader.model is None:
            st.info("💡 Демо-режим: интеллектуальная оценка")
    
    uploaded_file = st.file_uploader(
        "Выберите CSV-файл с данными экзамена", 
        type=['csv', 'txt'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Загрузка файла
            with st.spinner("📥 Загружаем файл..."):
                df = safe_read_csv(uploaded_file)
            
            st.success(f"✅ Файл загружен: {len(df)} строк, {len(df.columns)} колонок")
            
            # Проверка обязательных колонок
            required_columns = ['№ вопроса', 'Текст вопроса', 'Транскрибация ответа']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Отсутствуют обязательные колонки: {', '.join(missing_columns)}")
                st.info(f"📋 Найдены колонки: {', '.join(df.columns)}")
            else:
                # Показываем информацию о данных
                st.subheader("📊 Информация о данных")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Распределение по вопросам:**")
                    question_stats = df['№ вопроса'].value_counts().sort_index()
                    for q_num, count in question_stats.items():
                        max_score = {1: 1, 2: 2, 3: 1, 4: 2}.get(q_num, 2)
                        st.write(f"- Вопрос {q_num}: {count} ответов (макс. {max_score} баллов)")
                
                with col2:
                    st.write("**Пример данных:**")
                    display_cols = ['№ вопроса', 'Текст вопроса', 'Транскрибация ответа']
                    if 'Оценка экзаменатора' in df.columns:
                        display_cols.append('Оценка экзаменатора')
                    st.dataframe(df[display_cols].head(3))
                
                if st.button("🚀 Запустить оценку", type="primary", key="batch"):
                    with st.spinner("⚡ Обрабатываем данные..."):
                        try:
                            result_df = grader.predict_batch_gpu_optimized(df, batch_size=batch_size)
                            
                            if result_df is not None:
                                st.balloons()
                                st.subheader("📈 Результаты оценки")
                                
                                # Показываем результаты
                                display_columns = ['№ вопроса', 'Текст вопроса', 'Транскрибация ответа', 'Оценка экзаменатора_predicted']
                                if 'Оценка экзаменатора' in result_df.columns:
                                    display_columns.insert(3, 'Оценка экзаменатора')
                                
                                st.dataframe(result_df[display_columns].head(10))
                                
                                # Статистика
                                st.subheader("📊 Статистика оценок")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    avg_grade = result_df['Оценка экзаменатора_predicted'].mean()
                                    st.metric("Средняя оценка", f"{avg_grade:.2f}")
                                with col2:
                                    min_grade = result_df['Оценка экзаменатора_predicted'].min()
                                    st.metric("Мин. оценка", f"{min_grade}")
                                with col3:
                                    max_grade = result_df['Оценка экзаменатора_predicted'].max()
                                    st.metric("Макс. оценка", f"{max_grade}")
                                with col4:
                                    total_count = len(result_df)
                                    st.metric("Всего ответов", total_count)
                                
                                # Распределение
                                st.subheader("📊 Распределение оценок")
                                grade_counts = result_df['Оценка экзаменатора_predicted'].value_counts().sort_index()
                                st.bar_chart(grade_counts)
                                
                                # Скачивание
                                st.subheader("💾 Скачать результаты")
                                csv_data = result_df.to_csv(index=False, sep=';').encode('utf-8')
                                st.download_button(
                                    label="📥 Скачать результаты (CSV)",
                                    data=csv_data,
                                    file_name="graded_results.csv",
                                    mime="text/csv",
                                    key="download_full"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Ошибка при обработке: {e}")
                            
        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {e}")

# Боковая панель
with st.sidebar:
    st.header("⚡ О системе")
    
    # Информация о модели
    st.subheader("📦 Модель")
    existing_files, missing_files = check_model_files("my_trained_model_2")
    
    if existing_files:
        st.success(f"Файлы: {len(existing_files)}/{len(existing_files) + len(missing_files)}")
        for file in existing_files:
            st.write(f"✅ {file}")
    
    if missing_files:
        st.warning("Отсутствуют:")
        for file in missing_files:
            st.write(f"❌ {file}")
    
    st.subheader("🎯 Режим")
    if grader.model is not None:
        st.success("ML модель активна")
        st.info(f"Устройство: {grader.device}")
    else:
        st.info("Демо-режим")
        st.write("Для ML модели установите:")
        st.code("pip install torch transformers")

# Футер
st.markdown("---")
st.markdown(
    "**Система оценки экзаменационных ответов** • "
    "Поддержка Git LFS • "
    "Автоматический режим ML/Демо"
)
