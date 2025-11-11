import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re
import numpy as np
from typing import List
import time

# Конфигурация страницы
st.set_page_config(
    page_title="Автоматическая оценка экзамена по русскому языку",
    page_icon="🇷🇺",
    layout="wide"
)

# Инициализация состояния сессии
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = {
        'is_processing': False,
        'total_rows': 0,
        'results': None,
        'selected_column': None
    }

if 'graded_results' not in st.session_state:
    st.session_state.graded_results = None

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'grader_instance' not in st.session_state:
    st.session_state.grader_instance = None

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Заголовок приложения
st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
st.markdown("""
Это демо-версия системы оценки письменных ответов на русском языке.
Загрузите CSV-файл с ответами студентов и выберите диапазон строк для оценки.
""")

# Упрощенная модель как fallback
class SimpleRussianGrader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess_text(self, text):
        text = str(text).strip()
        if not text:
            return ""
        return text

    def predict(self, text):
        try:
            if not text or len(str(text).strip()) == 0:
                return 0.0
                
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            
            if len(words) == 0:
                return 0.0
            
            # Простая эвристика на основе длины текста
            length_score = min(len(words) / 50, 1.0)
            final_score = length_score * 5
            
            return round(final_score, 2)
            
        except Exception:
            return 0.0

    def predict_batch(self, texts: List[str]) -> List[float]:
        return [self.predict(text) for text in texts]

# Основной класс для оценки
class RussianExamGrader:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and os.path.exists(model_path):
            try:
                st.info(f"🔄 Загружаем модель из: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
                st.success("✅ Обученная модель успешно загружена!")
                self.use_simple = False
            except Exception as e:
                st.error(f"❌ Ошибка загрузки модели: {e}")
                st.warning("🔄 Используем упрощенную модель")
                self.use_simple = True
                self.simple_grader = SimpleRussianGrader()
        else:
            st.warning("🔄 Используем упрощенную модель")
            self.use_simple = True
            self.simple_grader = SimpleRussianGrader()

    def preprocess_text(self, text):
        text = str(text).strip()
        if not text:
            return ""
        return text

    def predict(self, text):
        if self.use_simple:
            return self.simple_grader.predict(text)
            
        try:
            if not text or len(str(text).strip()) == 0:
                return 0.0
                
            processed_text = self.preprocess_text(text)
            
            inputs = self.tokenizer(
                processed_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = outputs.logits.cpu().numpy()

            grade = float(prediction[0][0])
            grade = max(0, min(5, grade))
            return round(grade, 2)
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            return self.simple_grader.predict(text)

    def predict_batch(self, texts: List[str]) -> List[float]:
        if self.use_simple:
            return self.simple_grader.predict_batch(texts)
            
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            inputs = self.tokenizer(
                processed_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.cpu().numpy()

            grades = predictions[:, 0].tolist()
            grades = [max(0, min(5, float(grade))) for grade in grades]
            return [round(grade, 2) for grade in grades]
            
        except Exception as e:
            st.error(f"Ошибка при пакетном предсказании: {e}")
            return self.simple_grader.predict_batch(texts)

# Функция для чтения CSV
def safe_read_csv(uploaded_file):
    """Чтение CSV с приоритетом для UTF-8 и разделителя ';'"""
    try:
        # Пробуем UTF-8 с разделителем ';'
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=';', on_bad_lines='skip')
        if len(df.columns) > 0 and len(df) > 0:
            st.success(f"✅ Файл прочитан: {len(df)} строк, {len(df.columns)} столбцов")
            return df
    except Exception as e:
        st.warning(f"Не удалось прочитать с UTF-8 и ';': {e}")
    
    # Другие варианты
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', on_bad_lines='skip')
        if len(df.columns) > 0:
            st.info("Файл прочитан с разделителем ','")
            return df
    except:
        pass
        
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp1251', sep=';', on_bad_lines='skip')
        if len(df.columns) > 0:
            st.info("Файл прочитан с кодировкой cp1251")
            return df
    except:
        pass
    
    raise ValueError("Не удалось прочитать файл")

# Функция для обработки данных с выбором диапазона
def process_dataset_range(df, grader, selected_column, start_row, end_row, chunk_size=500):
    """Обработка выбранного диапазона строк с прогрессом"""
    try:
        if selected_column not in df.columns:
            st.error(f"Столбец '{selected_column}' не найден")
            return None
        
        # Выбираем нужный диапазон строк
        selected_df = df.iloc[start_row:end_row].copy()
        total_rows = len(selected_df)
        
        st.info(f"📊 Обрабатывается диапазон: строки {start_row}-{end_row} ({total_rows} строк)")
        
        all_grades = []
        start_time = time.time()
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for start_idx in range(0, total_rows, chunk_size):
            if not st.session_state.processing_state['is_processing']:
                st.warning("Обработка остановлена")
                break
                
            chunk_end_idx = min(start_idx + chunk_size, total_rows)
            chunk = selected_df.iloc[start_idx:chunk_end_idx]
            
            # Обновляем прогресс
            progress = chunk_end_idx / total_rows
            progress_bar.progress(progress)
            
            # Обновляем статус
            elapsed = time.time() - start_time
            rows_per_sec = chunk_end_idx / elapsed if elapsed > 0 else 0
            status_text.text(f"Обработано: {chunk_end_idx}/{total_rows} строк ({rows_per_sec:.1f} строк/сек)")
            
            # Обрабатываем чанк
            answers = chunk[selected_column].astype(str).tolist()
            chunk_grades = grader.predict_batch(answers)
            all_grades.extend(chunk_grades)
        
        progress_bar.empty()
        status_text.empty()
        
        if len(all_grades) == total_rows:
            selected_df['predicted_grade'] = all_grades
            
            total_time = time.time() - start_time
            st.success(f"✅ Обработка завершена! Обработано {len(all_grades)} ответов за {total_time:.1f} сек")
            
            return selected_df
        else:
            st.error(f"❌ Обработано только {len(all_grades)} из {total_rows} строк")
            return None
            
    except Exception as e:
        st.error(f"❌ Ошибка при обработке: {e}")
        return None

# Загрузка модели
def load_grader():
    model_path = "my_trained_model_2"
    
    # Проверяем пути
    possible_paths = [
        model_path,
        f"./{model_path}",
        f"../{model_path}",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return RussianExamGrader(path)
    
    # Если модель не найдена, используем упрощенную
    st.warning("⚠️ Модель не найдена, используем упрощенную версию")
    return RussianExamGrader()

# Основной интерфейс
tab1, tab2, tab3 = st.tabs(["🎯 Оценить один ответ", "📊 Оценить файл CSV", "📈 Результаты"])

with tab1:
    st.header("Оценка одного ответа")
    
    if not st.session_state.model_loaded:
        if st.button("🔄 Загрузить модель"):
            with st.spinner("Загружаем модель..."):
                grader = load_grader()
                st.session_state.grader_instance = grader
                st.session_state.model_loaded = True
                st.success("✅ Модель готова!")
                st.rerun()
    else:
        user_input = st.text_area(
            "Введите ответ студента на русском языке:",
            height=150,
            placeholder="Напишите здесь ответ на экзаменационный вопрос...",
            key="single_answer"
        )

        if st.button("Оценить ответ", type="primary"):
            if user_input.strip():
                with st.spinner("🤖 Модель оценивает ответ..."):
                    grade = st.session_state.grader_instance.predict(user_input)
                
                st.success(f"**Предсказанная оценка: {grade} / 5**")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Оценка", f"{grade}/5")
                with col2:
                    st.progress(grade / 5.0)
                
                if grade >= 4.0:
                    st.info("🎉 Отличный ответ!")
                elif grade >= 3.0:
                    st.info("👍 Хороший ответ")
                elif grade >= 2.0:
                    st.warning("⚠️ Удовлетворительный ответ")
                else:
                    st.error("❌ Ответ требует улучшений")
            else:
                st.warning("⚠️ Пожалуйста, введите текст для оценки.")

with tab2:
    st.header("Пакетная оценка из CSV-файла")
    
    if st.session_state.processing_state['is_processing']:
        st.warning("🔄 Идет обработка файла...")
        if st.button("⏹️ Остановить обработку"):
            st.session_state.processing_state['is_processing'] = False
            st.rerun()
    
    st.markdown("""
    Загрузите CSV-файл с ответами студентов. **Рекомендуемый формат:** UTF-8 с разделителем ';'
    """)
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Сначала загрузите модель на вкладке 'Оценить один ответ'")
    else:
        uploaded_file = st.file_uploader("Выберите CSV-файл", type=['csv'])
        
        if uploaded_file is not None:
            if st.session_state.uploaded_data is None:
                with st.spinner("📖 Читаем файл..."):
                    df = safe_read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
            else:
                df = st.session_state.uploaded_data
            
            st.subheader("📊 Предпросмотр данных")
            st.write(f"**Размер:** {len(df)} строк × {len(df.columns)} столбцов")
            
            # Показываем первые строки
            with st.expander("👀 Посмотреть данные"):
                st.dataframe(df.head(10))
            
            st.subheader("🎯 Выберите столбец с ответами")
            selected_column = st.selectbox("Столбец с ответами:", df.columns, index=0)
            
            st.subheader("📋 Выбор диапазона для оценки")
            
            col1, col2 = st.columns(2)
            with col1:
                start_row = st.number_input(
                    "Начальная строка:",
                    min_value=0,
                    max_value=len(df)-1,
                    value=0,
                    help="Нумерация с 0"
                )
            with col2:
                end_row = st.number_input(
                    "Конечная строка:",
                    min_value=1,
                    max_value=len(df),
                    value=min(1000, len(df)),
                    help="Не включительно (как в Python slicing)"
                )
            
            # Показываем выбранный диапазон
            if start_row < end_row:
                selected_range_df = df.iloc[start_row:end_row]
                st.info(f"**Выбран диапазон:** строки {start_row}-{end_row} ({len(selected_range_df)} строк)")
                
                with st.expander("👀 Посмотреть выбранные строки"):
                    st.dataframe(selected_range_df.head(10))
                
                # Быстрые варианты диапазонов
                st.subheader("⚡ Быстрый выбор")
                quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
                
                with quick_col1:
                    if st.button("Первые 100", use_container_width=True):
                        st.session_state.start_row = 0
                        st.session_state.end_row = 100
                        st.rerun()
                with quick_col2:
                    if st.button("Первые 1000", use_container_width=True):
                        st.session_state.start_row = 0
                        st.session_state.end_row = 1000
                        st.rerun()
                with quick_col3:
                    if st.button("Последние 100", use_container_width=True):
                        st.session_state.start_row = max(0, len(df) - 100)
                        st.session_state.end_row = len(df)
                        st.rerun()
                with quick_col4:
                    if st.button("Все строки", use_container_width=True):
                        st.session_state.start_row = 0
                        st.session_state.end_row = len(df)
                        st.rerun()
                
                # Настройки обработки
                st.subheader("⚙️ Настройки обработки")
                chunk_size = st.slider("Размер части:", 100, 1000, 500, 100)
                
                if st.button("🚀 Начать оценку выбранного диапазона", type="primary"):
                    if not st.session_state.processing_state['is_processing']:
                        st.session_state.processing_state['is_processing'] = True
                        
                        result_df = process_dataset_range(
                            df, 
                            st.session_state.grader_instance, 
                            selected_column, 
                            start_row, 
                            end_row, 
                            chunk_size
                        )
                        
                        st.session_state.processing_state['is_processing'] = False
                        
                        if result_df is not None:
                            st.session_state.graded_results = result_df
                            st.session_state.processing_state['selected_column'] = selected_column
                            st.success("✅ Результаты готовы! Перейдите на вкладку 'Результаты'")
            else:
                st.error("❌ Начальная строка должна быть меньше конечной")

with tab3:
    st.header("📈 Результаты оценки")
    
    if st.session_state.graded_results is not None:
        result_df = st.session_state.graded_results
        
        st.success(f"✅ Обработано ответов: {len(result_df)}")
        
        # Статистика
        st.subheader("📊 Статистика оценок")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_grade = result_df['predicted_grade'].mean()
            st.metric("Средняя оценка", f"{avg_grade:.2f}")
        with col2:
            min_grade = result_df['predicted_grade'].min()
            st.metric("Мин. оценка", f"{min_grade:.2f}")
        with col3:
            max_grade = result_df['predicted_grade'].max()
            st.metric("Макс. оценка", f"{max_grade:.2f}")
        with col4:
            st.metric("Всего ответов", len(result_df))
        
        # Распределение
        st.subheader("📈 Распределение оценок")
        grade_counts = result_df['predicted_grade'].value_counts().sort_index()
        st.bar_chart(grade_counts)
        
        # Таблица
        st.subheader("📋 Детали оценок")
        selected_column = st.session_state.processing_state.get('selected_column', 'answer')
        
        page_size = st.slider("Строк на странице:", 10, 100, 20)
        page = st.number_input("Страница:", min_value=1, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(
            result_df.iloc[start_idx:end_idx][[selected_column, 'predicted_grade']],
            height=400
        )
        
        # Скачивание
        st.subheader("💾 Скачать результаты")
        csv_data = result_df.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button(
            label=f"📥 Скачать ({len(result_df)} строк)",
            data=csv_data,
            file_name="graded_answers.csv",
            mime="text/csv"
        )
        
    else:
        st.info("ℹ️ Результаты появятся здесь после обработки файла")

# Боковая панель
with st.sidebar:
    st.header("ℹ️ О системе")
    st.markdown("""
    **Формат файла:**
    - CSV с кодировкой UTF-8
    - Разделитель: ;
    - Столбец с текстом ответов
    """)
    
    st.header("📊 Статус")
    if st.session_state.model_loaded:
        st.success("✅ Модель загружена")
    else:
        st.error("❌ Модель не загружена")
    
    if st.session_state.uploaded_data is not None:
        st.info(f"📁 Файл: {len(st.session_state.uploaded_data)} строк")
    
    if st.session_state.processing_state['is_processing']:
        st.warning("🔄 Идет обработка")
    else:
        st.success("✅ Готов к работе")

st.markdown("---")
st.markdown("**Система оценки экзаменационных ответов** • Выбор диапазона строк • UTF-8 с разделителем ';'")
