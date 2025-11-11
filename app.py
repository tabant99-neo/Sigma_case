import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os
import re
import numpy as np
from typing import List
import time
import gc

# Конфигурация страницы
st.set_page_config(
    page_title="Оценка экзамена по русскому языку",
    page_icon="🇷🇺",
    layout="wide"
)

# Оптимизации PyTorch для скорости
torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Инициализация состояния сессии
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = {
        'is_processing': False,
        'current_index': 0,
        'total_rows': 0,
        'start_time': 0,
        'results': None,
        'original_df': None,
        'selected_column': None
    }

if 'graded_results' not in st.session_state:
    st.session_state.graded_results = None

# Заголовок приложения
st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
st.markdown("""
Это демо-версия системы оценки письменных ответов на русском языке.
Загрузите CSV-файл с ответами студентов или введите текст вручную.
""")

# Упрощенная модель для демонстрации
class SimpleRussianGrader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Эвристики для оценки русских текстов
        self.quality_indicators = {
            'length_weight': 0.2,
            'vocabulary_weight': 0.3,
            'structure_weight': 0.3,
            'grammar_weight': 0.2
        }
        
        # Примеры хороших фраз на русском
        self.good_phrases = [
            'мне кажется', 'по моему мнению', 'с одной стороны', 'с другой стороны',
            'таким образом', 'в заключение', 'во-первых', 'во-вторых', 'в-третьих',
            'кроме того', 'например', 'таким образом', 'следовательно', 'однако',
            'поэтому', 'в результате', 'в целом', 'подводя итог'
        ]

    def preprocess_text(self, text):
        """Базовая очистка текста."""
        text = str(text).lower().strip()
        if not text:
            return ""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def analyze_text_quality(self, text):
        """Анализ качества текста с помощью эвристик."""
        if not text or len(str(text).strip()) == 0:
            return 0.0
            
        text = self.preprocess_text(text)
        words = text.split()
        
        if len(words) == 0:
            return 0.0
        
        # 1. Оценка по длине
        length_score = min(len(words) / 30, 1.0)  # Нормализуем к 30 словам
        
        # 2. Оценка по разнообразию лексики
        unique_words = len(set(words))
        vocab_score = min(unique_words / max(len(words), 1) * 2, 1.0)
        
        # 3. Оценка структуры (наличие хороших фраз)
        structure_score = 0
        for phrase in self.good_phrases:
            if phrase in text:
                structure_score += 0.05
        structure_score = min(structure_score, 1.0)
        
        # 4. Простая оценка грамматики (количество очень коротких слов)
        short_words = sum(1 for word in words if len(word) <= 2)
        grammar_score = 1.0 - min(short_words / max(len(words), 1) * 1.5, 1.0)
        
        # Итоговая оценка
        final_score = (
            length_score * self.quality_indicators['length_weight'] +
            vocab_score * self.quality_indicators['vocabulary_weight'] +
            structure_score * self.quality_indicators['structure_weight'] +
            grammar_score * self.quality_indicators['grammar_weight']
        )
        
        return min(final_score * 5, 5.0)  # Масштабируем до 5 баллов

    def predict(self, text):
        """Предсказание оценки для одного текста."""
        try:
            return round(self.analyze_text_quality(text), 2)
        except Exception:
            return 0.0

    def predict_batch(self, texts: List[str]) -> List[float]:
        """Пакетное предсказание для ускорения обработки."""
        return [self.predict(text) for text in texts]

# Основной класс с использованием простой модели
class RussianExamGrader:
    def __init__(self):
        self.simple_grader = SimpleRussianGrader()
        st.success("✅ Модель инициализирована (упрощенная версия)")

    def predict(self, text):
        return self.simple_grader.predict(text)

    def predict_batch(self, texts: List[str]) -> List[float]:
        return self.simple_grader.predict_batch(texts)

# Функция для безопасного чтения CSV
def safe_read_csv(uploaded_file):
    """Безопасное чтение CSV с различными кодировками и разделителями"""
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            for sep in [',', ';', '\t']:
                try:
                    uploaded_file.seek(0)
                    # Используем chunksize для больших файлов
                    chunks = []
                    for chunk in pd.read_csv(uploaded_file, encoding=encoding, sep=sep, chunksize=10000):
                        chunks.append(chunk)
                    
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        if len(df.columns) > 0 and len(df) > 0:
                            st.success(f"✅ Файл прочитан: {len(df)} строк, {len(df.columns)} столбцов")
                            st.info(f"Кодировка: {encoding}, разделитель: '{sep}'")
                            return df
                except Exception as e:
                    continue
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    
    # Последняя попытка
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip', engine='python')
        if len(df) > 0:
            st.info("Файл прочитан с пропуском проблемных строк")
            return df
    except:
        pass
    
    raise ValueError("Не удалось прочитать файл")

# Оптимизированная функция для обработки больших CSV файлов
def process_large_dataset(df, grader, selected_column, chunk_size=1000):
    """Обработка больших датасетов по частям с сохранением прогресса"""
    try:
        if selected_column not in df.columns:
            st.error(f"Столбец '{selected_column}' не найден")
            return None
        
        total_rows = len(df)
        st.info(f"📊 Начало обработки: {total_rows} строк")
        
        # Создаем контейнеры для UI
        progress_container = st.container()
        status_container = st.container()
        stats_container = st.container()
        
        all_grades = []
        processed_rows = 0
        start_time = time.time()
        
        # Обрабатываем файл по частям
        for start_idx in range(0, total_rows, chunk_size):
            if not st.session_state.processing_state['is_processing']:
                st.warning("Обработка остановлена")
                break
                
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]
            
            # Обновляем UI
            with progress_container:
                progress = end_idx / total_rows
                st.progress(progress)
            
            with status_container:
                elapsed = time.time() - start_time
                rows_per_sec = end_idx / elapsed if elapsed > 0 else 0
                remaining_time = (total_rows - end_idx) / rows_per_sec if rows_per_sec > 0 else 0
                
                st.write(f"""
                **Прогресс:** {end_idx}/{total_rows} строк ({progress:.1%})
                **Скорость:** {rows_per_sec:.1f} строк/сек
                **Осталось:** {remaining_time:.0f} сек
                **Текущая часть:** {start_idx}-{end_idx}
                """)
            
            # Обрабатываем текущую часть
            answers = chunk[selected_column].astype(str).tolist()
            chunk_grades = grader.predict_batch(answers)
            all_grades.extend(chunk_grades)
            
            processed_rows = end_idx
            
            # Показываем промежуточную статистику
            with stats_container:
                if len(all_grades) > 0:
                    current_avg = np.mean(all_grades)
                    current_min = min(all_grades)
                    current_max = max(all_grades)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Средняя оценка", f"{current_avg:.2f}")
                    with col2:
                        st.metric("Мин. оценка", f"{current_min:.2f}")
                    with col3:
                        st.metric("Макс. оценка", f"{current_max:.2f}")
            
            # Принудительное обновление интерфейса
            st.rerun()
        
        # Завершаем обработку
        st.session_state.processing_state['is_processing'] = False
        
        if len(all_grades) == total_rows:
            result_df = df.copy()
            result_df['predicted_grade'] = all_grades
            
            total_time = time.time() - start_time
            st.success(f"✅ Обработка завершена! Обработано {len(all_grades)} ответов за {total_time:.1f} сек")
            st.info(f"⚡ Средняя скорость: {len(all_grades)/total_time:.1f} ответов/сек")
            
            return result_df
        else:
            st.error(f"❌ Обработано только {len(all_grades)} из {total_rows} строк")
            return None
            
    except Exception as e:
        st.error(f"❌ Ошибка при обработке: {e}")
        st.session_state.processing_state['is_processing'] = False
        return None

# Инициализация модели
@st.cache_resource
def load_grader():
    return RussianExamGrader()

# Загружаем модель
try:
    grader = load_grader()
except Exception as e:
    st.error(f"❌ Не удалось инициализировать систему оценки: {e}")
    st.stop()

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["🎯 Оценить один ответ", "📊 Оценить файл CSV", "📈 Результаты"])

with tab1:
    st.header("Оценка одного ответа")
    user_input = st.text_area(
        "Введите ответ студента на русском языке:",
        height=150,
        placeholder="Напишите здесь ответ на экзаменационный вопрос...",
        key="single_answer"
    )

    if st.button("Оценить ответ", type="primary", key="single"):
        if user_input.strip():
            with st.spinner("🤖 Модель оценивает ответ..."):
                start_time = time.time()
                grade = grader.predict(user_input)
                processing_time = time.time() - start_time
            
            st.success(f"**Предсказанная оценка: {grade} / 5**")
            
            # Визуализация оценки
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Оценка", f"{grade}/5")
            with col2:
                st.progress(grade / 5.0)
            
            # Интерпретация оценки
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
    
    # Показываем статус текущей обработки
    if st.session_state.processing_state['is_processing']:
        st.warning("🔄 Идет обработка файла...")
        if st.button("⏹️ Остановить обработку"):
            st.session_state.processing_state['is_processing'] = False
            st.rerun()
    
    st.markdown("""
    Загрузите CSV-файл с ответами студентов. Поддерживаются большие файлы (10,000+ строк).
    """)
    
    uploaded_file = st.file_uploader(
        "Выберите CSV-файл", 
        type=['csv'],
        key="file_uploader"
    )
    
    if uploaded_file is not None and not st.session_state.processing_state['is_processing']:
        try:
            # Читаем файл
            with st.spinner("📖 Читаем файл..."):
                df = safe_read_csv(uploaded_file)
            
            st.subheader("📊 Предпросмотр данных")
            st.write(f"**Размер данных:** {len(df)} строк × {len(df.columns)} столбцов")
            
            # Показываем информацию о данных
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Всего строк", len(df))
            with col2:
                st.metric("Всего столбцов", len(df.columns))
            
            # Показываем первые строки
            with st.expander("👀 Посмотреть первые 10 строк"):
                st.dataframe(df.head(10))
            
            # Выбор столбца с ответами
            st.subheader("🎯 Выберите столбец с ответами")
            selected_column = st.selectbox(
                "Выберите столбец, содержащий тексты ответов:",
                df.columns,
                index=0
            )
            
            # Настройки обработки
            st.subheader("⚙️ Настройки обработки")
            chunk_size = st.slider(
                "Размер части для обработки:",
                min_value=500,
                max_value=5000,
                value=1000,
                step=500,
                help="Меньшие значения используют меньше памяти, но могут работать медленнее"
            )
            
            if st.button("🚀 Начать оценку всех ответов", type="primary"):
                if len(df) > 10000:
                    st.warning(f"⚠️ Внимание: большой файл ({len(df)} строк). Обработка может занять несколько минут.")
                
                # Сохраняем состояние
                st.session_state.processing_state.update({
                    'is_processing': True,
                    'total_rows': len(df),
                    'original_df': df,
                    'selected_column': selected_column
                })
                
                # Запускаем обработку
                result_df = process_large_dataset(df, grader, selected_column, chunk_size)
                
                if result_df is not None:
                    st.session_state.graded_results = result_df
                    st.success("✅ Результаты готовы! Перейдите на вкладку 'Результаты'")
                    st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Ошибка при работе с файлом: {e}")

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
        
        # Распределение оценок
        st.subheader("📈 Распределение оценок")
        
        # Гистограмма
        grade_bins = pd.cut(result_df['predicted_grade'], 
                           bins=[0, 1, 2, 3, 4, 5], 
                           labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
        grade_distribution = grade_bins.value_counts().sort_index()
        st.bar_chart(grade_distribution)
        
        # Детальная таблица
        st.subheader("📋 Детали оценок")
        
        # Поиск и фильтрация
        search_col1, search_col2 = st.columns(2)
        with search_col1:
            min_filter = st.slider("Минимальная оценка", 0.0, 5.0, 0.0, 0.5)
        with search_col2:
            max_filter = st.slider("Максимальная оценка", 0.0, 5.0, 5.0, 0.5)
        
        filtered_df = result_df[
            (result_df['predicted_grade'] >= min_filter) & 
            (result_df['predicted_grade'] <= max_filter)
        ]
        
        st.write(f"**Найдено ответов:** {len(filtered_df)}")
        
        # Показываем данные с пагинацией
        page_size = 100
        total_pages = max(1, len(filtered_df) // page_size)
        
        page = st.number_input("Страница", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        st.dataframe(
            filtered_df.iloc[start_idx:end_idx][
                [st.session_state.processing_state['selected_column'], 'predicted_grade']
            ],
            height=400
        )
        
        # Скачивание результатов
        st.subheader("💾 Скачать результаты")
        
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Скачать все результаты ({len(result_df)} строк)",
            data=csv_data,
            file_name=f"graded_answers_{len(result_df)}_rows.csv",
            mime="text/csv"
        )
        
    else:
        st.info("ℹ️ Результаты оценки появятся здесь после обработки файла")

# Боковая панель
with st.sidebar:
    st.header("ℹ️ О системе")
    st.markdown("""
    **Возможности:**
    - Оценка отдельных ответов
    - Пакетная обработка CSV
    - Поддержка больших файлов
    - Сохранение результатов
    """)
    
    st.header("📊 Статус")
    if st.session_state.processing_state['is_processing']:
        st.warning("Идет обработка")
        progress = st.session_state.processing_state.get('current_index', 0) / max(1, st.session_state.processing_state.get('total_rows', 1))
        st.progress(progress)
    else:
        st.success("Готов к работе")
    
    if st.session_state.graded_results is not None:
        st.info(f"Обработано: {len(st.session_state.graded_results)} ответов")

# Футер
st.markdown("---")
st.markdown("**Система оценки экзаменационных ответов** • Поддержка больших файлов")
