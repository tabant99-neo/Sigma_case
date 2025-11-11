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
    page_title="Автоматическая оценка экзамена по русскому языку",
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

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'grader_instance' not in st.session_state:
    st.session_state.grader_instance = None

# Заголовок приложения
st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
st.markdown("""
Это демо-версия системы оценки письменных ответов на русском языке.
Загрузите CSV-файл с ответами студентов или введите текст вручную.
""")

# Основной класс для оценки с обученной моделью
class RussianExamGrader:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"🔄 Загружаем модель из: {model_path}")
        
        try:
            # Проверяем существование пути к модели
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Переносим модель на устройство (GPU/CPU)
            self.model.to(self.device)
            self.model.eval()
            
            st.success("✅ Обученная модель успешно загружена!")
            st.session_state.model_loaded = True
            
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке модели: {e}")
            st.info("""
            **Решение проблемы:**
            1. Убедитесь, что папка с моделью существует
            2. Проверьте наличие файлов: pytorch_model.bin, config.json
            3. Модель должна быть в папке 'my_trained_model_2'
            """)
            raise e

    def preprocess_text(self, text):
        """Базовая очистка текста."""
        text = str(text).strip()
        if not text:
            return ""
        # Минимальная предобработка - модель обучена на оригинальных текстах
        return text

    def predict(self, text):
        """Предсказание оценки для одного текста."""
        try:
            if not text or len(str(text).strip()) == 0:
                return 0.0
                
            processed_text = self.preprocess_text(text)
            
            # Токенизация
            inputs = self.tokenizer(
                processed_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = outputs.logits.cpu().numpy()

            # Преобразуем в оценку 0-5
            grade = float(prediction[0][0])
            grade = max(0, min(5, grade))
            return round(grade, 2)
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            return 0.0

    def predict_batch(self, texts: List[str]) -> List[float]:
        """Пакетное предсказание для ускорения обработки."""
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Пакетная токенизация
            inputs = self.tokenizer(
                processed_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # Пакетное предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.cpu().numpy()

            # Преобразуем все оценки
            grades = predictions[:, 0].tolist()
            grades = [max(0, min(5, float(grade))) for grade in grades]
            return [round(grade, 2) for grade in grades]
            
        except Exception as e:
            st.error(f"Ошибка при пакетном предсказании: {e}")
            # Резервный вариант - обработка по одному
            return [self.predict(text) for text in texts]

# Функция для безопасного чтения CSV с приоритетом для UTF-8 и разделителя ';'
def safe_read_csv(uploaded_file):
    """Безопасное чтение CSV с приоритетом для UTF-8 и разделителя ';'"""
    
    # Сначала пробуем UTF-8 с разделителем ';' (основной вариант)
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=';', on_bad_lines='skip')
        if len(df.columns) > 0 and len(df) > 0:
            st.success(f"✅ Файл прочитан с кодировкой UTF-8 и разделителем ';'")
            st.info(f"📊 Данные: {len(df)} строк, {len(df.columns)} столбцов")
            return df
    except Exception as e:
        st.warning(f"⚠️ Не удалось прочитать с UTF-8 и ';': {e}")
    
    # Затем пробуем другие варианты
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1', 'latin1']
    separators = [',', '\t']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            for sep in separators:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep, on_bad_lines='skip')
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
    
    # Последняя попытка с engine='python'
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
        if len(df) > 0:
            st.info("Файл прочитан с автоматическим определением разделителя")
            return df
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
    
    raise ValueError("Не удалось прочитать файл. Убедитесь, что файл в формате CSV с кодировкой UTF-8 и разделителем ';'")

# Оптимизированная функция для обработки больших CSV файлов
def process_large_dataset(df, grader, selected_column, chunk_size=500):
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
        start_time = time.time()
        
        # Обрабатываем файл по частям
        for start_idx in range(0, total_rows, chunk_size):
            if not st.session_state.processing_state['is_processing']:
                st.warning("Обработка остановлена пользователем")
                break
                
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]
            
            # Обновляем прогресс
            with progress_container:
                progress = end_idx / total_rows
                st.progress(progress)
            
            # Обновляем статус
            with status_container:
                elapsed = time.time() - start_time
                rows_per_sec = end_idx / elapsed if elapsed > 0 else 0
                remaining_time = (total_rows - end_idx) / rows_per_sec if rows_per_sec > 0 else 0
                
                st.write(f"""
                **Прогресс:** {end_idx}/{total_rows} строк ({progress:.1%})
                **Скорость:** {rows_per_sec:.1f} строк/сек
                **Осталось:** {remaining_time:.0f} сек
                """)
            
            # Обрабатываем текущую часть
            answers = chunk[selected_column].astype(str).tolist()
            chunk_grades = grader.predict_batch(answers)
            all_grades.extend(chunk_grades)
            
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
            time.sleep(0.1)  # Небольшая задержка для обновления UI
        
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
def load_grader():
    model_path = "my_trained_model_2"
    
    # Проверяем различные возможные пути
    possible_paths = [
        model_path,
        f"./{model_path}",
        f"../{model_path}",
        f"../../{model_path}",
        "C:/Users/tkubanychbekov/Documents/Russian_exam_grader/my_trained_model_2"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.info(f"🎯 Найдена модель по пути: {path}")
            return RussianExamGrader(path)
    
    # Если модель не найдена
    st.error("❌ Модель не найдена по указанным путям")
    st.info("""
    **Пожалуйста, убедитесь что:**
    1. Папка 'my_trained_model_2' находится в той же директории, что и этот скрипт
    2. В папке есть файлы: pytorch_model.bin, config.json, tokenizer_config.json
    3. Модель была успешно обучена и сохранена
    """)
    return None

# Основной интерфейс
tab1, tab2, tab3 = st.tabs(["🎯 Оценить один ответ", "📊 Оценить файл CSV", "📈 Результаты"])

with tab1:
    st.header("Оценка одного ответа")
    
    # Сначала загружаем модель если еще не загружена
    if not st.session_state.model_loaded:
        if st.button("🔄 Загрузить модель для оценки", key="load_model_single"):
            with st.spinner("Загружаем модель..."):
                grader = load_grader()
                if grader is None:
                    st.error("Не удалось загрузить модель")
                else:
                    st.session_state.grader_instance = grader
                    st.rerun()
    else:
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
                    grade = st.session_state.grader_instance.predict(user_input)
                    processing_time = time.time() - start_time
                
                st.success(f"**Предсказанная оценка: {grade} / 5**")
                st.info(f"⏱️ Время обработки: {processing_time:.2f} сек")
                
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
    **Рекомендуемый формат:** UTF-8 с разделителем ';'
    """)
    
    # Сначала проверяем загружена ли модель
    if not st.session_state.model_loaded:
        st.warning("⚠️ Модель не загружена")
        if st.button("🔄 Загрузить модель для пакетной обработки", key="load_model_batch"):
            with st.spinner("Загружаем модель..."):
                grader = load_grader()
                if grader is not None:
                    st.session_state.grader_instance = grader
                    st.success("✅ Модель готова к работе!")
                    st.rerun()
    else:
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
                    min_value=100,
                    max_value=1000,
                    value=500,
                    step=100,
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
                    
                    # Запускаем обработку с использованием grader из session_state
                    result_df = process_large_dataset(
                        df, 
                        st.session_state.grader_instance, 
                        selected_column, 
                        chunk_size
                    )
                    
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
        
        # Показываем данные с пагинацией
        page_size = 100
        total_pages = max(1, len(result_df) // page_size)
        
        page = st.number_input("Страница", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(result_df))
        
        selected_column = st.session_state.processing_state.get('selected_column', 'answer')
        
        st.dataframe(
            result_df.iloc[start_idx:end_idx][
                [selected_column, 'predicted_grade']
            ],
            height=400
        )
        
        # Скачивание результатов
        st.subheader("💾 Скачать результаты")
        
        csv_data = result_df.to_csv(index=False, sep=';').encode('utf-8')
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
    **Используется:**
    - Обученная модель DeepPavlov
    - Пакетная обработка
    - Поддержка больших файлов
    - Формат: CSV (UTF-8, ;)
    """)
    
    st.header("📊 Статус")
    if st.session_state.model_loaded:
        st.success("✅ Модель загружена")
    else:
        st.error("❌ Модель не загружена")
    
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
st.markdown("**Система оценки экзаменационных ответов** • Обученная модель DeepPavlov • Формат: CSV (UTF-8, ;)")
