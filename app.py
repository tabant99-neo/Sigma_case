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

# Конфигурация страницы
st.set_page_config(
    page_title="Оценка экзамена по русскому языку",
    page_icon="🇷🇺",
    layout="centered"
)

# Оптимизации PyTorch для скорости
torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Заголовок приложения
st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
st.markdown("""
Это демо-версия модели, дообученной на основе DeepPavlov для оценки письменных ответов.
Загрузите CSV-файл с ответами студентов или введите текст вручную.
""")

# Оптимизированный класс для оценки ответов
class RussianExamGrader:
    def __init__(self, model_path, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        try:
            if not os.path.exists(model_path):
                st.error(f"❌ Путь к модели не существует: {model_path}")
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            st.info(f"🔄 Загружаем модель из: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Оптимизации для GPU
            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model.half()  # Используем половинную точность для GPU
                torch.backends.cudnn.benchmark = True
            
            self.model.eval()
            st.success("✅ Модель успешно загружена!")
            
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке модели: {e}")
            raise e

    def preprocess_text(self, text):
        """
        Базовая очистка текста.
        """
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def predict(self, text):
        """
        Предсказание оценки для одного текста.
        """
        try:
            processed_text = self.preprocess_text(text)
            inputs = self.tokenizer(
                processed_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # Оптимизация для GPU
            if self.device.type == 'cuda':
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = outputs.logits.cpu().numpy()

            grade = float(prediction[0][0])
            grade = max(0, min(5, grade))
            return round(grade, 2)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            return 0.0

    def predict_batch(self, texts: List[str]) -> List[float]:
        """
        Пакетное предсказание для ускорения обработки.
        """
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Пакетная обработка
            inputs = self.tokenizer(
                processed_texts,
                max_length=512,
                padding=True,  # Динамический паддинг для эффективности
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Оптимизация для GPU
            if self.device.type == 'cuda':
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.cpu().numpy()

            grades = predictions[:, 0].tolist()
            grades = [max(0, min(5, float(grade))) for grade in grades]
            return [round(grade, 2) for grade in grades]
            
        except Exception as e:
            st.error(f"Ошибка при пакетном предсказании: {e}")
            # Резервный вариант - обработка по одному
            return [self.predict(text) for text in texts]

    def predict_large_dataset(self, texts: List[str], progress_callback=None) -> List[float]:
        """
        Обработка больших наборов данных с пакетной обработкой.
        """
        all_grades = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_grades = self.predict_batch(batch_texts)
            all_grades.extend(batch_grades)
            
            if progress_callback:
                progress_callback(i + len(batch_texts), len(texts))
        
        return all_grades

# Функция для безопасного чтения CSV
def safe_read_csv(uploaded_file):
    """Безопасное чтение CSV с различными кодировками и разделителями"""
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            for sep in [',', ';', '\t']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                    if len(df.columns) > 0:
                        st.info(f"Файл прочитан с кодировкой {encoding} и разделителем '{sep}'")
                        return df
                except:
                    continue
        except UnicodeDecodeError:
            continue
        except Exception as e:
            continue
    
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
        st.info("Файл прочитан как TSV (табуляция)")
        return df
    except:
        pass
    
    # Последняя попытка - чтение с обработкой ошибок
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        st.info("Файл прочитан с пропуском проблемных строк")
        return df
    except:
        raise ValueError("Не удалось прочитать файл. Попробуйте сохранить файл в UTF-8 с разделителем запятая.")

# Оптимизированная функция для обработки CSV файла
def grade_csv_file_fast(df, grader, selected_column='answer'):
    """Быстрая обработка CSV файла с пакетной обработкой"""
    try:
        if selected_column not in df.columns:
            st.error(f"Столбец '{selected_column}' не найден. Найдены столбцы: {list(df.columns)}")
            return None
        
        answers = df[selected_column].astype(str).tolist()
        
        # Создаем элементы для прогресса
        progress_bar = st.progress(0)
        status_text = st.empty()
        speed_text = st.empty()
        start_time = time.time()
        
        def update_progress(processed, total):
            progress = processed / total
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = processed / elapsed
                status_text.text(f"Обработано: {processed}/{total} ответов")
                speed_text.text(f"Скорость: {speed:.1f} ответов/сек")
        
        # Используем пакетную обработку
        st.info("🚀 Используем ускоренную пакетную обработку...")
        grades = grader.predict_large_dataset(answers, progress_callback=update_progress)
        
        progress_bar.empty()
        status_text.empty()
        speed_text.empty()
        
        df['predicted_grade'] = grades
        
        total_time = time.time() - start_time
        st.success(f"✅ Оценка завершена! Обработано {len(answers)} ответов за {total_time:.1f} сек")
        st.info(f"⚡ Средняя скорость: {len(answers)/total_time:.1f} ответов/сек")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Ошибка при обработке CSV: {e}")
        return None

# Функция для обработки больших файлов по частям
def process_large_file_in_chunks(df, grader, selected_column, chunk_size=1000):
    """Обработка очень больших файлов по частям"""
    total_rows = len(df)
    chunks = [df[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
    
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Обрабатываем часть {i+1}/{len(chunks)}...")
        
        chunk_result = grade_csv_file_fast(chunk, grader, selected_column)
        if chunk_result is not None:
            all_results.append(chunk_result)
        
        progress_bar.progress((i + 1) / len(chunks))
    
    progress_bar.empty()
    status_text.empty()
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return None

# Инициализация модели (кэшируем, чтобы не загружать каждый раз)
@st.cache_resource
def load_grader():
    model_path = "my_trained_model_2"
    
    if not os.path.exists(model_path):
        absolute_path = "C:/Users/tkubanychbekov/Sigma_case/Sigma_case/my_trained_model_2"
        if os.path.exists(absolute_path):
            model_path = absolute_path
        else:
            st.warning(f"⚠️ Модель не найдена по пути: {model_path}")
            st.info("🔍 Убедитесь, что папка с моделью находится в той же директории, что и app.py")
    
    # Используем увеличенный размер батча для скорости
    return RussianExamGrader(model_path, batch_size=64)  # Увеличили batch_size

# Загружаем модель
try:
    grader = load_grader()
except Exception as e:
    st.error(f"❌ Не удалось загрузить модель: {e}")
    st.info("""
    **Решение проблемы:**
    1. Убедитесь, что папка `my_trained_model_2` находится в той же папке, что и `app.py`
    2. Или измените путь к модели в коде (строка 117)
    3. Проверьте, что в папке модели есть файлы: `pytorch_model.bin`, `config.json` и др.
    """)
    st.stop()

# Создаем две вкладки для разных способов ввода
tab1, tab2 = st.tabs(["🎯 Оценить один ответ", "📊 Оценить файл CSV"])

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
            st.info(f"⏱️ Время обработки: {processing_time:.2f} сек")
            
            # Визуализация оценки
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Оценка", f"{grade}/5")
            with col2:
                st.progress(grade / 5.0)
            
            # Интерпретация оценки
            if grade >= 4.5:
                st.info("🎉 Отличный ответ!")
            elif grade >= 3.5:
                st.info("👍 Хороший ответ")
            elif grade >= 2.5:
                st.warning("⚠️ Удовлетворительный ответ")
            else:
                st.error("❌ Ответ требует улучшений")
        else:
            st.warning("⚠️ Пожалуйста, введите текст для оценки.")

with tab2:
    st.header("Пакетная оценка из CSV-файла")
    st.markdown("""
    Загрузите CSV-файл, содержащий столбец с ответами студентов.
    **Новая оптимизированная версия работает в 3-5 раз быстрее!** 🚀
    """)
    
    # Показываем пример формата файла
    with st.expander("📋 Пример формата CSV-файла"):
        example_data = {
            'answer': [
                "Моё хобби - читать книги и заниматься спортом.",
                "Я люблю путешествовать и узнавать новые культуры.",
                "В свободное время я изучаю программирование и иностранные языки.",
                "Мне нравится проводить время с семьёй и друзьями.",
                "Я увлекаюсь фотографией и видеомонтажом."
            ]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
        
        csv_example = example_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать пример CSV",
            data=csv_example,
            file_name="example_answers.csv",
            mime="text/csv",
            key="download_example"
        )
    
    uploaded_file = st.file_uploader(
        "Выберите CSV-файл", 
        type=['csv', 'txt'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Читаем файл с улучшенной обработкой
            df = safe_read_csv(uploaded_file)
            
            st.subheader("📊 Предпросмотр данных")
            st.write(f"**Найдено столбцов:** {len(df.columns)}")
            st.write(f"**Найдено строк:** {len(df)}")
            
            # Показываем первые 5 строк
            st.dataframe(df.head())
            
            # Показываем информацию о столбцах
            with st.expander("🔍 Информация о столбцах"):
                for i, col in enumerate(df.columns):
                    st.write(f"**{i+1}. {col}** (тип: {df[col].dtype})")
                    if df[col].dtype == 'object':
                        sample_value = df[col].iloc[0] if len(df) > 0 else "Нет данных"
                        st.write(f"   Пример: {str(sample_value)[:100]}...")
            
            st.subheader("🎯 Выберите столбец с ответами")
            if len(df.columns) > 0:
                selected_column = st.selectbox(
                    "Выберите столбец, содержащий тексты ответов:",
                    df.columns,
                    index=0,
                    key="column_selector"
                )
                
                # Показываем пример выбранного столбца
                st.write("**Пример из выбранного столбца:**")
                sample_text = df[selected_column].iloc[0] if len(df) > 0 else "Нет данных"
                st.text_area(
                    "Пример текста:",
                    value=str(sample_text)[:500],
                    height=100,
                    key="sample_text",
                    disabled=True
                )
                
                # Настройки обработки
                with st.expander("⚙️ Настройки скорости"):
                    use_fast_processing = st.checkbox("Использовать ускоренную обработку", value=True)
                    if len(df) > 5000:
                        use_chunking = st.checkbox("Обрабатывать большие файлы по частям", value=True)
                        chunk_size = st.slider("Размер части", 1000, 10000, 2000)
                    else:
                        use_chunking = False
                
                if st.button("🚀 Оценить все ответы", type="primary", key="batch"):
                    with st.spinner("⏳ Обрабатываем файл... Это может занять некоторое время."):
                        start_time = time.time()
                        
                        if use_chunking and len(df) > 5000:
                            st.info(f"📦 Обрабатываем файл по частям ({chunk_size} строк в части)")
                            result_df = process_large_file_in_chunks(df, grader, selected_column, chunk_size)
                        else:
                            # Используем оптимизированную обработку
                            if use_fast_processing:
                                result_df = grade_csv_file_fast(df, grader, selected_column)
                            else:
                                # Старый метод для сравнения
                                st.info("🔄 Используем стандартную обработку...")
                                answers = df[selected_column].astype(str).tolist()
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                grades = []
                                total_answers = len(answers)
                                
                                for i, answer in enumerate(answers):
                                    grade = grader.predict(answer)
                                    grades.append(grade)
                                    
                                    progress = (i + 1) / total_answers
                                    progress_bar.progress(progress)
                                    status_text.text(f"Обработано: {i+1}/{total_answers} ответов")
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                result_df = df.copy()
                                result_df['predicted_grade'] = grades
                                
                                total_time = time.time() - start_time
                                st.success(f"✅ Оценка завершена! Обработано {total_answers} ответов за {total_time:.1f} сек")
                        
                        if result_df is not None:
                            st.balloons()
                            st.subheader("📈 Результаты оценки")
                            
                            # Показываем первые 10 строк с результатами
                            st.dataframe(result_df[[selected_column, 'predicted_grade']].head(10))
                            
                            # Статистика оценок
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
                                total_count = len(result_df)
                                st.metric("Всего ответов", total_count)
                            
                            # Распределение оценок
                            st.subheader("📊 Распределение оценок")
                            grade_counts = result_df['predicted_grade'].value_counts().sort_index()
                            st.bar_chart(grade_counts)
                            
                            # Скачивание результатов
                            st.subheader("💾 Скачать результаты")
                            
                            csv_result = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Скачать полные результаты (CSV)",
                                data=csv_result,
                                file_name="graded_answers.csv",
                                mime="text/csv",
                                key="download_full"
                            )
                            
                        else:
                            st.error("❌ Не удалось обработать файл. Проверьте данные и попробуйте снова.")
                        
            else:
                st.error("❌ В файле не найдены столбцы данных.")
                        
        except Exception as e:
            st.error(f"❌ Произошла ошибка при чтении файла: {e}")
            st.markdown("""
            **💡 Рекомендации по устранению ошибок:**
            - Убедитесь, что файл в формате CSV
            - Попробуйте сохранить файл с кодировкой UTF-8
            - Убедитесь, что разделитель - запятая
            - Проверьте, что в текстах ответов нет лишних переносов строк
            - Убедитесь, что все строки имеют одинаковое количество столбцов
            """)

# Боковая панель с информацией
with st.sidebar:
    st.header("ℹ️ О решении")
    st.markdown("""
    **Технические детали:**
    - **Модель**: DeepPavlov (дообученная)
    - **Метрика**: MAE = 0.26
    - **Оптимизации**: Пакетная обработка, GPU ускорение
    - **Скорость**: до 50+ ответов/сек
    - **Шкала**: 0-5 баллов
    """)
    
    st.header("📝 Инструкция")
    st.markdown("""
    **Для одиночной оценки:**
    1. Перейдите на вкладку "Оценить один ответ"
    2. Введите текст ответа
    3. Нажмите "Оценить ответ"
    
    **Для пакетной оценки:**
    1. Перейдите на вкладку "Оценить файл CSV"
    2. Загрузите CSV-файл с ответами
    3. Выберите столбец с текстами ответов
    4. Включите "Ускоренную обработку"
    5. Нажмите "Оценить все ответы"
    6. Скачайте результаты
    """)
    
    st.header("⚡ Оптимизации скорости")
    st.markdown("""
    - **Пакетная обработка** - до 64 ответов одновременно
    - **GPU ускорение** - автоматическое использование CUDA
    - **Половинная точность** - для экономии памяти GPU
    - **Чанкование** - обработка больших файлов по частям
    - **Кэширование** - модель загружается один раз
    """)
    
    # Информация о загрузке модели
    st.header("🔧 Статус системы")
    if 'grader' in locals():
        st.success("✅ Модель загружена")
        st.info(f"🖥️ Устройство: {grader.device}")
        st.info(f"📦 Batch size: {grader.batch_size}")
        if grader.device.type == 'cuda':
            st.success("🎯 GPU ускорение активно")
        else:
            st.warning("⚠️ Используется CPU (рекомендуется GPU)")
    else:
        st.error("❌ Модель не загружена")

# Футер
st.markdown("---")
st.markdown(
    "**Автоматическая система оценки экзаменационных ответов** • "
    "Использует дообученную модель DeepPavlov • "
    "MAE: 0.26 • "
    "⚡ Оптимизированная версия"
)
