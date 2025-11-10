import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os
import re
import numpy as np

# Конфигурация страницы
st.set_page_config(
    page_title="Оценка экзамена по русскому языку",
    page_icon="🇷🇺",
    layout="centered"
)

# Заголовок приложения
st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
st.markdown("""
Это демо-версия модели, дообученной на основе DeepPavlov для оценки письменных ответов.
Загрузите CSV-файл с ответами студентов или введите текст вручную.
""")

# Класс для оценки ответов
class RussianExamGrader:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # Проверяем существование пути к модели
            if not os.path.exists(model_path):
                st.error(f"❌ Путь к модели не существует: {model_path}")
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            st.info(f"🔄 Загружаем модель из: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
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

            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = outputs.logits.cpu().numpy()

            grade = float(prediction[0][0])
            grade = max(0, min(5, grade))
            return round(grade, 2)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            return 0.0

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
                    if len(df.columns) > 0:  # Изменил условие на > 0
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

# Функция для обработки CSV файла
def grade_csv_file(csv_path, grader, output_path='graded_output.csv'):
    """Обработка CSV файла с ответами"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        if 'answer' not in df.columns:
            st.error(f"Столбец 'answer' не найден. Найдены столбцы: {list(df.columns)}")
            return None
        
        answers = df['answer'].astype(str).tolist()
        
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
        
        df['predicted_grade'] = grades
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        st.success(f"✅ Оценка завершена! Обработано {total_answers} ответов.")
        return df
        
    except Exception as e:
        st.error(f"❌ Ошибка при обработке CSV: {e}")
        return None

# Инициализация модели (кэшируем, чтобы не загружать каждый раз)
@st.cache_resource
def load_grader():
    # УКАЖИТЕ ПРАВИЛЬНЫЙ ПУТЬ К ВАШЕЙ МОДЕЛИ
    model_path = "my_trained_model_2"  # Изменил на ваше название папки
    
    # Если модель не находится в текущей директории, укажите полный путь
    if not os.path.exists(model_path):
        # Попробуем найти модель в абсолютном пути
        absolute_path = "C:/Users/tkubanychbekov/Documents/Russian_exam_grader/my_trained_model_2"
        if os.path.exists(absolute_path):
            model_path = absolute_path
        else:
            # Если не нашли, используем относительный путь
            st.warning(f"⚠️ Модель не найдена по пути: {model_path}")
            st.info("🔍 Убедитесь, что папка с моделью находится в той же директории, что и app.py")
    
    return RussianExamGrader(model_path)

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
                grade = grader.predict(user_input)
            
            st.success(f"**Предсказанная оценка: {grade} / 5**")
            
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
    Файл должен содержать как минимум один столбец с текстовыми ответами.
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
                
                if st.button("🚀 Оценить все ответы", type="primary", key="batch"):
                    with st.spinner("⏳ Обрабатываем файл... Это может занять некоторое время."):
                        # Создаем временный файл для исходных данных
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as tmp_input:
                            df_processed = df.rename(columns={selected_column: 'answer'})
                            # Сохраняем только нужные колонки для экономии памяти
                            df_processed[['answer']].to_csv(tmp_input.name, index=False)
                            tmp_input_path = tmp_input.name
                        
                        # Создаем временный файл для результатов
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_output:
                            tmp_output_path = tmp_output.name
                        
                        # Обрабатываем файл
                        try:
                            result_df = grade_csv_file(tmp_input_path, grader, tmp_output_path)
                            
                            if result_df is not None:
                                st.balloons()
                                st.subheader("📈 Результаты оценки")
                                
                                # Показываем первые 10 строк с результатами
                                st.dataframe(result_df[['answer', 'predicted_grade']].head(10))
                                
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
                                
                                # Гистограмма
                                grade_counts = result_df['predicted_grade'].value_counts().sort_index()
                                st.bar_chart(grade_counts)
                                
                                # Детальная статистика
                                with st.expander("🔍 Детальная статистика"):
                                    st.write("**Описательная статистика:**")
                                    st.write(result_df['predicted_grade'].describe())
                                    
                                    st.write("**Распределение по диапазонам:**")
                                    bins = [0, 1, 2, 3, 4, 5]
                                    labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
                                    result_df['grade_range'] = pd.cut(result_df['predicted_grade'], bins=bins, labels=labels)
                                    range_counts = result_df['grade_range'].value_counts().sort_index()
                                    st.bar_chart(range_counts)
                                
                                # Скачивание результатов
                                st.subheader("💾 Скачать результаты")
                                
                                # Восстанавливаем оригинальные данные с добавленной колонкой оценки
                                final_result = df.copy()
                                final_result['predicted_grade'] = result_df['predicted_grade']
                                
                                csv_result = final_result.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Скачать полные результаты (CSV)",
                                    data=csv_result,
                                    file_name="graded_answers.csv",
                                    mime="text/csv",
                                    key="download_full"
                                )
                                
                            else:
                                st.error("❌ Не удалось обработать файл. Проверьте данные и попробуйте снова.")
                                
                        except Exception as processing_error:
                            st.error(f"❌ Ошибка при обработке данных: {processing_error}")
                            st.info("💡 Попробуйте проверить формат CSV файла")
                        finally:
                            # Удаляем временные файлы
                            if os.path.exists(tmp_input_path):
                                os.unlink(tmp_input_path)
                            if os.path.exists(tmp_output_path):
                                os.unlink(tmp_output_path)
                        
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
    - **Функция**: Оценка письменных ответов на русском языке
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
    4. Нажмите "Оценить все ответы"
    5. Скачайте результаты
    """)
    
    st.header("⚙️ Требования к данным")
    st.markdown("""
    - CSV-файл с кодировкой UTF-8
    - Столбец с текстовыми ответами
    - Максимальная длина ответа: ~512 токенов
    - Поддерживаемые языки: русский
    """)
    
    # Информация о загрузке модели
    st.header("🔧 Статус системы")
    if 'grader' in locals():
        st.success("✅ Модель загружена")
        st.info(f"🖥️ Устройство: {grader.device}")
        st.info(f"📁 Путь к модели: {os.path.abspath('my_trained_model_2')}")
    else:
        st.error("❌ Модель не загружена")

# Футер
st.markdown("---")
st.markdown(
    "**Автоматическая система оценки экзаменационных ответов** • "
    "Использует дообученную модель DeepPavlov • "
    "MAE: 0.26"
)
