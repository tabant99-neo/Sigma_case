# sigma_case/app.py
import streamlit as st
import pandas as pd
import time
import sys
import os

# Добавляем текущую директорию в путь для импортов
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

# Теперь импортируем из текущей папки
from grader import RussianExamGrader
from utils import safe_read_csv, check_model_files, get_model_path

# Конфигурация страницы
st.set_page_config(
    page_title="Russian Exam Grader",
    page_icon="🇷🇺",
    layout="wide"
)

@st.cache_resource
def load_grader():
    return RussianExamGrader()

def main():
    """Основная функция приложения"""
    
    # Заголовок приложения
    st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
    st.markdown("""
    **⚡ Версия с интегрированной ML моделью**  
    Загрузите CSV-файл с транскрибациями ответов для оценки.
    """)

    # Загружаем градер
    grader = load_grader()

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
        model_path = get_model_path()
        existing_files, missing_files = check_model_files(model_path)
        
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
        "Интегрированная ML модель • "
        "Автоматический режим ML/Демо"
    )

if __name__ == "__main__":
    main()
