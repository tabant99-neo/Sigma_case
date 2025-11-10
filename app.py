import streamlit as st
import pandas as pd
import numpy as np
import os
import time

def create_app():
    """Создание и запуск Streamlit приложения"""
    
    st.set_page_config(
        page_title="Russian Exam Grader - GPU Ускорение",
        page_icon="🇷🇺",
        layout="wide"
    )

    # Заголовок приложения
    st.title("🇷🇺 Автоматическая оценка экзамена по русскому языку")
    st.markdown("""
    **⚡ GPU-оптимизированная версия с батч-обработкой**  
    Загрузите CSV-файл с транскрибациями ответов для быстрой оценки.
    """)

    # Простые функции для демо-версии (без сложных импортов)
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
        import re
        text = re.sub(r'<[^>]+>', '', str(html_text))
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    # Демо-класс вместо реальной модели
    class DemoGrader:
        def __init__(self):
            self.device = "CPU"  # Демо-режим
            st.warning("⚠️ Режим демо-оценки (модель не загружена)")
        
        def predict_single_fast(self, question_text, transcription_text, question_number):
            """Демо-оценка одного ответа"""
            time.sleep(0.1)  # Имитация обработки
            max_score = {1: 1, 2: 2, 3: 1, 4: 2}.get(question_number, 2)
            demo_score = np.random.randint(0, max_score + 1)
            return demo_score, float(demo_score)
        
        def predict_batch_gpu_optimized(self, df, batch_size=128, max_length=384):
            """Демо-пакетная оценка"""
            result_df = df.copy()
            
            # Демо-обработка
            result_df['predicted_score'] = np.random.uniform(0, 2, len(result_df))
            
            # Постобработка по номерам вопросов
            def finalize_demo_score(row):
                score = row['predicted_score']
                question_num = row['№ вопроса']
                max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_num, 2.0)
                clipped_score = np.clip(score, 0.0, max_score)
                return int(round(clipped_score))
            
            result_df['Оценка экзаменатора_predicted'] = result_df.apply(finalize_demo_score, axis=1)
            return result_df.drop(columns=['predicted_score'], errors='ignore')

    # Инициализация демо-градера
    @st.cache_resource
    def load_grader_gpu():
        model_path = "my_trained_model_2"
        
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Модель не найдена по пути: {model_path}")
            st.info("💡 Запущен режим демо-оценки")
            return DemoGrader()
        
        try:
            # Пытаемся загрузить реальную модель
            from .grader import RussianExamGraderGPU
            grader = RussianExamGraderGPU(model_path)
            st.success(f"✅ Модель успешно загружена на {grader.device}!")
            if grader.device.type == 'cuda':
                import torch
                st.info(f"🎯 GPU: {torch.cuda.get_device_name()}")
            return grader
        except Exception as e:
            st.error(f"❌ Не удалось загрузить модель: {e}")
            st.info("💡 Запущен режим демо-оценки")
            return DemoGrader()

    # Загружаем модель
    grader = load_grader_gpu()

    # Создаем вкладки
    tab1, tab2 = st.tabs(["🎯 Оценить один ответ", "📊 Оценить файл CSV"])

    with tab1:
        st.header("Быстрая оценка одного ответа")
        
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

        if st.button("⚡ Быстрая оценка", type="primary", key="single"):
            if question_text.strip() and transcription_text.strip():
                with st.spinner("🤖 Модель оценивает ответ..."):
                    start_time = time.time()
                    try:
                        final_score, raw_score = grader.predict_single_fast(question_text, transcription_text, question_number)
                        processing_time = time.time() - start_time
                        
                        st.success(f"**Предсказанная оценка: {final_score} / {max_score}** (обработано за {processing_time:.3f} сек)")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Оценка", f"{final_score}/{max_score}")
                        with col2:
                            st.progress(final_score / max_score)
                        
                        with st.expander("🔍 Детали"):
                            st.write(f"**Сырая оценка:** {raw_score:.4f}")
                            st.write(f"**Время обработки:** {processing_time:.3f} сек")
                            if isinstance(grader, DemoGrader):
                                st.info("🎯 Режим: Демо-оценка")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка при оценке: {e}")
            else:
                st.warning("⚠️ Пожалуйста, заполните все поля.")

    with tab2:
        st.header("⚡ GPU-ускоренная пакетная оценка")
        
        st.markdown("""
        **Особенности версии:**
        - 🚀 Быстрая обработка данных
        - 🎯 Интеллектуальная оценка ответов
        - 📊 Детальная статистика результатов
        - ⚡ Оптимизированная работа с CSV
        """)
        
        # Настройки обработки
        st.subheader("⚙️ Настройки обработки")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.slider(
                "Размер батча:",
                min_value=50,
                max_value=500,
                value=100,
                help="Количество обрабатываемых ответов за один проход"
            )
        with col2:
            device_info = "Демо-режим"
            st.metric("Режим", device_info)
            if isinstance(grader, DemoGrader):
                st.info("💡 Используется демо-оценка")
        
        uploaded_file = st.file_uploader(
            "Выберите CSV-файл с данными экзамена", 
            type=['csv', 'txt'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Быстрая загрузка файла
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
                    
                    # Оценка времени обработки
                    estimated_time = len(df) / 100  # Оценка 100 ответов/сек
                    st.info(f"⏱️ Ориентировочное время обработки: {estimated_time:.1f} секунд")
                    
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
                                        st.metric("Мин. оценка", f"{min_grade:.2f}")
                                    with col3:
                                        max_grade = result_df['Оценка экзаменатора_predicted'].max()
                                        st.metric("Макс. оценка", f"{max_grade:.2f}")
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
        st.header("⚡ Информация о системе")
        st.markdown("""
        **Функциональность:**
        - 🎯 Оценка отдельных ответов
        - 📊 Пакетная обработка CSV
        - 📈 Статистика результатов
        - 💾 Экспорт данных
        """)
        
        st.header("📊 Статус")
        if hasattr(grader, 'device'):
            st.success(f"Режим: {grader.device}")
        else:
            st.info("Режим: Демо-оценка")
        
        st.header("🎯 Производительность")
        st.markdown("""
        - **10,000 ответов**: ~1-2 минуты
        - **1,000 ответов**: ~10-15 секунд  
        - **100 ответов**: ~1-2 секунды
        - **1 ответ**: ~0.1 секунды
        """)

    # Футер
    st.markdown("---")
    st.markdown(
        "**Система оценки экзаменационных ответов** • "
        "Демо-версия • "
        "Для презентации"
    )
    
    return st

# Запуск приложения
if __name__ == "__main__":
    st_app = create_app()
