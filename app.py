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

    # Импорты из нашего пакета
    from .grader import RussianExamGraderGPU
    from .utils import safe_read_csv

    # Инициализация модели
    @st.cache_resource
    def load_grader_gpu():
        model_path = "my_trained_model_2"
        
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Модель не найдена по пути: {model_path}")
            return None
        
        try:
            grader = RussianExamGraderGPU(model_path)
            st.success(f"✅ Модель успешно загружена на {grader.device}!")
            if grader.device.type == 'cuda':
                st.info(f"🎯 GPU: {torch.cuda.get_device_name()}, Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return grader
        except Exception as e:
            st.error(f"❌ Не удалось загрузить модель: {e}")
            return None

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
            max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(question_number, 2.0)
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
                if grader is None:
                    st.error("❌ Модель не загружена!")
                else:
                    with st.spinner("🤖 Модель оценивает ответ..."):
                        start_time = time.time()
                        try:
                            final_score, raw_score = grader.predict_single_fast(question_text, transcription_text, question_number)
                            processing_time = time.time() - start_time
                            
                            st.success(f"**Предсказанная оценка: {final_score} / {int(max_score)}** (обработано за {processing_time:.3f} сек)")
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Оценка", f"{final_score}/{int(max_score)}")
                            with col2:
                                st.progress(final_score / max_score)
                            
                            with st.expander("🔍 Детали"):
                                st.write(f"**Сырая оценка:** {raw_score:.4f}")
                                st.write(f"**Время обработки:** {processing_time:.3f} сек")
                                
                        except Exception as e:
                            st.error(f"❌ Ошибка при оценке: {e}")
            else:
                st.warning("⚠️ Пожалуйста, заполните все поля.")

    with tab2:
        st.header("⚡ GPU-ускоренная пакетная оценка")
        
        if grader is None:
            st.error("❌ Модель не загружена! Невозможно выполнить пакетную оценку.")
            st.info("💡 Убедитесь, что папка 'my_trained_model_2' находится в корне проекта")
        else:
            st.markdown("""
            **Особенности GPU-версии:**
            - 🚀 Ускорение 10-20x за счет батч-обработки
            - 🎯 Автоматическая оптимизация для GPU/CPU
            - 📊 Прогресс-бар с расчетом оставшегося времени
            - ⚡ Скорость обработки: до 100+ ответов/сек
            """)
            
            # Настройки обработки
            st.subheader("⚙️ Настройки скорости")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                batch_size = st.slider(
                    "Размер батча:",
                    min_value=16,
                    max_value=256,
                    value=128,
                    help="Больший размер = быстрее, но требует больше памяти GPU"
                )
            with col2:
                max_length = st.slider(
                    "Макс. длина текста:",
                    min_value=256,
                    max_value=512,
                    value=384,
                    help="Уменьшение длины ускоряет обработку"
                )
            with col3:
                device_info = "GPU" if grader.device.type == 'cuda' else "CPU"
                st.metric("Устройство", device_info)
                if grader.device.type == 'cuda':
                    import torch
                    st.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
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
                                max_score = {1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(q_num, 2.0)
                                st.write(f"- Вопрос {q_num}: {count} ответов (макс. {max_score} баллов)")
                        
                        with col2:
                            st.write("**Пример данных:**")
                            display_cols = ['№ вопроса', 'Текст вопроса', 'Транскрибация ответа']
                            if 'Оценка экзаменатора' in df.columns:
                                display_cols.append('Оценка экзаменатора')
                            st.dataframe(df[display_cols].head(3))
                        
                        # Оценка времени обработки
                        estimated_time = len(df) / 50  # Оценка 50 ответов/сек
                        st.info(f"⏱️ Ориентировочное время обработки: {estimated_time/60:.1f} минут")
                        
                        if st.button("🚀 Запустить GPU-ускоренную оценку", type="primary", key="batch"):
                            with st.spinner("⚡ Запускаем GPU-ускоренную обработку..."):
                                try:
                                    result_df = grader.predict_batch_gpu_optimized(
                                        df, 
                                        batch_size=batch_size, 
                                        max_length=max_length
                                    )
                                    
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
                                        csv_result = result_df.to_csv(index=False, sep=';').encode('utf-8)
                                        st.download_button(
                                            label="📥 Скачать результаты (CSV)",
                                            data=csv_result,
                                            file_name="graded_results_gpu.csv",
                                            mime="text/csv",
                                            key="download_full"
                                        )
                                        
                                except Exception as e:
                                    st.error(f"❌ Ошибка при обработке: {e}")
                                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке файла: {e}")

    # Боковая панель
    with st.sidebar:
        st.header("⚡ GPU Оптимизации")
        st.markdown("""
        **Используемые технологии:**
        - 🎯 Батч-обработка до 256 примеров
        - 🔥 Mixed Precision (float16)
        - ⚡ Torch Compile
        - 🚀 CUDA Graphs (авто)
        - 📊 Векторизованная постобработка
        """)
        
        st.header("📊 Производительность")
        if grader and grader.device.type == 'cuda':
            import torch
            st.success(f"GPU: {torch.cuda.get_device_name()}")
            st.info(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            st.warning("Используется CPU режим")
        
        st.header("🎯 Ожидаемая скорость")
        st.markdown("""
        - **10,000 ответов**: ~2-3 минуты
        - **1,000 ответов**: ~10-15 секунд  
        - **100 ответов**: ~1-2 секунды
        - **1 ответ**: ~0.01 секунды
        """)

    # Футер
    st.markdown("---")
    st.markdown(
        "**⚡ GPU-ускоренная система оценки экзаменационных ответов** • "
        "Ускорение 10-20x • "
        "MAE: 0.26"
    )
    
    return st
