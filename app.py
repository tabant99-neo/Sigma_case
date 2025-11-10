import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Russian Exam Grader", page_icon="🇷🇺")
st.title("🇷🇺 Russian Exam Grader - Demo")
st.info("Демо-версия для тестирования")

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузите CSV файл", type=['csv'])

if uploaded_file is not None:
    try:
        # Пробуем разные кодировки и разделители
        success = False
        for encoding in ['utf-8', 'cp1251', 'windows-1251']:
            for sep in [';', ',', '\t']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                    if len(df.columns) > 1:
                        st.success(f"✅ Файл загружен! {len(df)} строк, {len(df.columns)} колонок")
                        st.write("📊 Предпросмотр данных:")
                        st.dataframe(df.head())
                        success = True
                        break
                except:
                    continue
            if success:
                break
        
        if not success:
            # Последняя попытка
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            st.success(f"✅ Файл загружен (с пропуском ошибок)! {len(df)} строк")
            st.dataframe(df.head())
        
        # Демо-обработка
        if st.button("🎯 Запустить демо-оценку"):
            st.info("🤖 Имитация работы модели...")
            
            # Создаем демо-оценки
            demo_df = df.copy()
            demo_df['predicted_grade'] = np.random.choice([0, 1, 2], size=len(demo_df))
            
            st.success("✅ Демо-оценка завершена!")
            st.write("📈 Результаты:")
            
            # Показываем только основные колонки + оценка
            display_cols = []
            for col in ['№ вопроса', 'Текст вопроса', 'Транскрибация ответа', 'predicted_grade']:
                if col in demo_df.columns:
                    display_cols.append(col)
            
            if not display_cols:
                display_cols = demo_df.columns[:3].tolist() + ['predicted_grade']
            
            st.dataframe(demo_df[display_cols].head(10))
            
            # Статистика
            st.write("📊 Статистика:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Средняя оценка", f"{demo_df['predicted_grade'].mean():.2f}")
            with col2:
                st.metric("Всего ответов", len(demo_df))
            with col3:
                st.metric("Режим", "Демо")
            
            # Скачивание
            csv = demo_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Скачать демо-результаты",
                csv,
                "demo_graded_results.csv",
                "text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        st.info("💡 Попробуйте сохранить файл в UTF-8 с разделителем ';'")

st.markdown("---")
st.markdown("*Для полной функциональности с моделью требуется установка дополнительных зависимостей*")
