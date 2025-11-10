import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import re
import streamlit as st  # Добавляем импорт Streamlit

class RussianExamGrader:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text):
        """
        Базовая очистка текста. Можете добавить больше правил из вашего ноутбука.
        """
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def predict(self, text):
        """
        Предсказание оценки для одного текста.
        """
        processed_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            max_length=512,  # Используйте то значение, которое использовали при обучении
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits.cpu().numpy()

        # Предполагаем, что это задача регрессии и модель выдает одно число.
        # Если у вас классификация, адаптируйте эту часть.
        grade = float(prediction[0][0])
        
        # Можете добавить пост-обработку (например, обрезку от 0 до 5)
        grade = max(0, min(5, grade))
        return round(grade, 2)

# Функция для обработки CSV файла (с интеграцией Streamlit)
def grade_csv_file(csv_path, model_path, output_path='graded_output.csv'):
    """Обработка CSV файла с ответами"""
    try:
        grader = RussianExamGrader(model_path)
        
        # Читаем CSV
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Проверяем наличие столбца с ответами
        if 'answer' not in df.columns:
            st.error(f"Столбец 'answer' не найден. Найдены столбцы: {list(df.columns)}")
            return None
        
        answers = df['answer'].astype(str).tolist()
        
        # Создаем прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        grades = []
        total_answers = len(answers)
        
        for i, answer in enumerate(answers):
            grade = grader.predict(answer)
            grades.append(grade)
            
            # Обновляем прогресс-бар и статус
            progress = (i + 1) / total_answers
            progress_bar.progress(progress)
            status_text.text(f"Обработано: {i+1}/{total_answers} ответов")
        
        # Завершаем прогресс
        progress_bar.empty()
        status_text.empty()
        
        df['predicted_grade'] = grades
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        st.success(f"Оценка завершена! Обработано {total_answers} ответов.")
        return df
        
    except Exception as e:
        st.error(f"Ошибка при обработке CSV: {e}")
        return None

# Альтернативная функция без Streamlit (для использования вне приложения)
def grade_csv_file_simple(csv_path, model_path, output_path='graded_output.csv'):
    """Упрощенная версия без Streamlit"""
    try:
        grader = RussianExamGrader(model_path)
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        if 'answer' not in df.columns:
            print(f"Столбец 'answer' не найден. Найдены столбцы: {list(df.columns)}")
            return None
        
        answers = df['answer'].astype(str).tolist()
        
        grades = []
        for i, answer in enumerate(answers):
            grade = grader.predict(answer)
            grades.append(grade)
            if i % 100 == 0:  # Прогресс каждые 100 записей
                print(f"Обработано: {i}/{len(answers)}")
        
        df['predicted_grade'] = grades
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Оценка завершена! Обработано {len(answers)} ответов.")
        return df
        
    except Exception as e:
        print(f"Ошибка при обработке CSV: {e}")
        return None

# Эта часть будет использоваться в Streamlit
if __name__ == "__main__":
    # Для тестирования скрипта
    grader = RussianExamGrader('my_trained_model_2')
    test_text = "Моё любимое хобби - это читать книги и гулять с друзьями."
    print(f"Текст: {test_text}")
    print(f"Оценка: {grader.predict(test_text)}")
