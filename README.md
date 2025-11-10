# Russian Exam Grader

Автоматическая система оценки экзаменационных ответов на русском языке.

## Особенности

- Использует дообученную модель DeepPavlov/rubert-base-cased
- Полный пайплайн предобработки данных
- Оценка транскрибаций ответов
- MAE: 0.26

## Локальный запуск

```bash
pip install -r requirements.txt
streamlit run app.py
