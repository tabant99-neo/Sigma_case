"""
Russian Exam Grader Package
Автоматическая система оценки экзаменационных ответов
"""

__version__ = "1.0.0"
__author__ = "Sigma Case"
__description__ = "GPU-ускоренная система оценки письменных ответов на русском языке"

# Импорты для удобного доступа
from .grader import RussianExamGraderGPU
from .utils import clean_html, preprocess_data_fast, safe_read_csv, finalize_score_vectorized

__all__ = [
    'RussianExamGraderGPU', 
    'clean_html', 
    'preprocess_data_fast', 
    'safe_read_csv',
    'finalize_score_vectorized'
]
