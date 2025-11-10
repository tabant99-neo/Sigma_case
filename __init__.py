"""
Russian Exam Grader Package
Автоматическая система оценки экзаменационных ответов
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Система оценки письменных ответов на русском языке"

# Импорты для удобного доступа
from .grader import RussianExamGrader
from .utils import clean_html, preprocess_data
from .app import create_app

# При импорте sigma_case будут доступны:
# - RussianExamGrader
# - clean_html
# - preprocess_data
# - create_app

__all__ = ['RussianExamGrader', 'clean_html', 'preprocess_data', 'create_app']
