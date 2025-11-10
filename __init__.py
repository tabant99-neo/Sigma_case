"""
Sigma Case - Automatic Russian Exam Grading System
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Automatic grading system for Russian language exams using ML"

from .app import main
from .grader import RussianExamGrader
from .utils import clean_html_simple, safe_read_csv

__all__ = ['main', 'RussianExamGrader', 'clean_html_simple', 'safe_read_csv']
