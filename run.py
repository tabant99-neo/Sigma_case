#!/usr/bin/env python3
"""
Точка входа для запуска приложения
"""

import sys
import os

# Добавляем текущую директорию в Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sigma_case.app import create_app

if __name__ == "__main__":
    # Запускаем приложение
    st = create_app()
    
    # Дополнительная информация в sidebar
    with st.sidebar:
        st.header("ℹ️ О системе")
        st.markdown("""
        **Russian Exam Grader**
        - Версия: 1.0.0
        - Автор: Your Name
        - Описание: Система оценки экзаменационных ответов
        """)
