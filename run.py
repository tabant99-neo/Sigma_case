#!/usr/bin/env python3
"""
Точка входа для запуска Streamlit приложения
"""

import sys
import os

# Добавляем текущую директорию в Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sigma_case.app import create_app

if __name__ == "__main__":
    # Создаем и запускаем приложение
    st = create_app()
