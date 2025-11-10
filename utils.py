import pandas as pd
import re
import os

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
    text = re.sub(r'<[^>]+>', '', str(html_text))
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def get_model_path():
    """Возвращает абсолютный путь к папке с моделью"""
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "my_trained_model_2")  # ← ИЗМЕНЕНО
    return model_path

def check_model_files(model_path):
    """Проверяем наличие всех файлов модели"""
    required_files = [
        'config.json', 
        'tokenizer_config.json',
        'vocab.txt',
        'special_tokens_map.json'
    ]
    
    # Проверяем наличие весов модели в любом формате
    model_weight_files = [
        'model.safetensors',  # Новый формат
        'pytorch_model.bin',  # Старый формат
        'tf_model.h5'         # TensorFlow формат
    ]
    
    existing_files = []
    missing_files = []
    
    # Проверяем основные файлы
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    # Проверяем веса модели
    model_weights_found = False
    for weight_file in model_weight_files:
        file_path = os.path.join(model_path, weight_file)
        if os.path.exists(file_path):
            existing_files.append(weight_file)
            model_weights_found = True
            break
    
    if not model_weights_found:
        missing_files.append('model_weights (любой формат)')
    
    return existing_files, missing_files
