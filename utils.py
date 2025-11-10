import pandas as pd
import numpy as np
import re

def clean_html(html_text):
    """Очистка HTML разметки"""
    if pd.isna(html_text): 
        return ""
    text = re.sub(r'<[^>]+>', '', str(html_text))
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def preprocess_data(df):
    """Предобработка данных"""
    df_copy = df.copy()
    
    # Удаляем пустые строки
    mask = ~(df_copy['Текст вопроса'].isna() | df_copy['Транскрибация ответа'].isna())
    df_copy = df_copy[mask].copy()
    
    # Очистка HTML
    df_copy['Текст_вопроса_очищенный'] = df_copy['Текст вопроса'].apply(clean_html)
    
    # Формируем Input_Text
    df_copy['Input_Text'] = "ЗАДАНИЕ: " + df_copy['Текст_вопроса_очищенный'] + \
                           " | ДИАЛОГ: " + df_copy['Транскрибация ответа']
    
    return df_copy

def safe_read_csv(uploaded_file):
    """Безопасное чтение CSV"""
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            for sep in [';', ',', '\t']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                    if len(df.columns) > 0:
                        return df
                except:
                    continue
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # Последние попытки
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
    except:
        pass
    
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
    except:
        raise ValueError("Не удалось прочитать файл")
