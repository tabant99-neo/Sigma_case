import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

def clean_html(html_text):
    """Быстрая очистка HTML"""
    if pd.isna(html_text): 
        return ""
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'–\s*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def normalize_score(score_series):
    return score_series.astype(float)

def preprocess_data_fast(df):
    """Оптимизированная предобработка"""
    df_copy = df.copy()
    
    # Быстрое удаление пустых строк
    mask = ~(df_copy['Текст вопроса'].isna() | df_copy['Транскрибация ответа'].isna())
    if 'Оценка экзаменатора' in df_copy.columns:
        mask &= ~df_copy['Оценка экзаменатора'].isna()
    
    df_copy = df_copy[mask].copy()
    
    # Векторизованная очистка HTML
    df_copy['Текст_вопроса_очищенный'] = df_copy['Текст вопроса'].apply(clean_html)
    
    # Быстрое формирование Input_Text
    df_copy['Input_Text'] = "ЗАДАНИЕ: " + df_copy['Текст_вопроса_очищенный'] + \
                           " | ДИАЛОГ: " + df_copy['Транскрибация ответа']
    
    if 'Оценка экзаменатора' in df_copy.columns and not df_copy['Оценка экзаменатора'].isnull().all():
        df_copy['labels'] = normalize_score(df_copy['Оценка экзаменатора'])
    else:
        df_copy['labels'] = np.nan
        
    return df_copy

def finalize_score_vectorized(scores, question_numbers):
    """Векторизованная постобработка оценок"""
    max_scores = np.array([{1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0}.get(q, 2.0) for q in question_numbers])
    clipped_scores = np.clip(scores, 0.0, max_scores)
    final_scores = np.round(clipped_scores).astype(int)
    return np.clip(final_scores, 0, max_scores.astype(int))

def safe_read_csv(uploaded_file):
    """Безопасное чтение CSV"""
    encodings = ['utf-8', 'cp1251', 'windows-1251', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            for sep in [',', ';', '\t']:
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
