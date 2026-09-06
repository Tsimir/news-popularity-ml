import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_and_prepare_data(data_path='data/OnlineNewsPopularity.csv', random_state=42):
    """
    Загружает датасет, удаляет лишние колонки, отсекает выбросы и масштабирует признаки.
    Возвращает:
        X_scaled: масштабированные признаки
        y_reg: целевая для регрессии (shares)
        y_clf: целевая для классификации (is_popular, 70-й перцентиль)
    """
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    columns_to_drop = ['url', 'timedelta']
    df = df.drop(columns=columns_to_drop)
    
    threshold = 100000
    df = df[df['shares'] <= threshold]
    
    # Признаки и целевая для регрессии
    X = df.drop('shares', axis=1)
    y_reg = df['shares']
    
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Целевая для классификации
    perc_70 = np.percentile(df['shares'], 70)
    y_clf = (df['shares'] > perc_70).astype(int)
    
    return X_scaled, y_reg, y_clf, df