import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

def load_data():
    # Загружаем датасет с pandas
    dataset = fetch_openml(name="house_prices", as_frame=True)
    df = dataset.frame  # полный DataFrame
    X_df = df.iloc[:, :10]  # первые 10 признаков
    y = df[dataset.target.name].values.reshape(-1,1)

    # Заполняем пропуски
    X_df = X_df.fillna(0)
    y = np.nan_to_num(y)  # на всякий случай

    # Разделяем на числовые и категориальные
    X_num = X_df.select_dtypes(include=[np.number])
    X_cat = X_df.select_dtypes(exclude=[np.number])

    # Приводим все категориальные к строкам и кодируем
    if not X_cat.empty:
        X_cat_str = X_cat.astype(str)  # вот ключ: все в строки
        enc = OrdinalEncoder()
        X_cat_enc = enc.fit_transform(X_cat_str)
        X = np.hstack([X_num.values, X_cat_enc])
    else:
        X = X_num.values

    # Нормализация
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # В тензоры
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return TensorDataset(X_tensor, y_tensor)

# Тест при запуске файла
if __name__ == "__main__":
    dataset = load_data()
    X, y = dataset.tensors
    print("Data shape:", X.shape, y.shape)
    print("First row:", X[0], y[0])