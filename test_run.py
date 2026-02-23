import torch
import pytest
from model import SimpleNet
from data import load_data

@pytest.mark.e2e
def test_model_forward():
    # Загружаем данные
    dataset = load_data()
    X, y = dataset.tensors

    # Берём несколько примеров
    sample_X = X[:5]

    # Создаём модель
    model = SimpleNet(input_size=X.shape[1])

    # Прогоняем через модель
    with torch.no_grad():  # отключаем градиенты
        pred = model(sample_X)

    # Проверки для pytest
    assert pred.shape[0] == sample_X.shape[0], "Количество выходов не совпадает с количеством входов"
    assert pred.shape[1] == model.out_features, "Выходное размерность не совпадает с размерностью модели"

    # Дополнительно можно просто вывести результаты
    print("Input shape:", sample_X.shape)
    print("Output shape:", pred.shape)
    print("Predictions:", pred)