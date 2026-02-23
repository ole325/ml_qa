import torch
from model import SimpleNet
from data import load_data

# Загружаем данные
dataset = load_data()
X, y = dataset.tensors

# Берём несколько примеров
sample_X = X[:5]

# Создаём модель
model = SimpleNet(input_size=X.shape[1])

# Прогоняем через модель
with torch.no_grad():  # отключаем градиенты, чтобы просто проверить forward
    pred = model(sample_X)

print("Input shape:", sample_X.shape)
print("Output shape:", pred.shape)
print("Predictions:", pred)