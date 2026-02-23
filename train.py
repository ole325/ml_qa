import torch
from torch.utils.data import DataLoader
from model import SimpleNet
from data import load_data
import torch.nn as nn

# Загружаем данные
dataset = load_data()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Создаём модель
model = SimpleNet(input_size=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Обучение
for epoch in range(50):
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")