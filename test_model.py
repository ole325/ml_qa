import torch
import pytest
from model import SimpleNet  # ваша модель


# ================= Junior Level =================

def test_output_shape():
    """Проверяем форму выхода модели"""
    torch.manual_seed(42)
    model = SimpleNet()
    x = torch.randn(4, 10)
    y = model(x)
    assert y.shape == (4, 1)


def test_deterministic_forward():
    """Проверяем детерминированность при фиксированном seed"""
    torch.manual_seed(42)
    model1 = SimpleNet()
    x1 = torch.randn(1, 10)
    y1 = model1(x1)

    torch.manual_seed(42)
    model2 = SimpleNet()
    x2 = torch.randn(1, 10)
    y2 = model2(x2)

    assert torch.allclose(y1, y2)


def test_training_reduces_loss():
    """Проверяем, что один шаг обучения уменьшает loss"""
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    x = torch.randn(20, 10)
    y_true = torch.randn(20, 1)

    # начальный loss
    y_pred = model(x)
    loss_before = criterion(y_pred, y_true)

    # один шаг обучения
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    # новый loss
    y_pred_after = model(x)
    loss_after = criterion(y_pred_after, y_true)

    assert loss_after.item() < loss_before.item()


def test_no_nan_inf():
    """Проверяем, что на случайных данных нет NaN или Inf"""
    torch.manual_seed(42)
    model = SimpleNet()
    x = torch.randn(10, 10) * 1000  # большие значения для проверки stability
    y = model(x)
    assert torch.isfinite(y).all()


def test_output_range():
    """Простейшая sanity check — выход в разумных пределах"""
    torch.manual_seed(42)
    model = SimpleNet()
    x = torch.randn(10, 10)
    y = model(x)
    # допустим ожидаем, что значения выхода < 1e3
    assert (y.abs() < 1e3).all()


# ================= Middle Level =================

def test_loss_decreases_sqrt():
    """Проверка на нестандартных арифметических зависимостях (sqrt)"""
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    x = torch.randn(10, 10)
    y_true = torch.sqrt(torch.abs(torch.randn(10, 1)))  # sqrt данных

    loss_before = criterion(model(x), y_true)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    loss_after = criterion(model(x), y_true)

    assert loss_after.item() < loss_before.item()


def test_loss_decreases_square():
    """Проверка на квадратичные зависимости"""
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    x = torch.randn(10, 10)
    y_true = torch.randn(10, 1) ** 2  # квадратные данные

    loss_before = criterion(model(x), y_true)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    loss_after = criterion(model(x), y_true)

    assert loss_after.item() < loss_before.item()


def test_batch_consistency():
    """Проверяем одинаковые батчи → одинаковый loss"""
    torch.manual_seed(42)
    model = SimpleNet()
    x = torch.randn(5, 10)
    y_true = torch.randn(5, 1)
    criterion = torch.nn.MSELoss()

    loss1 = criterion(model(x), y_true)
    loss2 = criterion(model(x), y_true)
    assert torch.allclose(loss1, loss2)


def test_small_batch_training():
    """Проверка, что маленький батч тоже уменьшает loss"""
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    x = torch.randn(2, 10)
    y_true = torch.randn(2, 1)

    loss_before = criterion(model(x), y_true)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    loss_after = criterion(model(x), y_true)

    assert loss_after.item() < loss_before.item()