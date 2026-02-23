import torch
from model import SimpleNet


def test_output_shape():
    torch.manual_seed(42)

    model = SimpleNet()
    x = torch.randn(4, 10)
    y = model(x)

    assert y.shape == (4, 1)

def test_deterministic_forward():
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

