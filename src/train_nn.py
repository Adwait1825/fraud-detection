import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from src.data_loader import load_data
from src.model import FraudNet


def prepare_tensors(X_train, X_test, y_train, y_test):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    return X_train_t, X_test_t, y_train_t, y_test_t


def train_nn(X_train_t, y_train_t, input_dim, epochs=10, batch_size=256, lr=1e-3):
    model = FraudNet(input_dim)

    n_pos = y_train_t.sum()
    n_neg = len(y_train_t) - n_pos
    pos_weight = (n_neg / n_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model
