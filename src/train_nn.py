from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader  
from src.data_loader import load_data
from src.model import FraudNet
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def prepare_tensors(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def train_nn(
    X_train_tensor,
    y_train_tensor,
    input_dim,
    epochs=10,
    batch_size=256,
    lr=1e-3,
):

    model = FraudNet(input_dim)
    n_pos = y_train_tensor.sum()
    n_neg = len(y_train_tensor) - n_pos
    pos_weight = (n_neg / n_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
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


def evaluate_nn(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        y_true = y_test_tensor.numpy()
        y_pred = preds.numpy()
        y_proba = probs.numpy()

        print("=== Neural Network Evaluation ===")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))
        print("ROC-AUC" , roc_auc_score(y_true, y_proba))


        return y_pred, y_proba