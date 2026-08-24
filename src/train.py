from src.data_loader import load_data
from src.baseline import train_baseline, evaluate_baseline
from src.model import FraudNet
from src.train_nn import prepare_tensors, train_nn, evaluate_nn

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("=== Baseline Logistic Regression ===")
    baseline_model = train_baseline(X_train, y_train)
    evaluate_baseline(baseline_model, X_test, y_test)

    print("\n=== Training Neural Network ===")
    X_train_t, X_test_t, y_train_t, y_test_t = prepare_tensors(X_train, X_test, y_train, y_test)
    nn_model = train_nn(X_train_t, y_train_t, input_dim=X_train_t.shape[1])

    print("\n=== Evaluating Neural Network ===")
    evaluate_nn(nn_model, X_test_t, y_test_t)

   