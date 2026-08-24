import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc_comparison(y_test, baseline_proba, nn_proba, save_path="results/roc_comparison.png"):
    fpr_base, tpr_base, _ = roc_curve(y_test, baseline_proba)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_base, tpr_base, label="Logistic Regression (baseline)")
    plt.plot(fpr_nn, tpr_nn, label="Neural Network")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Baseline vs Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved ROC comparison plot to {save_path}")
