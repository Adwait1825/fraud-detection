from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_baseline(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",

        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("===Baseline Logistic Regression ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:" ,  roc_auc_score(y_test, y_proba))

    return y_pred, y_proba


from src.data_loader import load_data
from src.baseline import train_baseline, evaluate_baseline  

X_train, X_test, y_train, y_test = load_data()      
model=train_baseline(X_train, y_train)
evaluate_baseline(model, X_test, y_test)    