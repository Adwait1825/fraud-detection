from src.data_loader import load_data
from src.baseline import train_baseline, evaluate_baseline  

X_train, X_test, y_train, y_test = load_data()      
model=train_baseline(X_train, y_train)
evaluate_baseline(model, X_test, y_test)    