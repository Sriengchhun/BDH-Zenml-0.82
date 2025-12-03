import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from zenml import step
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# @step
# def evaluator(
#     X_test: pd.DataFrame, 
#     y_test: pd.Series, 
#     model: ClassifierMixin
# ) -> float:
#     """Evaluates the scikit-learn model."""
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Test Accuracy: {acc * 100:.2f}%")
#     return acc

@step
def evaluator(
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model: BaseEstimator
) -> float:
    """Evaluator for IsolationForest anomaly detection."""
    
    # Predict: +1 = normal, -1 = anomaly
    y_pred = model.predict(X_test)

    # Convert to anomaly labels: 1 = anomaly, 0 = normal
    y_pred = (y_pred == -1).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    return acc

