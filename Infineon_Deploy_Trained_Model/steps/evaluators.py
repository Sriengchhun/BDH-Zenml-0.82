import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin
from zenml import step


@step
def evaluator(
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model: ClassifierMixin
) -> float:
    """Evaluates the scikit-learn model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc
