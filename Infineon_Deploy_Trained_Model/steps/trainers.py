from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from typing import Optional
from zenml import step
import joblib
import time
from datetime import datetime
import os
from constants import MODEL_NAME    
import joblib


@step(enable_cache=False)
def load_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[
    Annotated[Optional[ClassifierMixin], "trained_model"],
    Annotated[Optional[float], "training_acc"],
]:

    """Load a pre-trained model from a specific directory and evaluate its training accuracy."""
    
    # Define the directory where models are stored
    model_dir = "/app/Store_Trained_Model"
    
    # Append .pkl if not already included in the model_name
    if not model_name.endswith('.pkl'):
        model_name = f'{model_name}.pkl'
    
    # Construct the full file path
    model_path = os.path.join(model_dir, model_name)
    
    print(f"Model path: {model_path}")
    
    try:
        # Load the pre-trained model
        loaded_model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    # Calculate training accuracy
    train_acc = loaded_model.score(X_train.to_numpy(), y_train.to_numpy())
    
    # Print model details for debugging (optional)
    print(f'Loaded model: {loaded_model}')
    if hasattr(loaded_model, 'feature_importances_'):
        print(f'Feature importances: {loaded_model.feature_importances_}')
    
    print(f"Train Accuracy: {train_acc}")
    print('*' * 100)
    print(f'The model has been loaded successfully. Here is the trained model: {loaded_model}')
    print('*' * 100)

    return loaded_model, train_acc