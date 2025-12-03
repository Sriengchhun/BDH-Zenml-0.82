from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Optional, Union
from zenml import step
import logging
from datetime import datetime
import pytz
from sklearn.ensemble import IsolationForest
import numpy as np
import os


@step(enable_cache=False)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    selected_model: Optional[str] = "Decision_Tree",
    n_estimators: int = 100,
    max_samples: Union[int, float, str] = "auto",
    contamination: Union[float, str] = "auto",   # e.g., 0.05 or "auto"
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0

) -> Tuple[
    Annotated[BaseEstimator, "trained_model"],
    Annotated[float, "training_acc"],
]:
    """
    Train an IsolationForest for anomaly detection.
    """
    X_np = X_train.to_numpy()

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    logging.info("Fitting IsolationForest...")
    model.fit(X_np)

    # Predict on train to report an interpretable metric (not accuracy):
    # +1 = normal/inlier, -1 = anomaly/outlier
    train_pred = model.predict(X_np)
    inlier_rate = float(np.mean(train_pred == 1))

    print("-" * 50)
    print("Trained IsolationForest Hyperparameters:")
    print(model.get_params())
    print(f"Train inlier rate (fraction predicted NORMAL on train): {inlier_rate:.4f}")
    print("-" * 50)

    # Timestamp (Asia/Bangkok as in your original code)
    local_time = datetime.now()
    local_time_utc = local_time.astimezone(pytz.utc)
    thailand_timezone = pytz.timezone('Asia/Bangkok')
    thailand_time = local_time_utc.astimezone(thailand_timezone)
    thailand_time_formatted = thailand_time.strftime("%Y-%m-%d-%H-%M-%S")
    logging.info("IsolationForest has been trained successfully.")
    print(f" Time stamp = {thailand_time_formatted}")

    os.environ["MODEL_NAME_Anomaly"] = model_name

    # Return model and "training_acc" placeholder as inlier rate
    train_acc = inlier_rate
    return model, train_acc