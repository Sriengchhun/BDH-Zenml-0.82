from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from typing import Optional, Union
from zenml import step
import logging
from datetime import datetime
import pytz

@step(enable_cache=False)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_model: Optional[str] = "Decision_Tree",
    max_depth_dtc: Optional[int] = None,
    min_samples_split: Optional[Union[int, float]] = 2,
    min_samples_leaf: Optional[Union[int, float]] = 1,
    max_features_dtc: Optional[Union[int, float, str]] = None,

    max_depth_rf: Optional[int] = None,
    n_estimators_rf: Optional[int] = 100,
    max_features_rf: Optional[Union[int, float, str]] = "sqrt",

    n_estimators_gb: Optional[int] = 100,
    max_depth_gb: Optional[int] = 3,
    learning_rate: Optional[float] = 0.1,
    min_weight_fraction_leaf: Optional[float] = 0.0,
    max_features: Optional[Union[int, float, str]] = "sqrt",
    bootstrap: Optional[bool] = True,
    oob_score: Optional[bool] = False,
    subsample: Optional[float] = 1.0,
    # loss_gb: Optional[str] = 'log_loss',
    random_state: Optional[int] = None,
    criterion: Optional[str] = 'gini',
    criterion_gb: Optional[str] = 'squared_error'
) -> Tuple[
    Annotated[ClassifierMixin, "trained_model"],
    Annotated[float, "training_acc"],
]:
    """Train a specified model among Decision Tree, RandomForest, and GradientBoosting."""
    
    if selected_model == "Decision_Tree":
        model = DecisionTreeClassifier(max_depth=max_depth_dtc, 
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, 
                                       max_features=max_features_dtc, 
                                       criterion = criterion ,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, random_state=random_state)
        # model = DecisionTreeClassifier()
        print("Decision Tree")
    elif selected_model == "Random_Forest":
        model = RandomForestClassifier(n_estimators=n_estimators_rf, 
                                       max_depth=max_depth_rf,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features_rf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                       bootstrap=bootstrap, oob_score=oob_score, criterion = criterion, random_state=random_state)
        print("Random Forest")
    elif selected_model == "Gradient_Boosting":
        model = GradientBoostingClassifier(
                                        n_estimators=n_estimators_gb,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth_gb, 
                                        # loss=loss_gb,  
                                        subsample=subsample,
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
                                           criterion = criterion_gb, 
                                           random_state=random_state)
        print("Gradient Boosting")
    else:
        raise ValueError("Invalid selected model. Please choose from 'Decision_Tree', 'Random_Forest', or 'Gradient_Boosting'")
    
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    train_acc = model.score(X_train.to_numpy(), y_train.to_numpy())

    print(f"-"*50)
    print("Trained Model Hyperparameters:")
    print(model.get_params())
    print(f"Train Accuracy - {selected_model}: {train_acc}")
    print(f"-"*50)
    

    local_time = datetime.now()
    local_time_utc = local_time.astimezone(pytz.utc)
    thailand_timezone = pytz.timezone('Asia/Bangkok')
    thailand_time = local_time_utc.astimezone(thailand_timezone)
    thailand_time_formatted = thailand_time.strftime("%Y-%m-%d-%H-%M-%S")
    logging.info("Model has been trained successfully.. ")
    print(f' Time stamp = {thailand_time_formatted}')

    return model, train_acc