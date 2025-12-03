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
import joblib   
import os
from constants import MODEL_NAME
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



def save_model_to_storage(trained_model, model_filename):
    storage_path = "/app/Full_hyper_infienon/my_model"  # Adjust this path to your desired storage location
    os.makedirs(storage_path, exist_ok=True)
    model_filepath = os.path.join(storage_path, model_filename)
    joblib.dump(trained_model, model_filepath)
    return model_filepath


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

     # Predict on training and test data
    # y_train_pred = model.predict(X_train.to_numpy())
    # y_test_pred = model.predict(X_test.to_numpy())

    print(f"-"*50)
    print("Trained Model Hyperparameters:")
    print(model.get_params())
    print(f"Train Accuracy - {selected_model}: {train_acc}")
    print(f"-"*50)
    
    # model_filename = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    model_filename = f"{MODEL_NAME}.pkl"
    model_filepath = save_model_to_storage(model, model_filename)
    print(f"Model saved to {model_filepath}")    
    print(f"============== model = {model} ===================")


    return model, train_acc