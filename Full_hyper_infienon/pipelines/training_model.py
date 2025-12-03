from steps.bento_builder import bento_builder
from steps.deployer import bentoml_model_deployer
from steps.deployment_trigger_step import (
    deployment_trigger,
)
from steps.evaluators import evaluator
from steps.importers import training_data_loader
from steps.trainers import train_model

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML
from typing import Optional, Union
import os
import joblib







docker_settings = DockerSettings(required_integrations=[BENTOML])
    
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def Training_model_pipeline_Gesture(
    table_name: str,
    limit_num: int,
    test_size: float = 0.2,
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
    subsample: Optional[float] = 1.0,
    criterion: Optional[str] = "gini",
    criterion_gb: Optional[str] = "squared_error",
    # loss_gb: Optional[str] = 'log_loss'    
):
    """Train a model and deploy it with BentoML."""
    X_train, X_test, y_train, y_test = training_data_loader(table_name, limit_num, test_size)
    
    trained_model, training_acc = train_model(
        X_train=X_train,
        y_train=y_train,
        selected_model=selected_model,
        max_depth_dtc=max_depth_dtc,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features_dtc=max_features_dtc,

        max_depth_rf=max_depth_rf,
        n_estimators_rf=n_estimators_rf,
        max_features_rf=max_features_rf,

        n_estimators_gb=n_estimators_gb,
        max_depth_gb=max_depth_gb,
        learning_rate=learning_rate,
        subsample=subsample,
        criterion=criterion,
        criterion_gb=criterion_gb,  
        # loss_gb=loss_gb
        
    )
    
    test_accuracy = evaluator(X_test=X_test, y_test=y_test, model=trained_model)
    decision = deployment_trigger(accuracy=test_accuracy, min_accuracy=0.60)
    bento = bento_builder(model=trained_model)
    bentoml_model_deployer(bento=bento, deploy_decision=decision)


# if __name__ == "__main__":
#     config = "deploy_and_predict"
#     test_size = 0.2
#     selected_model = "Decision_Tree"
#     table_name = "infineon_gesture"
#     limit_num = 25000
#     max_depth_dtc = None
#     max_depth_rf = None
#     n_estimators_rf = 150
#     max_depth_gb = 1
#     learning_rate = 0.05
    
#     criterion = "gini"

#     Training_model_pipeline_Gesture(
#         test_size=test_size,
#         selected_model=selected_model,
#         table_name=table_name,
#         limit_num=limit_num,
#         max_depth_dtc=max_depth_dtc,
#         max_depth_rf=max_depth_rf,
#         n_estimators_rf=n_estimators_rf,
#         max_depth_gb=max_depth_gb,
#         learning_rate=learning_rate,
#         criterion=criterion
#     )


