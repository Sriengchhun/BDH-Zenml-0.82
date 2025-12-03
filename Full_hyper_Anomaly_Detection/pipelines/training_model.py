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

docker_settings = DockerSettings(required_integrations=[BENTOML])
    
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def Training_pipeline_for_Anomaly_Classification(
    table_name: str,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_samples: Union[int, float, str] = "auto",
    contamination: Union[float, str] = "auto",
    max_features: Union[int, float] = 1.0,
    bootstrap: bool = False,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
):
    """Train IsolationForest & deploy."""

    X_train, X_test, y_train, y_test = training_data_loader(table_name, test_size)
    
    trained_model, training_acc = train_model(
        X_train=X_train,
        y_train=y_train,   # unused but required by ZenML signatures
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )
    
    test_accuracy = evaluator(X_test=X_test, y_test=y_test, model=trained_model)
    decision = deployment_trigger(accuracy=test_accuracy, min_accuracy=0.60)
    bento = bento_builder(model=trained_model)
    bentoml_model_deployer(bento=bento, deploy_decision=decision)