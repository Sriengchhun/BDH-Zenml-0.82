from steps.bento_builder import bento_builder
from steps.deployer import bentoml_model_deployer
from steps.deployment_trigger_step import (
    deployment_trigger,
)
from steps.evaluators import evaluator
from steps.importers import training_data_loader
from steps.trainers import load_model

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML
from typing import Optional
import logging


docker_settings = DockerSettings(required_integrations=[BENTOML])
    
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def Deploy_Trained_Infineon_model_pipeline(test_size: float = 0.2, model_name: Optional[str] = None ):
    """Load the Trained a model and deploy it with BentoML."""
    X_train, X_test, y_train, y_test = training_data_loader(test_size)
    logging.info(f"Start load the model.") 
    # Load the pre-trained model
    trained_model, training_acc = load_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name
    )
    if trained_model is None:
        logging.error("Failed to load the model.")
        print("Failed to load the model. Exiting pipeline.")
        return
    
    logging.info("Loaded the model successfully.") 
    test_accuracy = evaluator(X_test=X_test, y_test=y_test, model=trained_model)
    decision = deployment_trigger(accuracy=test_accuracy, min_accuracy=0.60)
    bento = bento_builder(model=trained_model)
    
    print(f"Trained model is == : {trained_model}")
    bentoml_model_deployer(bento=bento, deploy_decision=decision)


