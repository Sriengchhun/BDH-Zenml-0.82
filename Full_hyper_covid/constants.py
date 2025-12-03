"""Decision Tree Model constants."""
import time
from datetime import datetime

# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d")

# Update the MODEL_NAME with the timestamp

MODEL_NAME = f"trained_model_covid19_{timestamp}"
SERVICE_NAME = f"Swagger_API_Service_{timestamp}"
PIPELINE_STEP_NAME = "bentoml_model_deployer_step"
PIPELINE_NAME = "training_pipeline_covide_19"