"""Decision Tree Model constants."""
import time
from datetime import datetime

# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d")

# Update the MODEL_NAME with the timestamp

MODEL_NAME = f"infineon_{timestamp}"
SERVICE_NAME = f"infineon_API_Service_{timestamp}"
PIPELINE_STEP_NAME = "bentoml_model_deployer_infineon_step"
PIPELINE_NAME = "infineon"