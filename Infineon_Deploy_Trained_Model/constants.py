"""Decision Tree Model constants."""
import time
from datetime import datetime

# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d")

# Update the MODEL_NAME with the timestamp


MODEL_NAME = f"Upload_Trained_Infineon_model{timestamp}"
SERVICE_NAME = f"Upload_Trained_Infineon_model{timestamp}"
PIPELINE_STEP_NAME = "Upload_Trained_Infineon_model"
PIPELINE_NAME = "Upload_Trained_Infineon_model"