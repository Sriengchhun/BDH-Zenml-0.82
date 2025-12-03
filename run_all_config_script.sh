#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Run pip install
pip install -r requirements.txt

# Optional: show success message
echo "✅ Packages installed successfully."

# Define the script directory
SCRIPT_DIR="/app/script_file"

# Run each script
source "$SCRIPT_DIR/setup.sh"
source "$SCRIPT_DIR/fix_bentoml_deployer.sh"
source "$SCRIPT_DIR/fix_bentoml_deployment.sh"
source "$SCRIPT_DIR/fix_bentoml_model_deployer.sh"
source "$SCRIPT_DIR/fix_deploy_multiple_model_and_zenml_list.sh"

echo "✅ All scripts sourced successfully."
