#!/usr/bin/env bash

CONSTANT_FILE="/opt/venv/lib64/python3.11/site-packages/zenml/constants.py"


sed -i 's/DEFAULT_LOCAL_SERVICE_IP_ADDRESS = "127.0.0.1"/DEFAULT_LOCAL_SERVICE_IP_ADDRESS = "0.0.0.0"/' $CONSTANT_FILE
# exec "$@"
echo "✅ Successfully completed the update of file: $CONSTANT_FILE"

CONSTANT_FILE_CONFIG="/opt/venv/lib64/python3.11/site-packages/bentoml/_internal/configuration/v1/default_configuration.yaml"

sed -i '/^  http:/,/^    response:/c\
  http:\
    host: 0.0.0.0\
    port: 3000\
    cors:\
      enabled: true\
      access_control_allow_origins: "*\"\
      access_control_allow_credentials: true\
      access_control_allow_methods: ["GET", "OPTIONS", "POST", "HEAD", "PUT"]\
      access_control_allow_headers: null\
      access_control_allow_origin_regex: null\
      access_control_max_age: null\
      access_control_expose_headers: ["Content-Length"]\
    response:\
      trace_id: false' "$CONSTANT_FILE_CONFIG"

# Remove duplicate occurrences of 'trace_id: false'
awk '!($0 == "      trace_id: false" && seen++) { print }' "$CONSTANT_FILE_CONFIG" > "${CONSTANT_FILE_CONFIG}.tmp"

# Replace the original file with the updated one
mv "${CONSTANT_FILE_CONFIG}.tmp" "$CONSTANT_FILE_CONFIG"
echo "✅ Successfully completed the update of file: $CONSTANT_FILE_CONFIG"
exec "$@"

mkdir ../artifact

zenml model-deployer register bentoml_deployer --flavor=bentoml
zenml artifact-store register local_hpe -f local --path=/artifact/
zenml stack register local_dev -o default -a local_hpe -d bentoml_deployer --set
zenml init

# rm -rf ./Full_hyper_activity/.zen ./Full_hyper_covid/.zen ./Full_hyper_infienon/.zen ./Object_detection/.zen ./Pytorch_classification/.zen
# rm -rf ./Activity_Deploy_Trained_Model/.zen ./Covid19_Deploy_Trained_Model/.zen ./Infineon_Deploy_Trained_Model/.zen ./Object_detection_upload/.zen

cp -r .zen ./Full_hyper_activity 
cp -r .zen ./Full_hyper_covid 
cp -r .zen ./Full_hyper_infienon 

cp -r .zen ./Activity_Deploy_Trained_Model
cp -r .zen ./Covid19_Deploy_Trained_Model
cp -r .zen ./Infineon_Deploy_Trained_Model


## run set up: ./setup.sh

cd /app/Full_hyper_infienon && zenml stack set local_dev
cd /app/Full_hyper_activity && zenml stack set local_dev
cd /app/Full_hyper_covid && zenml stack set local_dev