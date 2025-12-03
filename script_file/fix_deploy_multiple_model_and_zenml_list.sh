#!/usr/bin/env bash

# Define the path to the file
MODEL_DEPLOYER_FILE="/opt/venv/lib64/python3.11/site-packages/zenml/model_deployers/base_model_deployer.py"

# 1. Update the default value of continuous_deployment_mode to True
sed -i 's/continuous_deployment_mode: bool = False/continuous_deployment_mode: bool = True/' "$MODEL_DEPLOYER_FILE"

# 2. Comment out the call to self.delete_model_server
sed -i 's/^\(\s*\)self\.delete_model_server(service\.uuid)/\1# self.delete_model_server(service.uuid)/' "$MODEL_DEPLOYER_FILE"

# Confirmation message
echo "✅ Successfully updated default value of 'continuous_deployment_mode' in: $MODEL_DEPLOYER_FILE"


#### Modify Zenml Columns 

# Define the target file
UTILS_FILE="/opt/venv/lib64/python3.11/site-packages/zenml/cli/utils.py"
TMP_FILE="${UTILS_FILE}.tmp"

# Check and insert the import if it's not already in the file
if ! grep -q "^from urllib.parse import urlparse" "$UTILS_FILE"; then
    awk '
    BEGIN { inserted=0 }
    /^import / && !inserted {
        print
        next
    }
    /^from / && !inserted {
        print "from urllib.parse import urlparse"
        inserted=1
    }
    { print }
    ' "$UTILS_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$UTILS_FILE"
fi


awk '
BEGIN { in_func=0 }

/^def pretty_print_model_deployer[[:space:]]*\(/ {
    in_func=1
    print "def pretty_print_model_deployer("
    print "    model_services: List[\"BaseService\"], model_deployer: \"BaseModelDeployer\""
    print ") -> None:"
    print "    \"\"\"Given a list of served_models, print all associated key-value pairs."
    print ""
    print "    Args:"
    print "        model_services: list of model deployment services"
    print "        model_deployer: Active model deployer"
    print "    \"\"\""
    print "    model_service_dicts = []"
    print "    for model_service in model_services:"
    print "        dict_uuid = str(model_service.uuid)"
    print "        dict_pl_name = model_service.config.pipeline_name"
    print "        dict_pl_stp_name = model_service.config.pipeline_step_name"
    print "        dict_model_name = model_service.config.model_name"
    print "        type = model_service.SERVICE_TYPE.type"
    print "        flavor = model_service.SERVICE_TYPE.flavor"
    print "        model_url = model_service.get_prediction_url()"
    print "        port = urlparse(model_url).port"
    print "        model_service_dicts.append("
    print "            {"
    print "                \"STATUS\": get_service_state_emoji(model_service.status.state),"
    print "                \"UUID\": dict_uuid,"
    print "                \"PORT\": port,"
    print "                \"MODEL_NAME\": dict_model_name,"
    print "                \"PIPELINE_NAME\": dict_pl_name,"
    print "                \"PIPELINE_STEP_NAME\": dict_pl_stp_name,"
    print "                \"TYPE\": type,"
    print "                \"FLAVOR\": flavor,"
    print "            }"
    print "        )"
    print "    print_table("
    print "        model_service_dicts, UUID=table.Column(header=\"UUID\", min_width=36)"
    print "    )"
    print ""
    print ""
    next
}

in_func {
    # Skip lines inside original function until another function is found or EOF
    if (/^def[[:space:]]+/ && !/^def pretty_print_model_deployer/) {
        in_func=0
        print
    }
    next
}

{ print }
' "$UTILS_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$UTILS_FILE"

echo "✅ Rewrite complete: 'pretty_print_model_deployer' function updated in $UTILS_FILE"
# Execute the container's CMD 
exec "$@"