from fastapi import FastAPI, HTTPException, Query
from subprocess import run, CalledProcessError
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum


app = FastAPI(
    title="BDH Zenml API",
    description="""Zenml API helps you do awesome of Training and Deploy the models. 🚀""",
    version="0.0.1",
    contact={
        "name": "BDH Zenml API Support",
        "email": "sriengchhunchheang@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

### -------------- CORS Middleware ---------------
origins = ["*"]
# Add the HTTPSRedirectMiddleware to automatically redirect HTTP requests to HTTPS
# app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define constants
zenml_executable = "zenml"
cwd_activity_classification = "/app/Activity_classification"
cwd_covid19 = "/app/Select_Dataset"
cwd_infineon = "/app/Select_Hyper_parameter_Infineon"   
full_hyperparameter = "/app/Full_hyper_infienon"
activity_parameter = "/app/Full_hyper_activity" 

class Criterion(str, Enum):
    gini = "gini"
    entropy = "entropy"

class Criterion_gb(str, Enum):
    squared_error = "squared_error"
    friedman_mse = "friedman_mse"

class Models(str, Enum):
    Decision_Tree = "Decision_Tree"
    Random_Forest = "Random_Forest"
    Gradient_Boosting = "Gradient_Boosting"


def run_zenml_command(cmd, cwd):
    try:
        result = run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
        output = result.stdout.splitlines()
        return {"success": True, "output": output}
    except CalledProcessError as e:
        return {"success": False, "message": e.stderr.splitlines()}

# Define route for starting ZenML
@app.get("/zenml-start", tags=['Start/Stop/Delete/List Model'])
def start_zenml(model_uuid: str):
    cmd = [zenml_executable, "model-deployer", "models", "start", model_uuid]
    return run_zenml_command(cmd, cwd_activity_classification)

# Define route for stopping ZenML
@app.get("/zenml-stop", tags=['Start/Stop/Delete/List Model'])
def stop_zenml(model_uuid: str):
    cmd = [zenml_executable, "model-deployer", "models", "stop", model_uuid]
    return run_zenml_command(cmd, cwd_activity_classification)

# Define route for deleting ZenML
@app.get("/zenml-delete", tags=['Start/Stop/Delete/List Model'])
def delete_zenml(model_uuid: str):
    cmd = [zenml_executable, "model-deployer", "models", "delete", model_uuid]
    return run_zenml_command(cmd, cwd_activity_classification)

# Define route for listing ZenML
@app.get("/zenml-list", tags=['Start/Stop/Delete/List Model'])
def list_zenml():
    cmd = [zenml_executable, "model-deployer", "models", "list"]
    return run_zenml_command(cmd, cwd_activity_classification)

# Define route for regular training model
@app.get("/zenml-train-v1", tags=['Apple Watch'])
def train_model(test_size: float, select_model: str, table_name: str, limit_num: int):
    run_script_cmd = ["python", "run.py", "--test_size", str(test_size), "--select_model", select_model, 
                      "--table_name", table_name, "--limit_num", str(limit_num)]
    return run_zenml_command(run_script_cmd, cwd_activity_classification)

@app.get("/zenml-train-v2", tags=['Apple Watch'])
def Train_model_for_Activity(
    test_size: float = Query(default=0.2, description="Test size"),
    select_model: str = Query(default="Decision_Tree", description="Selected model"),
    table_name: str = Query(default="applewatch", description="Table name"),
    limit_num: int = Query(default=2500, description="Number of rows for each labeled"),
):
    run_script_cmd = [
        "python", "run.py",
        "--test_size", str(test_size),
        "--select_model", select_model,
        "--table_name", table_name,
        "--limit_num", str(limit_num),
    ]
    return run_zenml_command(run_script_cmd, cwd_activity_classification)


@app.get("/zenml-train-activity", tags=['Apple Watch'])
def train_model_Hyperparameter_for_Activity(
    test_size: float = Query(default=0.2, description="Test size, type: float, default: 0.2"),
    select_model: Models = Query(default=Models.Decision_Tree, description="Selected model, type: str, default: 'Decision_Tree'"),
    table_name: str = Query(default="applewatch", description="Table name, type: str, default: 'applewatch'"),
    limit_num: int = Query(default=2500, description="Limit number of rows, type: int, default: 2500"),
    max_depth_dtc: str = Query(default=None, description="Max depth for Decision Tree, type: int or None, default: None"),
    min_samples_split: str = Query(default="2", description="Min samples split for Decision Tree, type: int or float, default: 2"),
    min_samples_leaf: str = Query(default="1", description="Min samples leaf for Decision Tree, type: int or float, default: 1"),
    max_features_dtc: str = Query(default=None, description="Max features for Decision Tree, type: int, float, str, or None, default: None"),
    criterion: Criterion = Query(default=Criterion.gini, description="Criterion for Decision Tree and Random Forest, type: str, default: 'gini'"),
    max_depth_rf: str = Query(default=None, description="Max depth for Random Forest, type: int or None, default: None"),
    n_estimators_rf: int = Query(default=100, description="Number of estimators for Random Forest, type: int, default: 100"),
    max_features_rf: str = Query(default="sqrt", description="Max features for Random Forest, type: int, float, str, or None, default: 'sqrt'"),
    n_estimators_gb: int = Query(default=100, description="Number of estimators for Gradient Boosting, type: int, default: 100"),
    max_depth_gb: str = Query(default="3", description="Max depth for Gradient Boosting, type: int, default: 3"),
    learning_rate: float = Query(default=0.1, description="Learning rate for Gradient Boosting, type: float, default: 0.1"),
    subsample: float = Query(default=1.0, description="Subsample for Gradient Boosting, type: float, default: 1.0"),
    criterion_gb: Criterion_gb = Query(default=Criterion_gb.squared_error, description="Criterion for Gradient Boosting, type: str, default: 'squared_error'")
):
    run_script_cmd = [
        "python", "run.py",
        "--test_size", str(test_size),
        "--select_model", select_model.value,
        "--table_name", table_name,
        "--limit_num", str(limit_num),
        "--max_depth_dtc", str(max_depth_dtc),
        "--min_samples_split", min_samples_split,
        "--min_samples_leaf", min_samples_leaf,
        "--max_features_dtc", str(max_features_dtc),
        "--criterion", criterion.value,
        "--max_depth_rf", str(max_depth_rf),
        "--n_estimators_rf", str(n_estimators_rf),
        "--max_features_rf", max_features_rf,
        "--n_estimators_gb", str(n_estimators_gb),
        "--max_depth_gb", str(max_depth_gb),
        "--learning_rate", str(learning_rate),
        "--subsample", str(subsample),
        "--criterion_gb", criterion_gb.value
    ]
    return run_zenml_command(run_script_cmd, activity_parameter)


# Define route for Covid-19 training model
@app.get("/zenml-train-covid19", tags=['Covid-19'])
def train_model_covid19(test_size: float, select_model: str, table_name: str, limit_num: int):
    run_script_cmd = ["python", "run.py", "--test_size", str(test_size), "--select_model", select_model, 
                      "--table_name", table_name, "--limit_num", str(limit_num)]
    return run_zenml_command(run_script_cmd, cwd_covid19)



@app.get("/zenml-train-infineon", tags=['Infineon'])
def train_model_Hyperparameter_for_Infineon(
    test_size: float = Query(default=0.2, description="Test size, type: float, default: 0.2"),
    select_model: Models = Query(default=Models.Decision_Tree, description="Selected model, type: str, default: 'Decision_Tree'"),
    table_name: str = Query(default="infineon_gesture", description="Table name, type: str, default: 'infineon_gesture'"),
    limit_num: int = Query(default=2500, description="Limit number of rows, type: int, default: 2500"),
    max_depth_dtc: str = Query(default=None, description="Max depth for Decision Tree, type: int or None, default: None"),
    min_samples_split: str = Query(default="2", description="Min samples split for Decision Tree, type: int or float, default: 2"),
    min_samples_leaf: str = Query(default="1", description="Min samples leaf for Decision Tree, type: int or float, default: 1"),
    max_features_dtc: str = Query(default=None, description="Max features for Decision Tree, type: int, float, str, or None, default: None"),
    criterion: Criterion = Query(default=Criterion.gini, description="Criterion for Decision Tree and Random Forest, type: str, default: 'gini'"),
    max_depth_rf: str = Query(default=None, description="Max depth for Random Forest, type: int or None, default: None"),
    n_estimators_rf: int = Query(default=100, description="Number of estimators for Random Forest, type: int, default: 100"),
    max_features_rf: str = Query(default="sqrt", description="Max features for Random Forest, type: int, float, str, or None, default: 'sqrt'"),
    n_estimators_gb: int = Query(default=100, description="Number of estimators for Gradient Boosting, type: int, default: 100"),
    max_depth_gb: str = Query(default="3", description="Max depth for Gradient Boosting, type: int, default: 3"),
    learning_rate: float = Query(default=0.1, description="Learning rate for Gradient Boosting, type: float, default: 0.1"),
    subsample: float = Query(default=1.0, description="Subsample for Gradient Boosting, type: float, default: 1.0"),
    criterion_gb: Criterion_gb = Query(default=Criterion_gb.squared_error, description="Criterion for Gradient Boosting, type: str, default: 'squared_error'")
):
    run_script_cmd = [
        "python", "run.py",
        "--test_size", str(test_size),
        "--select_model", select_model.value,
        "--table_name", table_name,
        "--limit_num", str(limit_num),
        "--max_depth_dtc", str(max_depth_dtc),
        "--min_samples_split", min_samples_split,
        "--min_samples_leaf", min_samples_leaf,
        "--max_features_dtc", str(max_features_dtc),
        "--criterion", criterion.value,
        "--max_depth_rf", str(max_depth_rf),
        "--n_estimators_rf", str(n_estimators_rf),
        "--max_features_rf", max_features_rf,
        "--n_estimators_gb", str(n_estimators_gb),
        "--max_depth_gb", str(max_depth_gb),
        "--learning_rate", str(learning_rate),
        "--subsample", str(subsample),
        "--criterion_gb", criterion_gb.value
    ]
    return run_zenml_command(run_script_cmd, full_hyperparameter)



# Define route for killing ports
@app.get("/Check-port", tags=['Check/Kill Port'])
def check_port():
    cmd = ['./checkport.sh', "3032", "3334", "8000", "8001", "8002", "8003", "8004", "8005", "8006", "8007", "8008", "8009"]
    return run_zenml_command(cmd, cwd_covid19) 

@app.get("/kill-port", tags=['Check/Kill Port'])
def kill_port():
    cmd = ["./kill_multiple_port.sh", "3032", "3334", "8000", "8001", "8002", "8003", "8004", "8005", "8006", "8007", "8008", "8009"]
    return run_zenml_command(cmd, cwd_covid19)


##uvicorn api_app:app --host 0.0.0.0 --port 8085 --reload