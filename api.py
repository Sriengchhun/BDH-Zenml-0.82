import logging
import os
import json
import shutil
import threading
import sys
import base64
import uuid
import torch


from pprint import pprint
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, status
# from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from subprocess import run, CalledProcessError, Popen
from zenml.client import Client
from urllib.parse import urlparse
from pydantic import BaseModel
from enum import Enum
from typing import List, Union
from typing_extensions import Annotated
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from typing import Optional, Dict
from postgresqlDB import ConnectPostgresqlDB


''' Setting config variable and constants'''
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Specify class invocation
db = ConnectPostgresqlDB()


# Configure FastAPI
app = FastAPI(
    title="BDH Zenml 0.82 API",
    description="""Zenml API helps you do awesome of Training and Deploy the models. 🚀""",
    version="2.1.1",
    contact={
        "name": "BDH Zenml API Support (Time Series)",
        "email": "sriengchhunchheang@gmail.com",
        "name": "BDH Zenml API Support (Image)",
        "email": "k.surisa@vamstack.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

FAKE_USERS = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("x-brain@zenml"),
    }
}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    user = FAKE_USERS.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_token(data: Dict[str, str], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    token = create_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(401, "Invalid token")
        return username
    except JWTError:
        raise HTTPException(401, "Invalid token")

@app.get("/users/me")
def read_users_me(current_user: str = Depends(get_current_user)):
    return {"user": current_user}


### -------------- CORS Middleware ---------------
# origins = ["*"]
# Add the HTTPSRedirectMiddleware to automatically redirect HTTP requests to HTTPS
# app.add_middleware(HTTPSRedirectMiddleware)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

#####################################################################################################################
### parameter and constants
#####################################################################################################################

# Define constants
zenml_executable = "zenml"
cwd_main = "/app"
full_hyperparameter = "/app/Full_hyper_infineon"
activity_parameter = "/app/Full_hyper_activity" 
covid_19_parameter = "/app/Full_hyper_covid"
IoT_parameter = "/app/Full_hyper_IoT"
cwd_object_detection = "/app/Object_detection"
cwd_object_classification = "/app/Pytorch_classification"
Anomaly_parameter = "/app/Full_hyper_Anomaly_Detection"

detection_datasets = f"{cwd_object_detection}/datasets"
detection_train = f"{cwd_object_detection}/train"
classification_datasets = f"{cwd_object_classification}/datasets"
classification_train = f"{cwd_object_classification}/train"
yolo_pretrain = f"{cwd_object_detection}/pre-train"
symbolic_local = f"/data/local-files/?d=input"
bentoml_models =  "/root/bentoml/models"
bentoml_service = "/root/bentoml/bentos"


''' class of variable '''
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


class ActivityHyperParams(BaseModel):
    project_id: int = Query(..., description="Project ID for download datasets from Label Studio")
    user_id : str = Query(..., description="User ID")
    uuid : str = Query(..., description="uuid of model")
    model_name: str = Query(default="Activity")
    test_size: float = Query(default=0.2, description="Test size, type: float, default: 0.2")
    select_model: Models = Query(default=Models.Decision_Tree, description="Selected model, type: str, default: 'Decision_Tree'")
    table_name: str = Query(default="applewatch", description="Table name, type: str, default: 'applewatch'")
    limit_num: int = Query(default=2500, description="Limit number of rows, type: int, default: 2500")
    max_depth_dtc: str = Query(default="None", description="Max depth for Decision Tree, type: int or None, default: None")
    min_samples_split: str = Query(default="2", description="Min samples split for Decision Tree, type: int or float, default: 2")
    min_samples_leaf: str = Query(default="1", description="Min samples leaf for Decision Tree, type: int or float, default: 1")
    max_features_dtc: str = Query(default="None", description="Max features for Decision Tree, type: int, float, str, or None, default: None")
    criterion: Criterion = Query(default=Criterion.gini, description="Criterion for Decision Tree and Random Forest, type: str, default: 'gini'")
    max_depth_rf: str = Query(default="None", description="Max depth for Random Forest, type: int or None, default: None")
    n_estimators_rf: int = Query(default=100, description="Number of estimators for Random Forest, type: int, default: 100")
    max_features_rf: str = Query(default="sqrt", description="Max features for Random Forest, type: int, float, str, or None, default: 'sqrt'")
    n_estimators_gb: int = Query(default=100, description="Number of estimators for Gradient Boosting, type: int, default: 100")
    max_depth_gb: str = Query(default="3", description="Max depth for Gradient Boosting, type: int, default: 3")
    learning_rate: float = Query(default=0.1, description="Learning rate for Gradient Boosting, type: float, default: 0.1")
    subsample: float = Query(default=1.0, description="Subsample for Gradient Boosting, type: float, default: 1.0")
    criterion_gb: Criterion_gb = Query(default=Criterion_gb.squared_error, description="Criterion for Gradient Boosting, type: str, default: 'squared_error'") 


class Covid19HyperParams(BaseModel):
    project_id: int = Query(..., description="Project ID for download datasets from Label Studio")
    user_id : str = Query(..., description="User ID")
    uuid : str = Query(..., description="uuid of model")
    model_name: str = Query(default="Covid19")
    test_size: float = Query(default=0.2, description="Test size, type: float, default: 0.2")
    select_model: Models = Query(default=Models.Decision_Tree, description="Selected model, type: str, default: 'Decision_Tree'")
    table_name: str = Query(default="wesafe_ai", description="Table name, type: str, default: 'applewatch'")
    limit_num: int = Query(default=25000, description="Limit number of rows, type: int, default: 2500")
    max_depth_dtc: str = Query(default="None", description="Max depth for Decision Tree, type: int or None, default: None")
    min_samples_split: str = Query(default="2", description="Min samples split for Decision Tree, type: int or float, default: 2")
    min_samples_leaf: str = Query(default="1", description="Min samples leaf for Decision Tree, type: int or float, default: 1")
    max_features_dtc: str = Query(default="None", description="Max features for Decision Tree, type: int, float, str, or None, default: None")
    criterion: Criterion = Query(default=Criterion.gini, description="Criterion for Decision Tree and Random Forest, type: str, default: 'gini'")
    max_depth_rf: str = Query(default="None", description="Max depth for Random Forest, type: int or None, default: None")
    n_estimators_rf: int = Query(default=100, description="Number of estimators for Random Forest, type: int, default: 100")
    max_features_rf: str = Query(default="sqrt", description="Max features for Random Forest, type: int, float, str, or None, default: 'sqrt'")
    n_estimators_gb: int = Query(default=100, description="Number of estimators for Gradient Boosting, type: int, default: 100")
    max_depth_gb: str = Query(default="3", description="Max depth for Gradient Boosting, type: int, default: 3")
    learning_rate: float = Query(default=0.1, description="Learning rate for Gradient Boosting, type: float, default: 0.1")
    subsample: float = Query(default=1.0, description="Subsample for Gradient Boosting, type: float, default: 1.0")
    criterion_gb: Criterion_gb = Query(default=Criterion_gb.squared_error, description="Criterion for Gradient Boosting, type: str, default: 'squared_error'")


class InfineonHyperParams(BaseModel):
    project_id: int = Query(..., description="Project ID for download datasets from Label Studio")
    user_id : str = Query(..., description="User ID")
    uuid : str = Query(..., description="uuid of model")
    model_name: str = Query(default="Infineon")
    test_size: float = Query(default=0.2, description="Test size, type: float, default: 0.2")
    select_model: Models = Query(default=Models.Decision_Tree, description="Selected model, type: str, default: 'Decision_Tree'")
    table_name: str = Query(default="infineon_gesture", description="Table name, type: str, default: 'infineon_gesture'")
    limit_num: int = Query(default=2500, description="Limit number of rows, type: int, default: 2500")
    max_depth_dtc: str = Query(default="None", description="Max depth for Decision Tree, type: int or None, default: None")
    min_samples_split: str = Query(default="2", description="Min samples split for Decision Tree, type: int or float, default: 2")
    min_samples_leaf: str = Query(default="1", description="Min samples leaf for Decision Tree, type: int or float, default: 1")
    max_features_dtc: str = Query(default="None", description="Max features for Decision Tree, type: int, float, str, or None, default: None")
    criterion: Criterion = Query(default=Criterion.gini, description="Criterion for Decision Tree and Random Forest, type: str, default: 'gini'")
    max_depth_rf: str = Query(default="None", description="Max depth for Random Forest, type: int or None, default: None")
    n_estimators_rf: int = Query(default=100, description="Number of estimators for Random Forest, type: int, default: 100")
    max_features_rf: str = Query(default="sqrt", description="Max features for Random Forest, type: int, float, str, or None, default: 'sqrt'")
    n_estimators_gb: int = Query(default=100, description="Number of estimators for Gradient Boosting, type: int, default: 100")
    max_depth_gb: str = Query(default="3", description="Max depth for Gradient Boosting, type: int, default: 3")
    learning_rate: float = Query(default=0.1, description="Learning rate for Gradient Boosting, type: float, default: 0.1")
    subsample: float = Query(default=1.0, description="Subsample for Gradient Boosting, type: float, default: 1.0")
    criterion_gb: Criterion_gb = Query(default=Criterion_gb.squared_error, description="Criterion for Gradient Boosting, type: str, default: 'squared_error'")

class IoTHyperParams(BaseModel):
    project_id: int = Query(..., description="Project ID for download datasets from Label Studio")
    user_id : str = Query(..., description="User ID")
    uuid : str = Query(..., description="uuid of model")
    model_name: str = Query(default="IoT")
    test_size: float = Query(default=0.2, description="Test size, type: float, default: 0.2")
    select_model: Models = Query(default=Models.Decision_Tree, description="Selected model, type: str, default: 'Decision_Tree'")
    table_name: str = Query(default="nexpie_sensors", description="Table name, type: str, default: 'infineon_gesture'")
    max_depth_dtc: str = Query(default="None", description="Max depth for Decision Tree, type: int or None, default: None")
    min_samples_split: str = Query(default="2", description="Min samples split for Decision Tree, type: int or float, default: 2")
    min_samples_leaf: str = Query(default="1", description="Min samples leaf for Decision Tree, type: int or float, default: 1")
    max_features_dtc: str = Query(default="None", description="Max features for Decision Tree, type: int, float, str, or None, default: None")
    criterion: Criterion = Query(default=Criterion.gini, description="Criterion for Decision Tree and Random Forest, type: str, default: 'gini'")
    max_depth_rf: str = Query(default="None", description="Max depth for Random Forest, type: int or None, default: None")
    n_estimators_rf: int = Query(default=100, description="Number of estimators for Random Forest, type: int, default: 100")
    max_features_rf: str = Query(default="sqrt", description="Max features for Random Forest, type: int, float, str, or None, default: 'sqrt'")
    n_estimators_gb: int = Query(default=100, description="Number of estimators for Gradient Boosting, type: int, default: 100")
    max_depth_gb: str = Query(default="3", description="Max depth for Gradient Boosting, type: int, default: 3")
    learning_rate: float = Query(default=0.1, description="Learning rate for Gradient Boosting, type: float, default: 0.1")
    subsample: float = Query(default=1.0, description="Subsample for Gradient Boosting, type: float, default: 1.0")
    criterion_gb: Criterion_gb = Query(default=Criterion_gb.squared_error, description="Criterion for Gradient Boosting, type: str, default: 'squared_error'")

class AnomalyDetectionHyperParams(BaseModel):
    project_id: int = Query(..., description="Project ID for downloading datasets")
    user_id: str = Query(..., description="User ID")
    uuid: str = Query(..., description="uuid of model")
    model_name: str = Query(default="Anomaly", description="Model name")
    table_name: str = Query(default="iot_sensors", description="Table name in DB")
    test_size: float = Query(default=0.2, description="Test size, default=0.2")
    n_estimators: int = Query(default=100, description="Number of estimators (trees), default=100")
    max_samples: Union[int, float, str] = Query(default="auto", description="Max samples per tree, 'auto' or number/float")
    contamination: Union[float, str] = Query(default="auto", description="Contamination level, e.g. 0.05 or 'auto'")
    max_features: Union[int, float] = Query(default=1.0, description="Max features to use, default=1.0")
    # bootstrap: bool = Query(default=False, description="Whether bootstrap samples are used")
    n_jobs: Optional[int] = Query(default=-1, description="Number of parallel jobs, default=-1")
    random_state: Union[int, float, str] = Query(default="None", description="Random state for reproducibility, default=None")
    # verbose: int = Query(default=0, description="Verbose output level, default=0")


class DetectionParams(BaseModel):
    project_id: int = Query(..., description="Project ID for download datasets from Label Studio")
    img_ids: List[int] = Query(..., description="Image ID for download datasets from Label Studio")
    user_id : str = Query(..., description="User ID")
    authorization_token: str = Query(default="3f5d56d3caf780d2114fd9d4871744fd20efde45", description="Token of User, Project owner for download datasets from Label Studio")
    uuid : str = Query(..., description="uuid of model")
    imgsz: int = Query(default=640, description="train, val image size (pixels)")
    batch_size: int = Query(default=-1, description="total batch size for all GPUs, -1 for autobatch")
    epochs: int = Query(default=100, description="total training epochs")
    neural_network_architecture: str = Query(default="YOLOv5", description="type for train")
    weights: str = Query(default=f"yolov5s.pt", description="initial weights path")
    model_name: str = Query(default="Object_detection")


class InputClasses(BaseModel):
    sub_folderID: str
    name: str


class ClassificationParams(BaseModel):
    project_id: int = Query(..., description="Project ID")
    user_id : str = Query(..., description="User ID")
    uuid : str = Query(..., description="uuid of model")
    datasetes_path: str = Query(default="608175dae5cd71001119f3bd/1",description="Path for download datasets from X-brain")
    input_classes: List[InputClasses] = Query(default=[{"sub_folderID":"1", "name":"dog"}, {"sub_folderID":"2", "name":"cat"}], description="Class for download datasets from X-brain")
    batch_size: int = Query(default=4, description="total batch size for all GPUs")
    epochs: int = Query(default=100, description="total training epochs")
    learning_rate: float = Query(default=0.001, description="determines the step size at each iteration")
    neural_network_architecture: str = Query(default=f"ResNet50", description="neural network for training")
    weights: str = Query(default=f"IMAGENET1K_V2", description="initial weights path")
    model_name: str = Query(default="Pytorch_classification")


#####################################################################################################################
### start event 
#####################################################################################################################

def setup_zenml_stack():
    status = { "process": "text", "output": "default fn result" }

    '''
    ฟังก์ชันสำหรับตั้งค่า ZenML Stack & เปิด ZenML Dashboard
    1. คำสั่งเปิด ZenML Dashboard
    2. คำสั่งตรวจสอบรายการของ ZenML Stack
        2.1 ถ้ายังไม่มีการลงทะเบียน จะมีข้อมูลในรายการแค่ 8 บรรทัดที่เป็นค่าพื้นฐานของ ZenML ต้องทำการลงทะเบียนผ่าน Script set.sh
        2.2 ถ้าลงทะเบียนเรียบร้อย บังคับเรียกใช้งาน stack ที่ลงทะเบียนไว้ตลอด
    '''
    

    '''
    1. คำสั่งเปิด ZenML Dashboard
    '''
    # Popen(["uvicorn", "zenml.zen_server.zen_server_api:app", "--port", "8080", "--host", "0.0.0.0"])

    ''' 
    2. ใช้คำสั่งตรวจสอบรายการของ ZenML Stack
        2.1 ถ้ายังไม่มีการลงทะเบียน จะมีข้อมูลในรายการแค่ 8 บรรทัดที่เป็นค่าพื้นฐานของ ZenML ต้องทำการลงทะเบียนผ่าน Script set.sh
        2.2 ถ้าลงทะเบียนเรียบร้อย บังคับเรียกใช้งาน stack ที่ลงทะเบียนไว้ตลอด
    '''

    try:
        # ตรวจสอบและแสดงรายการของ ZenML Stack ที่ถูกใช้งาน
        cmd = ["zenml", "stack", "list"]
        result = run(cmd, capture_output=True, text=True, cwd='/app', check=True)
        output = result.stdout.splitlines()
        pprint(output)
 
        # กรณียังไม่มีการลงทะเบียน ตั้งเงื่อนไขเพื่อทำการลงทะเบียนผ่าน script set.sh
        if len(output) <= 8:
            os.system("sh set.sh")

            # ตรวจสอบรายการของ ZenML Stack ที่ถูกใช้งาน หลังทำการลงทะเบียน
            cmd = ["zenml", "stack", "list"]
            result = run(cmd, capture_output=True, text=True, cwd='/app', check=True)
            output = result.stdout.splitlines()

            # ระบุการทำงานและส่งรายการของ ZenML Stack ที่ถูกใช้งาน
            status["process"] = "Register ZenML stack"
            status["output"] = output
            return status
        
        # กรณีที่ลงทะเบียนแล้ว บังคับเรียกใช้ stack ที่ลงทะเบียนไว้ตลอด
        cmd = ["zenml", "stack", "set", "local_dev"]
        result = run(cmd, capture_output=True, text=True, cwd='/app', check=True)
        output = result.stdout.splitlines()

        # ระบุการทำงานและส่งรายการของ ZenML Stack ที่กำลังใช้งาน
        status["process"] = "Set ZenML stack"
        status["output"] = output
        return status

    except Exception as e:
        print("new")
        print(e)
        
        # ถ้าการทำงานไม่ตรงกับที่ตั้งไว้ ให้วน loop จนกว่าจะตั้งค่าสำเร็จ 
        status = setup_zenml_stack()
        return status


def stop_services():

    '''
    ฟังก์ชันสำหรับเปลี่ยนสถานะการทำงานของ model ที่อยู่ใน database
    1. คำสั่งเปิด ZenML Dashboard
    2. คำสั่งตรวจสอบรายการของ ZenML Stack
        2.1 ถ้ายังไม่มีการลงทะเบียน จะมีข้อมูลในรายการแค่ 8 บรรทัดที่เป็นค่าพื้นฐานของ ZenML ต้องทำการลงทะเบียนผ่าน Script set.sh
        2.2 ถ้าลงทะเบียนเรียบร้อย บังคับเรียกใช้งาน stack ที่ลงทะเบียนไว้ตลอด
    '''

    status = { "process": "Stop service", "output": "default fn result" }
    output = db.get_active_services()

    if output == []:
        status["output"] = "No services are running in the database." 

        return status       
    
    for data in output:
        model_id = data["model_id"]
        db.reset_service_status(model_id)

    status["output"] = "Stop services and update status_deplyer on database." 

    return status


def continuous_training():
    status = { "process": "Ongoing Training", "output": "default fn result" }
    output = db.inspect_queued_tasks()
    # print(output)

    if output == []:

        status["output"] = "No process are trainning in the database." 
        return status

    model_id = output["model_id"]
    model_name = output["uuid_name"]
    service_name = f"{model_name}_service"
    status_deployer = output["status_deployer"]
    run_name = output["run_name"]

    check_model_path = [
        f"{bentoml_models}/{model_name}",
        f"{bentoml_service}/{service_name}",
        f"{classification_train}/{str(model_id)}",
        f"{detection_train}/{str(model_id)}",
    ]

    check_datasets = [
        f"{detection_datasets}/{model_id}",
        f"{classification_datasets}/{model_id}"
    ]

    try:
        status_train = output["status"]["task_status"]
        if status_train == 0:
            status["output"] = f"model_id {model_id} start"
            train_and_deploy()

            return status
        
        if status_train == 1 and run_name != None:
            pipeline_run = Client().get_pipeline_run(run_name)
            for step_name, step_run in pipeline_run.steps.items():
                
                outputs = step_run.outputs
                for output_name, output in outputs.items():
                    artifact_uri = output.uri
                    if os.path.isdir(artifact_uri):
                        shutil.rmtree(artifact_uri)
                        print(f"Delete ::> Step: {step_name}, Output: {output_name}, URI: {artifact_uri}")
        
            for directory_path in check_model_path:
                if os.path.isdir(directory_path):
                    shutil.rmtree(directory_path)
                    print(f"Delete ::> Local: {directory_path}")

            for directory_path in check_datasets:
                if os.path.exists(directory_path):
                    for root, _, files in os.walk(directory_path):
                        for filename in files:
                            file_path = os.path.join(root, filename)

                            # เช็คว่าเป็น symbolic link และเป็นไฟล์ภาพหรือไม่
                            if os.path.islink(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                                os.unlink(file_path)
                                # print(f"Unlinked: {file_path}")

                    shutil.rmtree(directory_path)

            db.wait_in_queue(model_id=model_id)
            train_and_deploy()
            
        return status

    except KeyError:

        if status_deployer != None:
            db.queue_complete(model_id)

            status["output"] = f"Update queue complete on model_id: {model_id}"

            train_and_deploy()

            return status


        if not os.path.exists(check_model_path[0]) or not os.path.exists(check_model_path[1]):

            status["output"] = f"model_id: {model_id} not deploy"

            db.queue_error(model_id)
            train_and_deploy()

            return status


        # find model and service on local
        latest_version = model_version(model_name)
        latest_service = model_service(model_name)

        version = latest_version["output"]
        service_name = latest_service["output"]["service_name"]
        service_version = latest_service["output"]["service_version"]

        db.update_model(model_id=model_id, version=version, service_name=service_name, service_version=service_version)

        deployer_data = model_deployer_data(model_name=model_name)

        if deployer_data["success"] != True:
            status["output"] = deployer_data["output"]
            db.queue_error(model_id)

            train_and_deploy()

            return status
 
        pipeline_name = deployer_data["output"]["PIPELINE_NAME"]
        run_name = deployer_data["output"]["RUN_NAME"]
        uuid_deployer = deployer_data["output"]["UUID"]
        status_deployer = deployer_data["output"]["STATUS"]
        url = deployer_data["output"]["PREDICTION_URL"]
        parsed_url = urlparse(url)
        port = parsed_url.port

        db.update_deployer(model_id, pipeline_name, run_name, uuid_deployer, status_deployer, port)
        db.queue_complete(model_id)

        status["output"] = f"Update on model_id: {model_id}"
        train_and_deploy()

        return status



def check_process():

    ''' 
    ฟังก์ชันสำหรับเรียงลำดับการทำงาน และแสดงผลการทำงานที่ Log Terminal
    1. ตั้งค่า ZenML Stack
    2. ปรับค่าสถานะการทำงานของ Model ใน database
    3. ตรวจสอบการทำงานที่อยู่ใน database  
    '''

    status_stack = setup_zenml_stack()
    pprint(status_stack)

    status_service = stop_services()
    print(status_service)

    status_training = continuous_training()
    print(status_training)


# @app.on_event("startup")
# async def startup_event():

#     '''
#     ฟังก์ชันสำหรับเริ่มทำงานโดยอัตโนมัติ โดยจะทำงานอยู่เบื้องหลัง
#     '''

#     # สร้าง thread สำหรับ background task 
#     task_thread = threading.Thread(target=check_process)
#     task_thread.daemon = True  # ทำให้ thread หยุดเมื่อ main thread หยุด
#     task_thread.start()


#####################################################################################################################
### function 
#####################################################################################################################

''' commond function'''
# shortcut for FastAPI response with status code and JSON payload
def fastapi_response(status_code, content):
    return JSONResponse(status_code = status_code, content = content)


''' Internal functions '''
# script of zenml command
def run_zenml_command(cmd, cwd):
    status = { "success": False, "output": "default fn result" }
    try:

        result = run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
        output = result.stdout.splitlines()

        status["success"] = True
        status["output"] = output

    except CalledProcessError as e:
        status["success"] = False
        status["output"] = e.stderr.splitlines()
    
    return status


# script of model training and send data to DB
def run_zenml_model(cmd, cwd, model_id, model_name):
    status = { "success": False, "output": "default fn result" }

    try:

        # run scrip zenml training
        result = run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
        output = result.stdout.splitlines()
        
        # updata model data on local to DB
        status["success"] = True
        status["output"] = output

        db.update_status(model_id=model_id, status=json.dumps(status))
        

        # find model and service on local
        latest_version = model_version(model_name)
        latest_service = model_service(model_name)

        version = latest_version["output"]
        service_name = latest_service["output"]["service_name"]
        service_version = latest_service["output"]["service_version"]

        db.update_model(model_id=model_id, version=version, service_name=service_name, service_version=service_version)
        

        # find model-deployer data in zenml
        deployer_data = model_deployer_data(model_name=model_name)

        pipeline_name = deployer_data["output"]["PIPELINE_NAME"]
        run_name = deployer_data["output"]["RUN_NAME"]
        uuid_deployer = deployer_data["output"]["UUID"]
        status_deployer = deployer_data["output"]["STATUS"]
        url = deployer_data["output"]["PREDICTION_URL"]
        parsed_url = urlparse(url)
        port = parsed_url.port

        db.update_deployer(model_id, pipeline_name, run_name, uuid_deployer, status_deployer, port)
        db.queue_complete(model_id)

        train_and_deploy()

    
    except CalledProcessError as e:
        status["success"] = False
        status["output"] = e.stderr.splitlines()

        check_model_path = [
            f"{classification_train}/{str(model_id)}",
            f"{detection_train}/{str(model_id)}",
        ]

        check_datasets = [
            f"{detection_datasets}/{model_id}",
            f"{classification_datasets}/{model_id}"
        ]

        for directory_path in check_datasets:
            if os.path.exists(directory_path):
                for root, _, files in os.walk(directory_path):
                    for filename in files:
                        file_path = os.path.join(root, filename)

                        # เช็คว่าเป็น symbolic link และเป็นไฟล์ภาพหรือไม่
                        if os.path.islink(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                            os.unlink(file_path)
                            print(f"Unlinked: {file_path}")

                shutil.rmtree(directory_path)

        for directory_path in check_model_path:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)

        db.update_status(model_id=model_id, status=json.dumps(status))
        db.queue_error(model_id)

        train_and_deploy()
    
    return status


# find model version from bentoml on local
def model_version(model_name):
    status = { "success": False, "output": "default fn result" }

    try:
        model_directory_path = os.path.join(bentoml_models, model_name.lower())

        subdirectories = [
            d for d in os.listdir(model_directory_path) 
            if os.path.isdir(os.path.join(model_directory_path, d))
        ]
        latest_directory = max(
            subdirectories,
            key=lambda d: os.path.getmtime(os.path.join(model_directory_path, d)))

        status["success"] = True
        status["output"] = latest_directory
    
    except Exception as e:
        status["success"] = False
        status["output"] = type(e).__name__, str(e)

    return status


# find name and veresion of bentoml service on local
def model_service(model_name):
    status = { "success": False, "output": "default fn result" }

    try:
        service_name = f"{model_name.lower()}_service"
        service_directory_path = os.path.join(bentoml_service, service_name)

        subdirectories = [
            d for d in os.listdir(service_directory_path) 
            if os.path.isdir(os.path.join(service_directory_path, d))
        ]
        latest_directory = max(
            subdirectories,
            key=lambda d: os.path.getmtime(os.path.join(service_directory_path, d)))

        status["success"] = True
        status["output"] = {"service_name":service_name, "service_version": latest_directory}
        
    except Exception as e:
        status["success"] = False
        status["output"] = type(e).__name__, str(e)

    return status


# find model-deployer data via zenml
def model_deployer_data(model_name):
    status = { "success": False, "output": "default fn result" }

    try:
        model_deployer_list = list_zenml_json()
        # print(model_deployer_list)

        if model_deployer_list["success"] == False:
            return  model_deployer_list

        ''' This function will do searching model-deployer data '''

        for model in model_deployer_list["output"]:
            name = model["model_name"]
            uuid = model["uuid"]
            # print(name, model_name)

            if name == model_name:
                status = check_model_deployer_data(uuid)
                break
            else:
                status["success"] = False
                status["output"] = "No model in zenml model-deployer"

    except Exception as e:
        status["success"] = False
        status["output"] = type(e).__name__, str(e)
    
    return status


def check_model_deployer_data(uuid_deployer):
    status = { "success": False, "output": "default fn result" }
    
    try:
        # run zenml command
        cmd = [zenml_executable, "model-deployer", "models", "describe", uuid_deployer]
        data = run_zenml_command(cmd, cwd_main)
        # print(data)
        
        if data["success"] == False:
            return data
        
        data_deployer = parse_zenml_describe_output(data["output"])
        # print(data_deployer) 
        
        status["success"] = True
        status["output"] = data_deployer
    
    except Exception as e:
        status["success"] = False
        status["output"] = type(e).__name__, str(e)
    
    return status


# function support zenml_describe / convert data to json format
def parse_zenml_describe_output(output_lines):
    columns_data = []
    lines = []
    n_line = 0   

    ''' หาจุดแบ่งของบรรทัดที่ต้องการด้วยอักขระ ┠ '''
    # ตัดหัวข้อคอลัมน์ออก แต่เหลือเส้นแบ่ง row ไว้
    for line in output_lines[3:]:
        # # Debugging: Print the raw line
        # print(f"Raw line: {line}")

        if line == '┠────────────────────────┼─────────────────────────────────────────────────────┨':
            lines.append(n_line)

        n_line += 1
    
    ''' อ่านชุดข้อมูลที่ถูกแบ่ง โดยใช้บรรทัดที่เจอเส้นคั่นเป็นจุดเริ่มต้นและจบในการอ่าน '''
    for i in range(len(lines)):

        # อ้างอิงจากบรรทัดที่เจอเส้นแบ่งเป็นจุดเริ่มต้นและจบ เพื่ออ่านเฉพาะชุดข้อมูลที่ต้องการ 
        start = 3 + lines[i]
        end = 3 + lines[i + 1] if i + 1 < len(lines) else None

        # print(output_lines[start:end])
        for row_set in output_lines[start:end]:
            if '┃' in row_set:
                values = row_set.split('│')

                cleaned_values = [value.strip() for value in values[0:]]
                columns_data.append(cleaned_values)

    combined_data = {}
    current_key = None
    
    for row in columns_data:
        if len(row[0].strip()) > 1:  # ถ้ามีข้อมูลคีย์ (เช่น '┃ BENTO_TAG')
            current_key = row[0].strip('┃').strip()  # ลบอักขระพิเศษและช่องว่าง
            combined_data[current_key] = row[1].strip('┃').strip()  # เริ่มด้วยค่าของข้อมูล
        else:
            # ถ้าเป็นแถวต่อจากข้อมูลเดิม ให้เพิ่มค่าต่อจากข้อมูลเดิม
            combined_data[current_key] += row[1].strip('┃').strip()
    
    return combined_data


#####################################################################################################################
### manage zenml 
#####################################################################################################################

@app.get("/zenml-list-json", tags=['Start/Stop/Delete Model'])
def list_zenml_json():
    status = { "success": False, "output": "default fn result" }

    # run zenml command
    cmd = [zenml_executable, "model-deployer", "models", "list"]
    result = run_zenml_command(cmd, cwd_main)
    
    # check command output
    if result["success"] != True:
        status["success"] = result["success"]
        status["output"] = result["output"]

        return fastapi_response(500, status)


    parsed_data = parse_zenml_output(result["output"])

    status["success"] = True
    status["output"] = parsed_data
    
    return status


# function support zenml-list-json / convert data to json format
def parse_zenml_output(output_lines):
    parsed_data = []
    sets = []
    n_line = 0

    ''' หาจุดแบ่งชุดข้อมูล โดยอ้างอิงจากบรรทัดที่เจอ status '''
    # ตัดหัวข้อคอลัมน์ออก
    for line in output_lines[4:]:

        # print(f"Raw line: {line}")

        # Split the line by the '│' character
        columns = [col.strip() for col in line.split("│")]

        status = columns[0].strip().strip('┃').strip()

        if status == "⏸" or status == "✅":
            sets.append(n_line)

        n_line += 1

    ''' อ่านชุดข้อมูลที่ถูกแบ่ง โดยใช้บรรทัดที่เจอ status เป็นจุดเริ่มต้นและจบในการอ่าน '''
    for i in range(len(sets)):
        # เริ่มเก็บค่าใหม่ทุกครั้ง
        columns_data = []

        # อ้างอิงจากบรรทัดที่เจอ status เป็นจุดเริ่มต้นและจบ เพื่ออ่านเฉพาะชุดข้อมูลที่ต้องการ 
        start = 4 + sets[i]
        end = 4 + sets[i + 1] if i + 1 < len(sets) else None

        for row_set in output_lines[start:end]:
            if '┃' in row_set:
                values = row_set.split('│')

                cleaned_values = [value.strip() for value in values[0:]]
                columns_data.append(cleaned_values)

        ''' รวมข้อมูลและส่งกลับ '''
        # สร้างรายการที่มีขนาดเท่ากับจำนวนคอลัมน์
        combined_data = [''] * len(columns_data[0])

        # รวมข้อมูลในแต่ละคอลัมน์
        for row in columns_data:
            for i in range(len(row)):
                if row[i].strip() and (i != 0 and i != 4):  
                    combined_data[i] += row[i].strip()  
                elif i == 0: 
                    combined_data[i] += row[i].strip('┃').strip()
                elif i == 4:
                    combined_data[i] += row[i].strip('┃').strip()  
        
        parsed_data.append({
            "status": combined_data[0],
            "uuid": combined_data[1],
            "pipeline_name": combined_data[2],
            "pipeline_step_name": combined_data[3],
            "model_name": combined_data[4]
        })

    return parsed_data


# Define route for starting ZenML
@app.put("/zenml-start", tags=['Start/Stop/Delete Model'])
def start_zenml(model_id: int, current_user: str = Depends(get_current_user)):
    status = { "success": False, "output": "default response from service function" }

    # search uuid deployer by model_id
    data = db.get_uuid_deployer(model_id)

    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    uuid_deployer = data["uuid_deployer"]
    uuid_name = data["uuid_name"]
    category = data["category"]

    if  uuid_deployer == None:
        status["success"] = False
        status["output"] = 'model not deploy'

        return fastapi_response(404, status)
    
    # set model_name from category 
    if category == "activity":
        os.environ["MODEL_NAME_Activity"] = uuid_name

    elif category == "covid19":
        os.environ["MODEL_NAME_Covid19"] = uuid_name

    elif category == "infineon":
        os.environ["MODEL_NAME_Infineon"] = uuid_name
    
    elif category == "IoT":
        os.environ["MODEL_NAME_IoT"] = uuid_name

    elif category == "Anomaly":
        os.environ["MODEL_NAME_Anomaly"] = uuid_name
    
    elif category == "detection":
        os.environ["MODEL_NAME_Obj"] = uuid_name
    
    else:
        parameters = data["parameter"]
        class_name = sorted([c["name"] for c in parameters["input_classes"]])
        NeuralNetwork = data["parameter"]["neural_network_architecture"]
        
        os.environ["MODEL_NAME_Class"] = uuid_name
        os.environ["CLASS_NAMES"] = f"{class_name}"
        os.environ["NeuralNetwork"] = NeuralNetwork

    
    # run zenml command for start service model
    cmd = [zenml_executable, "model-deployer", "models", "start", uuid_deployer]
    zenml_result = run_zenml_command(cmd, cwd_main)

    if zenml_result["success"] != True:
        status["success"] = False
        status["output"] = zenml_result["output"]

        return fastapi_response(400, status) 


    status["success"] = True
    status["output"] = zenml_result["output"]

    # check status_deployer and port 
    data = check_model_deployer_data(uuid_deployer)

    if data["success"] != True:
        status["success"] = False
        status["output"] = data["output"]

        return fastapi_response(400, status) 

    # update data model-deployer
    pipeline_name = data["output"]["PIPELINE_NAME"]
    run_name = data["output"]["RUN_NAME"]
    uuid_deployer = data["output"]["UUID"]
    status_deployer = data["output"]["STATUS"]
    url = data["output"]["PREDICTION_URL"]
    parsed_url = urlparse(url)
    port = parsed_url.port 

    print(f"run_name: {run_name}, port: {port}, status: {status_deployer}, uuid: {uuid_deployer}, pipeline_name: {pipeline_name}")
    db.update_deployer(model_id, pipeline_name, run_name, uuid_deployer, status_deployer, port)
            
    return status   


# Define route for stopping ZenML
@app.put("/zenml-stop", tags=['Start/Stop/Delete Model'])
def stop_zenml(model_id: int, current_user: str = Depends(get_current_user)):
    status = { "success": False, "output": "default response from service function" }

    # search uuid deployer by model_id
    data = db.get_uuid_deployer(model_id)

    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    uuid_deployer = data["uuid_deployer"]

    if  uuid_deployer == None:
        status["success"] = False
        status["output"] = 'model not deploy'

        return fastapi_response(404, status)     
    
    # run zenml command for start service model
    cmd = [zenml_executable, "model-deployer", "models", "stop", uuid_deployer]
    zenml_result = run_zenml_command(cmd, cwd_main)

    if zenml_result["success"] != True:
        status["success"] = False
        status["output"] = zenml_result["output"]

        return fastapi_response(400, status) 


    status["success"] = True
    status["output"] = zenml_result["output"]

    # check status_deployer and port 
    data = check_model_deployer_data(uuid_deployer)

    if data["success"] != True:
        status["success"] = False
        status["output"] = data["output"]

        return fastapi_response(400, status) 

    # update data model-deployer
    pipeline_name = data["output"]["PIPELINE_NAME"]
    run_name = data["output"]["RUN_NAME"]
    uuid_deployer = data["output"]["UUID"]
    status_deployer = data["output"]["STATUS"]
    url = data["output"]["PREDICTION_URL"]
    parsed_url = urlparse(url)
    port = parsed_url.port 

    db.update_deployer(model_id, pipeline_name, run_name, uuid_deployer, status_deployer, port)
            
    return status  


def delete_data(model_id):
    status = { "success": False, "output": "default response from service function" }

    data = db.get_local_data(model_id)
    # print(data)

    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    uuid_deployer = data["uuid_deployer"]
    model_name = data["uuid_name"].lower()
    run_name = data["run_name"]
    service_name = data["service_name"]

    check_model_path = [
        f"{bentoml_models}/{model_name}",
        f"{bentoml_service}/{service_name}",
        f"{classification_train}/{str(model_id)}",
        f"{detection_train}/{str(model_id)}",
    ]

    check_datasets = [
        f"{detection_datasets}/{model_id}",
        f"{classification_datasets}/{model_id}"
    ]

    try:
        if run_name:
            pipeline_run = Client().get_pipeline_run(run_name)
            for step_name, step_run in pipeline_run.steps.items():
                
                outputs = step_run.outputs
                for output_name, output in outputs.items():
                    artifact_uri = output.uri
                    if os.path.isdir(artifact_uri):
                        shutil.rmtree(artifact_uri)
                        print(f"Delete ::> Step: {step_name}, Output: {output_name}, URI: {artifact_uri}")
        
        for directory_path in check_model_path:
            if os.path.isdir(directory_path):
                shutil.rmtree(directory_path)
                print(f"Delete ::> Local: {directory_path}")

        for directory_path in check_datasets:
            if os.path.exists(directory_path):
                for root, _, files in os.walk(directory_path):
                    for filename in files:
                        file_path = os.path.join(root, filename)

                        # เช็คว่าเป็น symbolic link และเป็นไฟล์ภาพหรือไม่
                        if os.path.islink(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                            os.unlink(file_path)
                            # print(f"Unlinked: {file_path}")

                shutil.rmtree(directory_path)
    
        cmd = [zenml_executable, "model-deployer", "models", "delete", uuid_deployer]
        result = run_zenml_command(cmd, cwd_main)

    except Exception as e:
        print(e)

    db.queue_delete(model_id)
    status["success"] = True
    status["output"] = 'Deleted data success'

    return status


# Define route for deleting ZenML
@app.delete("/zenml-delete", tags=['Start/Stop/Delete Model'])
def delete_zenml(model_id: str, current_user: str = Depends(get_current_user)):
    status = delete_data(model_id)
    return status



# Define route for deleting ZenML
@app.delete("/zenml-delete-by-project", tags=['Start/Stop/Delete Model'])
def delete_model_by_project_id(user_id: str, project_id: str, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default response from service function" }

    data = db.get_by_project(user_id, project_id)

    if data == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    for result in json.loads(data):
        model_id = result["model_id"]
        status = delete_data(model_id)
    
    return status
    


# Define route for deleting ZenML
@app.delete("/zenml-delete-by-user", tags=['Start/Stop/Delete Model'])
def delete_model_by_user_id(user_id: str, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default response from service function" }

    data = db.get_data_by_user(user_id)

    if data == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    for result in json.loads(data):
        model_id = result["model_id"]
        
        status = delete_data(model_id)
    
    return status


#####################################################################################################################
### function train model and deploy
#####################################################################################################################

''' Internal functions '''
def train_and_deploy():

    status = { "errcode": 0, "output": "default response from service function" }
    
    ''' 1) get remain queues waiting for model training '''
    data = db.get_queue_tasks()
    # print(data)

    ''' found no queue left behind, do nothing '''
    if data == []:
        return

    ''' 2) had a next queue to be trained then update status to be 'processing started' '''
    model_id = data["model_id"]
    model_name = data["uuid_name"]
    params = data["parameter"]
    category = data["category"]
    db.queue_on_process(model_id)


    ''' 3) run python script for zenML to process the model training '''
    if category == "detection":

        path_weights = f"{detection_train}/{params['weights']}/weights/best.pt"
        if not os.path.isfile(path_weights):
            path_weights = f"{yolo_pretrain}/{params['weights']}"

        run_script_cmd = [
            "python", "run.py",
            "-c", str(params["neural_network_architecture"]),
            "--model_id", str(model_id),
            "--img_ids", str(params["img_ids"]),
            "--project_id", str(params["project_id"]),
            "--model_name", str(model_name),
            "--authorization_token", params["authorization_token"],
            "--imgsz", str(params["imgsz"]),
            "--batch_size", str(params["batch_size"]),
            "--epochs", str(params["epochs"]),
            "--weights", str(path_weights),
        ]

        run_zenml_model(run_script_cmd, cwd_object_detection, model_id, model_name)
    
    elif category == "classification":
        path_weights = f"{classification_train}/{params['weights']}/best.pth"
        mode = "Custom"
        
        if params['weights'] == "IMAGENET1K_V1" or params['weights'] == "IMAGENET1K_V2":
            mode = "Default"
            path_weights = params['weights']

        run_script_cmd = [
            "python", "run.py",
            "-c", "train_and_deploy",
            "--mode", str(mode),
            "--model_id", str(model_id),
            "--input_path", f"{symbolic_local}/{str(params['datasetes_path'])}",
            "--input_classes", str(params["input_classes"]),
            "--model_name", str(model_name),
            "--batch_size", str(params["batch_size"]),
            "--epochs", str(params["epochs"]),
            "--learning_rate", str(params["learning_rate"]),
            "--neural_network_architecture", str(params["neural_network_architecture"]),
            "--weights", str(path_weights)
        ]

        run_zenml_model(run_script_cmd, cwd_object_classification, model_id, model_name)        
    
    elif category == "activity":
        
        run_script_cmd = [
            "python", "run.py",
            "--model_id", str(model_id),
            "--model_name", str(model_name),
            "--test_size", str(params["test_size"]),
            "--select_model", params["select_model"],
            "--table_name", params["table_name"],
            "--limit_num", str(params["limit_num"]),
            "--max_depth_dtc", str(params["max_depth_dtc"]),
            "--min_samples_split", params["min_samples_split"],
            "--min_samples_leaf", params["min_samples_leaf"],
            "--max_features_dtc", str(params["max_features_dtc"]),
            "--criterion", params["criterion"],
            "--max_depth_rf", str(params["max_depth_rf"]),
            "--n_estimators_rf", str(params["n_estimators_rf"]),
            "--max_features_rf", params["max_features_rf"],
            "--n_estimators_gb", str(params["n_estimators_gb"]),
            "--max_depth_gb", str(params["max_depth_gb"]),
            "--learning_rate", str(params["learning_rate"]),
            "--subsample", str(params["subsample"]),
            "--criterion_gb", params["criterion_gb"]
        ]

        run_zenml_model(run_script_cmd, activity_parameter, model_id, model_name)        
    
    elif category == "covid19":

        run_script_cmd = [
            "python", "run.py",
            "--model_id", str(model_id),
            "--model_name", str(model_name),
            "--test_size", str(params["test_size"]),
            "--select_model", params["select_model"],
            "--table_name", params["table_name"],
            "--limit_num", str(params["limit_num"]),
            "--max_depth_dtc", str(params["max_depth_dtc"]),
            "--min_samples_split", params["min_samples_split"],
            "--min_samples_leaf", params["min_samples_leaf"],
            "--max_features_dtc", str(params["max_features_dtc"]),
            "--criterion", params["criterion"],
            "--max_depth_rf", str(params["max_depth_rf"]),
            "--n_estimators_rf", str(params["n_estimators_rf"]),
            "--max_features_rf", params["max_features_rf"],
            "--n_estimators_gb", str(params["n_estimators_gb"]),
            "--max_depth_gb", str(params["max_depth_gb"]),
            "--learning_rate", str(params["learning_rate"]),
            "--subsample", str(params["subsample"]),
            "--criterion_gb", params["criterion_gb"]
        ]
        
        run_zenml_model(run_script_cmd, covid_19_parameter, model_id, model_name)

    elif category == "Infineon":

        run_script_cmd = [
            "python", "run.py",
            "--model_id", str(model_id),
            "--model_name", str(model_name),
            "--test_size", str(params["test_size"]),
            "--select_model", params["select_model"],
            "--table_name", params["table_name"],
            "--limit_num", str(params["limit_num"]),
            "--max_depth_dtc", str(params["max_depth_dtc"]),
            "--min_samples_split", params["min_samples_split"],
            "--min_samples_leaf", params["min_samples_leaf"],
            "--max_features_dtc", str(params["max_features_dtc"]),
            "--criterion", params["criterion"],
            "--max_depth_rf", str(params["max_depth_rf"]),
            "--n_estimators_rf", str(params["n_estimators_rf"]),
            "--max_features_rf", params["max_features_rf"],
            "--n_estimators_gb", str(params["n_estimators_gb"]),
            "--max_depth_gb", str(params["max_depth_gb"]),
            "--learning_rate", str(params["learning_rate"]),
            "--subsample", str(params["subsample"]),
            "--criterion_gb", params["criterion_gb"]
        ]
        
        run_zenml_model(run_script_cmd, full_hyperparameter, model_id, model_name)                
    
    elif category == "IoT":

        run_script_cmd = [
            "python", "run.py",
            "--model_id", str(model_id),
            "--model_name", str(model_name),
            "--test_size", str(params["test_size"]),
            "--select_model", params["select_model"],
            "--table_name", params["table_name"],
            "--max_depth_dtc", str(params["max_depth_dtc"]),
            "--min_samples_split", params["min_samples_split"],
            "--min_samples_leaf", params["min_samples_leaf"],
            "--max_features_dtc", str(params["max_features_dtc"]),
            "--criterion", params["criterion"],
            "--max_depth_rf", str(params["max_depth_rf"]),
            "--n_estimators_rf", str(params["n_estimators_rf"]),
            "--max_features_rf", params["max_features_rf"],
            "--n_estimators_gb", str(params["n_estimators_gb"]),
            "--max_depth_gb", str(params["max_depth_gb"]),
            "--learning_rate", str(params["learning_rate"]),
            "--subsample", str(params["subsample"]),
            "--criterion_gb", params["criterion_gb"]
        ]
        
        run_zenml_model(run_script_cmd, IoT_parameter, model_id, model_name)

    else:

        run_script_cmd = [
            "python", "run.py",
            "--model_id", str(model_id),
            "--model_name", str(model_name),
            "--table_name", params["table_name"],
            "--test_size", str(params["test_size"]),
            "--n_estimators", str(params["n_estimators"]),
            "--max_samples", str(params["max_samples"]),   # int/float/"auto"
            "--contamination", str(params["contamination"]),  # float/"auto"
            "--max_features", str(params["max_features"]),    # int/float
            "--n_jobs", "None" if params.get("n_jobs") is None else str(params["n_jobs"]),
            "--random_state", "None" if params.get("random_state") is None else str(params["random_state"]),
            # "--verbose", str(params["verbose"]),
        ]
        run_zenml_model(run_script_cmd, Anomaly_parameter, model_id, model_name)
    
    return

#####################################################################################################################
### API train and upload model
#####################################################################################################################

@app.post("/zenml-train-activity_with_5_parameters_per_model", tags=['Apple Watch'])
async def train_model_Hyperparameter_for_Activity_applewatch(params:ActivityHyperParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    # os.environ["MODEL_NAME_Activity"] = params.model_name

    status = { "success": False, "output": "default fn result" }

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Limit train model. Please delete unused data."
        return fastapi_response(400, status)
    

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name, 
        uuid_name = uuid_name,
        project_id = params.project_id,
        category = "activity", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status



@app.post("/zenml-train-covid19_with_5_parameters", tags=['Covid-19'])
async def train_model_Hyperparameter_for_covid19(params:Covid19HyperParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default fn result" }

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Too many requests. Please delete unused data."
        return fastapi_response(400, status)

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name,
        uuid_name = uuid_name, 
        project_id = params.project_id,
        category = "covid19", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status


@app.post("/zenml-train-infineon", tags=['Infineon'])
async def train_model_infineon(params:InfineonHyperParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default fn result" }

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Too many requests. Please delete unused data."
        return fastapi_response(400, status)

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name,
        uuid_name = uuid_name, 
        project_id = params.project_id,
        category = "infineon", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status


@app.post("/zenml-train-IoT", tags=['IoT'])
async def train_model_IoT(params:IoTHyperParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default fn result" }

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Too many requests. Please delete unused data."
        return fastapi_response(400, status)

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name,
        uuid_name = uuid_name, 
        project_id = params.project_id,
        category = "IoT", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status


@app.post("/zenml-train-Anomaly", tags=['Anomaly Detection'])
async def train_model_Anomaly(params:AnomalyDetectionHyperParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default fn result" }

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Too many requests. Please delete unused data."
        return fastapi_response(400, status)

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name,
        uuid_name = uuid_name, 
        project_id = params.project_id,
        category = "Anomaly", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status



@app.get("/pre-train/detection", tags=['Pre-train Model'])
async def get_pretrain():

    data = {
        "YOLOv5": {
            "yolov5l.pt",
            "yolov5m.pt",
            "yolov5n.pt",
            "yolov5s.pt",
            "yolov5x.pt",
        },
        "YOLOv8": {
            "yolov8l.pt",
            "yolov8m.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8x.pt",
        }
    }

    return data


@app.get("/pre-train/classification", tags=['Pre-train Model'])
async def get_NN():

    data = {
        "ResNet50": "IMAGENET1K_V2",
        "VGG16": "IMAGENET1K_V1",
        "DenseNet121": "IMAGENET1K_V1"
    }
    return data


@app.get("/pre-train/custom", tags=['Pre-train Model'])
async def get_model(project_id, neural_network_architecture):

    data = db.query_model(project_id, neural_network_architecture)
    return data
    

@app.post("/train-and-deploy-object-detection", tags=['Object Detection'])
async def train_and_deploy_object_detection(params:DetectionParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default fn result" }
    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Too many requests. Please delete unused data."
        return fastapi_response(400, status)

    # params validation before do the model training trigger
    if params.batch_size not in range(-1, 64):
        status['output'] = 'batch_size exceeds the allowed limit.'
        status['success'] = False
        return fastapi_response(400, status)

    if params.batch_size == 0:
        status['output'] = 'batch_size must not be zero'
        status['success'] = False
        return fastapi_response(400, status)

    if params.epochs < 1:
        status['output'] = 'epochs must be 1 or greater'
        status['success'] = False
        return fastapi_response(400, status)

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name,
        uuid_name = uuid_name, 
        project_id = params.project_id,
        category = "detection", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status


@app.post("/train-and-deploy-object-classification", tags=['Object classification'])
async def train_and_deploy_object_classification(params:ClassificationParams, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):

    status = { "success": False, "output": "default fn result" }
    count = db.get_count_by_userID(params.user_id)

    if count >= 5:
        status['success'] = False
        status['output'] = "Too many requests. Please delete unused data."
        return fastapi_response(400, status)

    # params validation before do the model training trigger
    if params.batch_size not in range(4, 16):
        status['output'] = 'batch_size exceeds the allowed limit.'
        status['success'] = False
        return fastapi_response(400, status)

    if params.learning_rate == 0:
        status['output'] = 'learning_rate must not be zero'
        status['success'] = False
        return fastapi_response(400, status)

    if params.epochs < 1:
        status['output'] = 'epochs must be 1 or greater'
        status['success'] = False
        return fastapi_response(400, status)

    ''' 
        this service function will do following steps:
        1) insert a new queued task
        2) checking current task of model training function
        3) if model training function is busy, then response with designated queue number
        4) if model training function is available, then call the model training function to start the queue
    '''

    ''' 1) insert a new queued task '''
    # get the latest queue number for assigning the new queued task
    new_queue_number = db.get_designated_queue_number()

    uuid_name = str(uuid.uuid4()).replace("-", "")

    # insert a new queued task by designated queue number
    db.insert_data(
        user_id = params.user_id, 
        model_name = params.model_name,
        uuid_name = uuid_name,
        project_id = params.project_id,
        category = "classification", 
        uuid = params.uuid, 
        parameter = params.json(),
        status = json.dumps({ "output": "Wait for queue", "success": True, "task_status": 0 }),
        queue = new_queue_number
    )

    ''' 2) checking current task of model training function '''
    current_tasks = db.get_current_tasks()


    ''' 3) if model training function is busy, then response with designated queue number '''
    if len(current_tasks) > 0:
        status['success'] = True
        status['output'] = f"your task has been added to the queue, your queue number is {new_queue_number}"
        status['queue_number'] = new_queue_number
        return status


    ''' 4) if model training function is available, then call the model training function to start the queue '''
    status['success'] = True
    status['output'] = 'your task was added to the model training process, please check the process status.'

    background_tasks.add_task(train_and_deploy)

    return status



#####################################################################################################################
### get data
#####################################################################################################################
@app.get("/queue", tags=['Check process Train/Deploy'])
async def check_current_queue():

    result = db.check_queue()
    return result


@app.get("/check-status-model", tags=['Check process Train/Deploy'])
async def check_status(user_id,uuid):
    status = { "success": False, "output": "default fn result" }

    result = db.get_by_uuid(user_id=user_id, uuid=uuid)

    if result == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return result


@app.get("/check-model-id", tags=['Check process Train/Deploy'])
async def check_by_id(model_id):
    status = { "success": False, "output": "default fn result" }

    result = db.get_by_id(model_id=model_id)

    if result == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return result


@app.get("/check-project-id", tags=['Check process Train/Deploy'])
async def check_by_project_id(user_id,project_id):
    status = { "success": False, "output": "default fn result" }

    result = db.get_by_project(user_id=user_id, project_id=project_id)

    if result == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return result


@app.get("/user-model-history", tags=['Check process Train/Deploy'])
async def get_model_history(user_id):
    status = { "success": False, "output": "default fn result" }

    result = db.get_data_by_user(user_id=user_id)
    
    if result == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return result


@app.get("/all-user-model-history", tags=['Check process Train/Deploy'])
async def get_all_user_model_history():
    status = { "success": False, "output": "default fn result" }

    result = db.all_user_model_history()

    if result == b'[]':
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return result



@app.get("/graph", tags=['Graph/Evaluation'])
async def get_graph(model_id):

    status = { "success": False, "output": "default fn result" }
    data = db.get_graph(model_id)
    # print(data)

    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return data


@app.get("/evaluation", tags=['Graph/Evaluation'])
async def get_evaluation(model_id):
    status = { "success": False, "output": "default fn result" }
    data = db.get_evaluation(model_id)

    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    return data


@app.get("/image-val", tags=['Graph/Evaluation'])
async def get_val_detect(model_id):
    status = { "success": False, "output": "default fn result" }

    data = db.search_model(model_id)

    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    if data["category"] == "detection":
        dir_path = f"{detection_train}/{model_id}"
        filename = "val_batch0_pred.jpg"
    
    elif data["category"] == "classification":
        dir_path = f"{classification_train}/{model_id}"
        filename = "prediction_summary.jpg"

    if not os.path.exists(dir_path):
        status["output"] = f"data not found"
        return fastapi_response(404, status)

    valid_files = os.path.join(dir_path, filename)
    if not os.path.isfile(valid_files):
        status["output"] = f"No such file from model_id:{model_id}"
        return fastapi_response(404, status)

    with open(valid_files, "rb") as image:
        encoded_string = base64.b64encode(image.read())

    return encoded_string


@app.get("/all-user", tags=['Transaction'])
async def get_transaction(current_user: str = Depends(get_current_user)):
    status = { "success": False, "output": "default fn result" }

    result = db.query_user()
    return result

#####################################################################################################################
### download model
#####################################################################################################################

@app.get("/download_model", tags = ['Download Model'])  
async def download_file(model_id: int, current_user: str = Depends(get_current_user)):

    data = db.search_model(model_id)

    status = { "success": False, "output": "default fn result" }
    
    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    model_name = data["model_name"]   
    uuid_name = data["uuid_name"]    
    version = data["version"]

    split_text = model_name.split("_", 2) 
    filename = split_text[2]

    if version == None:
        logger.error(f"Model directory not found (version is none)")
        
        status["success"] = False
        status["output"] = 'Model directory not found'
        return fastapi_response(404, status)

    # Construct the model directory path
    model_directory_path = os.path.join(bentoml_models, uuid_name.lower(), version)
    logger.info(f'Looking for model directory: {model_directory_path}')
    
    # Check if the model directory exists
    if not os.path.exists(model_directory_path):
        logger.error(f"Model directory not found: {model_directory_path}")
        status["success"] = False
        status["output"] = 'Model directory not found'
        return fastapi_response(404, status)
        
    # Construct the full path to the model file
    model_file_path = os.path.join(model_directory_path, "saved_model.pkl")
    logger.info(f"Model file path: {model_file_path}")
    
    # Check if the model file exists
    if not os.path.exists(model_file_path):
        logger.error(f"Model file not found: {model_file_path}")
        
        status["success"] = False
        status["output"] = 'Model file not found'
        return fastapi_response(404, status)
    
    # Return the model file as a FileResponse with the model name as the filename
    download_filename = f"{filename}.pkl"

    return FileResponse(path=model_file_path, media_type='application/x-ai-model', filename=download_filename)


@app.get("/download_model_img", tags = ['Download Model'])  
async def download_file_img(model_id: int, background_tasks: BackgroundTasks, current_user: str = Depends(get_current_user)):
    data = db.search_model(model_id)

    status = { "success": False, "output": "default fn result" }
    
    if data == []:
        status["success"] = False
        status["output"] = 'data not found'

        return fastapi_response(404, status)

    model_name = data["model_name"]   
    uuid_name = data["uuid_name"]    
    version = data["version"]
    category = data["category"]

    if version == None:
        logger.error(f"Model directory not found (version is none)")
        
        status["success"] = False
        status["output"] = 'Model directory not found'
        return fastapi_response(404, status)

    split_text = model_name.split("_", 2) 
    filename = split_text[2]

    if category == 'detection':

        model_directory_path = f"{detection_train}/{model_id}/weights"
        model_file_path = os.path.join(model_directory_path, "best.pt")
        download_filename = f"{filename}.pt"
    
    else:
        model_directory_path = f"{classification_train}/{model_id}"
        model_file_path = os.path.join(model_directory_path, "best.pth")
        download_filename = f"{filename}.pth"

    logger.info(f"Model file path: {model_file_path}")

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        logger.error(f"Model file not found: {model_file_path}")

        status["success"] = False
        status["output"] = 'Model file not found'
        return fastapi_response(404, status)

    return FileResponse(path=model_file_path, media_type='application/x-ai-model', filename=download_filename)


#####################################################################################################################
### manage port
#####################################################################################################################

# Define route for killing ports
@app.get("/Check-port", tags=['Check/Kill Port'])
async def check_port(current_user: str = Depends(get_current_user)):
    cmd = ['./checkport.sh', "3001", "3031-3032", "3034", "3036", "8000-8079", "8081-8085", "8087-8102"]
    return run_zenml_command(cmd, cwd_main) 

@app.delete("/kill-port", tags=['Check/Kill Port'])
async def kill_port(ports: str = Query(..., description="Space-separated list of ports to kill"), current_user: str = Depends(get_current_user)):
    # Split the input string by spaces to get individual port numbers
    port_args = ports.split()
    cmd = ["./kill_multiple_port.sh"] + port_args
    return run_zenml_command(cmd, cwd_main)

@app.delete("/kill-multiple-port", tags=['Check/Kill Port'])
async def kill_all_port(current_user: str = Depends(get_current_user)):
    cmd = ["./kill_multiple_port.sh", "3001", "3031-3032", "3034", "3036", "8000-8079", "8081-8085", "8087-8102"]
    return run_zenml_command(cmd, cwd_main)



## uvicorn api:app --host 0.0.0.0 --port 7010 --reload
