import os
import sys
import torch
import json
import time
import pandas as pd
import threading
from ultralytics import YOLO, settings
from yolov5.train import main, parse_opt
# from zenml.client import Client
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step
from constants import MODEL_PATH
from typing import Tuple, Annotated


logger = get_logger(__name__)
# experiment_tracker = Client().active_stack.experiment_tracker

# step_operator = Client().active_stack.step_operator

# if not step_operator or not experiment_tracker:
#     raise RuntimeError(
#         "Your active stack needs to contain a step operator and an "
#         "experiment tracker to run this pipeline."
#     )

class TrainerParameters(BaseParameters):
    """Trainer params"""

    model_id: int
    project_id: int
    model_name: str
    imgsz: int
    batch_size: int
    epochs: int
    weights: str


@step(
    enable_cache=False,
    # step_operator=step_operator.name,
    # experiment_tracker=experiment_tracker.name,
)
def trainerV5(
    model_id: int,
    project_id: int,
    model_name: str,
    imgsz: int,
    batch_size: int,
    epochs: int,
    weights: str,
    datasets_path:str,
) -> Tuple[
        Annotated[str, "model_path"],
        Annotated[str, "model_name"],
]:

    """Train a neural net from scratch to recognize MNIST digits return our
    model or the learner"""

    try:

        ''' update evaluation to db'''
        # setting path postgresqlDB
        sys.path.append('/app/')
        # from postgresqlDB import ConnectPostgresqlDB

        # db = ConnectPostgresqlDB()
        # db.queue_on_train(model_id)

        opt = parse_opt(known=True)
        opt.model_id = model_id
        opt.imgsz = imgsz
        opt.batch_size = batch_size
        opt.epochs = epochs
        opt.exist_ok = True
        opt.data = datasets_path + "/data.yaml"
        opt.name = str(model_id)
        opt.weights = weights
        opt.project = MODEL_PATH

        main(opt)

        model_path = os.path.join(MODEL_PATH, opt.name, "weights", "best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the best model and return it
        model = torch.load(model_path)

        if model is None:
            raise RuntimeError(f"Failed to load model from {model_path}")
        
        # วนลูปทุกไดเรกทอรีย่อย
        for root, _, files in os.walk(datasets_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # เช็คว่าเป็น symbolic link และเป็นไฟล์ภาพหรือไม่
                if os.path.islink(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    os.unlink(file_path)
                    # print(f"Unlinked: {file_path}")
    
    except Exception as e:
        raise Exception("Error", f"{e}")
    
    
    return model_path, model_name



@step(
    enable_cache=False,
)
def trainerV8(
    model_id: int,
    project_id: int,
    model_name: str,
    imgsz: int,
    batch_size: int,
    epochs: int,
    weights: str,
    datasets_path:str,
# ) -> Output(model_path=str, model_name=str):
) -> Tuple[
        Annotated[str, "model_path"],
        Annotated[str, "model_name"],
]:

    """Train a neural net from scratch to recognize MNIST digits return our
    model or the learner"""

    try:

        ''' update evaluation to db'''
        # setting path postgresqlDB
        sys.path.append('/app/')
        # from postgresqlDB import ConnectPostgresqlDB

        # db = ConnectPostgresqlDB()
        # db.queue_on_train(model_id)
        
        model_id = model_id
        imgsz = imgsz
        batch_size = batch_size
        epochs = epochs
        data = datasets_path + "/data.yaml"
        name = str(model_id)
        weights = weights
        project = MODEL_PATH

        # Update a setting
        settings.update({"mlflow": False})

        # load a pretrained model (recommended for training)
        model = YOLO(weights)
        path_csv = f"{project}/{model_id}" 


## In case you don't want to train the model: close this below
        stop_event = threading.Event()
        # thread = threading.Thread(target=save_results_to_db, args=(model_id, path_csv, epochs, stop_event), daemon=True)
        # thread.start()

        results = model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch_size, name=name, project=project)
        # print(results)

        stop_event.set()
        # thread.join()

        print("result =", results.results_dict)
        # db.update_evaluation(model_id, json.dumps(results.results_dict))
        print(f'MODEL_PATH is ===============> {MODEL_PATH} <===================')

        model_path = os.path.join(MODEL_PATH, name, "weights", "best.pt")
        print(f'model_path is found in ===============> {model_path} <===================')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # วนลูปทุกไดเรกทอรีย่อย
        for root, _, files in os.walk(datasets_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # เช็คว่าเป็น symbolic link และเป็นไฟล์ภาพหรือไม่
                if os.path.islink(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    os.unlink(file_path)
                    # print(f"Unlinked: {file_path}")
    
    except Exception as e:
        raise Exception("Error", f"{e}")
    
    
    return model_path, model_name


# def save_results_to_db(model_id, path, epochs, stop_event):
        
#     results_csv = f"{path}/results.csv"
#     last_epoch = -1

#     while not stop_event.is_set():
#         if os.path.exists(results_csv):
#             try:
#                 df = pd.read_csv(results_csv)
#                 df.columns = df.columns.str.strip()

#                 if len(df) > last_epoch + 1:
#                     new_rows = df.iloc[last_epoch + 1:]
#                     # print("df = ",new_rows)

#                     for _, row in new_rows.iterrows():
#                         data_json = {
#                             "epoch": f'{int(row["epoch"])}/{epochs}',
#                             "box_loss": float(row["train/box_loss"]),
#                             "cls_loss": float(row["train/cls_loss"]),
#                             "dfl_loss": float(row["train/dfl_loss"]),
#                         }

#                         json_process = json.dumps(data_json, indent=2)
#                         print(json_process)

#                         # setting path postgresqlDB
#                         sys.path.append('/app/')
#                         from postgresqlDB import ConnectPostgresqlDB
#                         db = ConnectPostgresqlDB()
#                         db.update_plot_graph(model_id, json_process)

#                     last_epoch = len(df) - 1
#             except Exception as e:
#                 print(f"⚠️ Error reading CSV: {e}")
#         time.sleep(1)