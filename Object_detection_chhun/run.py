#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import click
from pipelines.train_pipeline import yolov5_pipeline, yolov8_pipeline
from steps import (
    bento_builderV5,
    bento_builderV8,
    bentoml_model_deployer,
    data_loader,
    deployment_trigger,
    model_loaderV5,
    model_loaderV8,
    trainerV5,
    trainerV8,
)

# from steps.data_loader import DatasetsParameters
from steps.trainer import TrainerParameters
from steps.model_loader import LoaderParameters

import sys
import shutil
import datetime
from constants import DATASET_PATH

YOLOV5 = "yolov5"
YOLOV8 = "yolov8"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([YOLOV5, YOLOV8]),
    default="None",
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model",
)

@click.option(
    "--model_id",
    type=int,
    help="Model ID from DB",
)


@click.option(
    "--project_id",
    type=int,
    help="Project ID for download datasets from Label Studio",
)

@click.option(
    "--img_ids",
    type=str,
    help="select datasets",
)

@click.option(
    "--model_name",
    type=str,
    help="Set model name by user",
)

@click.option(
    "--authorization_token",
    type=str,
    help="Token of User, Project owner for download datasets from Label Studio",
)

@click.option(
    "--imgsz",
    type=int,
    help="train, val image size (pixels)",
)

@click.option(
    "--batch_size",
    type=int,
    help="total batch size for all GPUs, -1 for autobatch",
)

@click.option(
    "--epochs",
    type=int,
    help="total training epochs",
)


@click.option(
    "--weights",
    type=str,
    help="initial weight path",
)


def main(
    config: str,
    model_id: int,
    project_id: int,
    img_ids: str,
    model_name: str,
    authorization_token: int,
    imgsz: int,
    batch_size: int,
    epochs: int,
    weights: str,
):
    
    yolov5 = config == YOLOV5
    yolov8 = config == YOLOV8

    if yolov5:
        training_pipeline = yolov5_pipeline(
            data_loader=data_loader(
                # params=DatasetsParameters(
                    project_id=project_id,
                    img_ids=img_ids,
                    authorization_token=authorization_token
                # )
            ),
            trainer=trainerV5(
                params=TrainerParameters(
                    model_id=model_id,
                    project_id=project_id,
                    model_name=model_name,
                    imgsz=imgsz, 
                    batch_size=batch_size,
                    epochs=epochs,
                    weights=weights
                )
            ),
            model_loader=model_loaderV5(
                params=LoaderParameters(
                    model_id=model_id,
                )
            ),
            deployment_trigger=deployment_trigger(),
            bento_builder=bento_builderV5,
            deployer=bentoml_model_deployer,
        )
        run_name = f"yolov5_pipeline-{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')}"

        ''' update evaluation to db'''
        # setting path postgresqlDB
        sys.path.append('/app/')
        # from postgresqlDB import ConnectPostgresqlDB

        # db = ConnectPostgresqlDB()
        # db.update_run_name(model_id, run_name)

        training_pipeline.run(run_name=run_name)
        shutil.rmtree(DATASET_PATH + str(project_id))
    
    if yolov8:
        training_pipeline = yolov8_pipeline(
            project_id=project_id,
            img_ids=img_ids,
            authorization_token=authorization_token,
            model_id=model_id,
            model_name=model_name,
            imgsz=imgsz,
            batch_size=batch_size,
            epochs=epochs,
            weights=weights,
            ## 9 parameters
        )

        run_name = f"Object_detection_pipeline-{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')}"
        print(run_name)
        sys.path.append('/app/')
        
        shutil.rmtree(DATASET_PATH + str(model_id))


if __name__ == "__main__":
    main()



### python run.py -c yolov8 --model_id 5 --project_id 87 --img_ids [0]  --model_name testv8 --imgsz 640 --batch_size -1 --epochs 5 --weights /app/Object_detection/pre-train/yolov8n.pt --authorization_token 3f5d56d3caf780d2114fd9d4871744fd20efde45


### python run.py -c yolov8 --model_id 64 --project_id 81 --img_ids [0] --model_name testv8 --imgsz 640 --batch_size -1 --epochs 1 --weights /app/Object_detection_chhun/pre-train/yolov8n.pt --authorization_token 3f5d56d3caf780d2114fd9d4871744fd20efde45

## python run.py -c yolov8 --model_id 10 --project_id 81 --img_ids [0] --model_name testv8 --imgsz 640 --batch_size -1 --epochs 1 --weights /app/Object_detection_chhun/pre-train/yolov8n.pt --authorization_token 3f5d56d3caf780d2114fd9d4871744fd20efde45
