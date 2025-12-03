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


from zenml.config import DockerSettings
# from zenml.integrations.constants import GCP, MLFLOW
from zenml.pipelines import pipeline
from steps.data_loader import data_loader
from steps.trainer import trainerV8
from steps.model_loader import model_loaderV8
from steps.deployment_trigger import deployment_trigger
from steps.bento_builderV8 import bento_builderV8
from steps.bento_deployer import bentoml_model_deployer
docker_settings = DockerSettings(
    parent_image="ultralytics/yolo:latest",
    requirements="./requirements.txt",
    dockerignore=".dockerignore",
)


@pipeline(
    enable_cache=False,
    settings={
        "docker": docker_settings,
    },
)
def yolov5_pipeline(
    trainer,
    model_loader,
    deployment_trigger,
    bento_builder,
    deployer,
    project_id: int,
    img_ids: str,
    authorization_token: str,

):
    # datasets_path, train, valid, test = data_loader()
    datasets_path = data_loader(project_id, img_ids, authorization_token)
    model_path, model_name = trainer(datasets_path)

    model = model_loader(model_path, model_name)
    decision = deployment_trigger(model_path)
    bento = bento_builder(model=model, model_name=model_name)
    deployer(deploy_decision=decision, bento=bento, model_name=model_name)


@pipeline(
    enable_cache=False,
    settings={
        "docker": docker_settings,
    },
)
def yolov8_pipeline(
    project_id: int,
    img_ids: str,
    authorization_token: str,
    model_id: int,
    model_name: str,
    imgsz: int,
    batch_size: int,
    epochs: int,
    weights: str,

):
    datasets_path = data_loader(project_id, img_ids, authorization_token)
    print(f'dataset Path ===> {datasets_path}')
    model_path, model_name = trainerV8(
                                        model_id=model_id,
                                        project_id=project_id,
                                        model_name=model_name,
                                        imgsz=imgsz,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        weights=weights,
                                        datasets_path=datasets_path,
                                        )
    print(f'model_path = {model_path} model_name = {model_name}')

    model = model_loaderV8(model_id, model_path, model_name)
    decision = deployment_trigger(model_path)
    bento = bento_builderV8(model_name=model_name, model=model)
    # deployer(deploy_decision=decision, bento=bento, model_name=model_name)
    bentoml_model_deployer(deploy_decision=decision, bento=bento, model_name=model_name)