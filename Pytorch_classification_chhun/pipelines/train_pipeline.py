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
from zenml.pipelines import pipeline
from steps.data_loader import data_loader
from steps.trainer import trainer
from steps.deployment_trigger import deployment_trigger
from steps.bento_builder import bento_builder
from steps.bento_deployer import bentoml_model_deployer
docker_settings = DockerSettings(
    requirements="./requirements.txt",
    dockerignore=".dockerignore",
)


@pipeline(
    enable_cache=False,
    settings={
        "docker": docker_settings,
    },
)
def pytorch_classification_pipeline(
    mode: str,
    model_id: int,
    project_id: int,
    input_path: str,
    input_classes: str,
    model_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    neural_network_architecture: str,
    weights: str
):
    datasets_path = data_loader(project_id, input_path, input_classes)
    print(f'train_pipeline datasetpath = {datasets_path}')
    # datasets_path = data_loader()
    # model_name, model_path, model = trainer(datasets_path)
    model_name, model_path, model = trainer(
                mode = mode,
                model_id=model_id,
                project_id=project_id,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                neural_network_architecture=neural_network_architecture,
                weights=weights,
                datasets_path=datasets_path,)
    print(f'train pipeline: ==> model_paht = {model_path}')
    decision = deployment_trigger(model_path)
    bento = bento_builder(model_name=model_name, model=model)
    bentoml_model_deployer(model_name=model_name, deploy_decision=decision, bento=bento)