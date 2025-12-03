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
from zenml.client import Client
from pipelines.train_pipeline import pytorch_classification_pipeline
from steps import (
    bento_builder,
    bentoml_model_deployer,
    data_loader,
    deployment_trigger,
    trainer,
)

# from steps.data_loader import DatasetsParameters
# from steps.trainer import TrainerParameters

import sys
import shutil
import datetime
from constants import DATASET_PATH


TRAIN_AND_DEPLOY = "train_and_deploy"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([TRAIN_AND_DEPLOY]),
    default="None",
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`train`), or to "
    "only run a prediction against the deployed model "
    "(`deploy`). By default both will be run "
    "(`train_and_deploy`).",
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
    "--input_path",
    type=str,
    help="select datasets",
)


@click.option(
    "--input_classes",
    type=str,
    help="select classes for datasets",
)


@click.option(
    "--model_name",
    type=str,
    help="Set model name by user",
)


@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="train, val image size (pixels)",
)


@click.option(
    "--batch_size",
    type=int,
    default=4,
    help="total batch size for all GPUs",
)

@click.option(
    "--epochs",
    type=int,
    default=1,
    help="total training epochs",
)


@click.option(
    "--neural_network_architecture",
    type=str,
    help="select Neural Network",
)

@click.option(
    "--weights",
    type=str,
    help="initial weights path",
)

@click.option(
    "--mode",
    type=str,
    default="Default",
    help="default pre-train or custom",
)


def main(
    config: str,
    mode: str,
    model_id: int,
    project_id: int,
    input_path: str,
    input_classes: str,
    model_name: str,
    learning_rate: int,
    batch_size: int,
    epochs: int,
    neural_network_architecture: str,
    weights: str

):
    train = config == TRAIN_AND_DEPLOY

    if train:
        training_pipeline = pytorch_classification_pipeline(
            project_id=project_id,
            input_path=input_path,
            input_classes=input_classes,
            mode=mode,
            model_id=model_id,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            neural_network_architecture=neural_network_architecture,
            weights=weights
        )
        run_name = f"pytorch_classification_pipeline-{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')}"
        print(run_name)

        # shutil.rmtree(DATASET_PATH + str(project_id))
        shutil.rmtree(DATASET_PATH + str(model_id))


if __name__ == "__main__":
    main()

# python run.py -c train_and_deploy --model_id 1 --mode Default --project_id 131 --model_name test --input_path /data/local-files/?d=input/608175dae5cd71001119f3bd/25/169 --input_classes [{"name": "dog", "sub_folderID": "171"}, {"name": "cat", "sub_folderID": "170"}] --batch_size 4 --neural_network_architecture ResNet50 --weights IMAGENET1K_V2 --learning_rate 0.001



# For test in local: # python run.py -c train_and_deploy --model_id 1 --mode Default --project_id 131 --model_name test --input_path "/app/Pytorch_classification_chhun/datasets/data_local" --input_classes "[{\"name\":\"dog\",\"sub_folderID\":\"171\"},{\"name\":\"cat\",\"sub_folderID\":\"170\"}]" --batch_size 4 --neural_network_architecture ResNet50 --weights IMAGENET1K_V2 --learning_rate 0.001

# python run.py -c train_and_deploy --model_id 1 --project_id 131 --model_name test_class --input_path /data/local-files/?d=input/608175dae5cd71001119f3bd/25/169 --input_classes "[{\"name\":\"dog\",\"sub_folderID\":\"171\"},{\"name\":\"cat\",\"sub_folderID\":\"170\"}]" --batch_size 4 --neural_network_architecture ResNet50 --weights IMAGENET1K_V2 --learning_rate 0.001