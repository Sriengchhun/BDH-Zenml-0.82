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

import os
import sys
import torch
from ultralytics import YOLO
from zenml.steps import Output, step , BaseParameters
from typing import Tuple, Annotated



class LoaderParameters(BaseParameters):
    """Model Loader params"""
    
    model_id: int


@step
def model_loaderV5(params:LoaderParameters, model_path:str, model_name:str) -> Output(model=torch.nn.Module):
    """Loads the trained models from previous training pipeline runs."""

    ''' update status to db'''
    # setting path postgresqlDB
    sys.path.append('/app/')
    from postgresqlDB import ConnectPostgresqlDB

    db = ConnectPostgresqlDB()
    db.queue_on_deployment(params.model_id)

    print(model_path)

    try:
        sys.path.insert(0, "yolov5")
        from model import wrapped_model

    except Exception as e:
        print(e)

    os.environ["MODEL_NAME_Obj"] = model_name

    return wrapped_model(model_path)


@step
def model_loaderV8(
    model_id: int,
    model_path: str,
    model_name: str,
) -> Annotated[torch.nn.Module, "model"]:
    """Loads the trained models from previous training pipeline runs."""

    ''' update status to db'''
    # setting path postgresqlDB
    sys.path.append('/app/')
    # from postgresqlDB import ConnectPostgresqlDB

    # db = ConnectPostgresqlDB()
    # db.queue_on_deployment(params.model_id)

    print(model_path)

    model = YOLO(model_path)

    os.environ["MODEL_NAME_Obj"] = model_name

    return model

