#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
from typing import List

from rich import print as rich_print
from zenml.integrations.bentoml.services import BentoMLDeploymentService
from zenml.steps import step, BaseParameters

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time 
import base64
from io import BytesIO
from constants import INFERENCE_IMAGES

class PredictParameters(BaseParameters):
    """Predict params"""

    img_data: str
    confidence: str


@step
def predictor(
    params: PredictParameters,
    service: BentoMLDeploymentService,
) -> None:
    """Run an inference request against the BentoML prediction service.

    Args:
        service: The BentoML service.
        data: The data to predict.
    """
    prev_time = time.time()
    # os.makedirs(os.path.join(INFERENCE_IMAGES), exist_ok=True)

    try:
        service.start(timeout=30)
        list_files = [file for file in os.listdir(params.img_data) if file.endswith((".jpg", ".jpeg", ".png"))]
        i = 0
        for image_filename in list_files:
            i = i+1
            check_time = time.time()
            img = cv2.imread(os.path.join(params.img_data,image_filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            service.predict("set_confidence", params.confidence)

            msg_datafram = service.predict("invocation", img)
            rich_print(msg_datafram)

            image_path = os.path.join(INFERENCE_IMAGES, f'{os.path.basename(image_filename)}')
            msg_image = service.predict("render", img)

            bytes_data = base64.b64decode(msg_image)
            image = Image.open(BytesIO(bytes_data))
            image.save(image_path)


    except Exception as e:
        print("Error during prediction: ", e)
