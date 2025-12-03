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
import cv2
import base64
import bentoml
import numpy as np
from bentoml.io import Text, Image, PandasDataFrame, Multipart
from typing import List

MODEL_NAME = os.getenv("MODEL_NAME_Obj")

class Yolov8Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):
        super().__init__()
        bento_model = bentoml.pytorch.get(MODEL_NAME)
        self.model = bentoml.pytorch.load_model(bento_model)
        self.inference_size = 640

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs: List[np.ndarray], conf: List[float]):
        results = self.model.predict(input_imgs, conf=conf[0], imgsz=self.inference_size)
        return [r.tojson() for r in results]

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs: List[np.ndarray], conf: List[float]):
        results = self.model.predict(input_imgs, conf=conf[0], imgsz=self.inference_size)
        return [r.plot() for r in results]  # r.plot() returns ndarray



SERVICE_NAME = f"{MODEL_NAME}_service"
yolo_runner = bentoml.Runner(Yolov8Runnable, max_batch_size=30)
svc = bentoml.Service(SERVICE_NAME, runners=[yolo_runner])


@svc.api(input=Multipart(input_img=Image(), conf=Text()), output=PandasDataFrame())
async def inference(input_img, conf):
    confidence = float(conf)
    results = await yolo_runner.inference.async_run([input_img], [confidence])
    import pandas as pd
    return pd.read_json(results[0])


@svc.api(input=Multipart(input_img=Image(), conf=Text()), output=Text())
async def render(input_img, conf):
    confidence = float(conf)
    results = await yolo_runner.render.async_run([input_img], [confidence])
    image_array = results[0]
    success, buffer = cv2.imencode(".png", image_array)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")