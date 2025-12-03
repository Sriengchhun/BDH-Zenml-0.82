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

import sys
import os

import bentoml
from bentoml.io import Text, Image, PandasDataFrame, Multipart

import base64
from io import BytesIO


MODEL_NAME = os.getenv("MODEL_NAME_Obj")

class Yolov5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import torch

        sys.path.insert(0, "yolov5")
        from models.common import AutoShape

        try:
            # model_path = INFERENCE_MODEL
            # self.model = torch.hub.load("yolov5", 'custom', path=model_path, source='local')
            
            bento_model = bentoml.pytorch.get(MODEL_NAME)
            self.model = bentoml.pytorch.load_model(bento_model)
            self.model = AutoShape(self.model.model.model)

            if torch.cuda.is_available():
                self.model.cuda()
            else:
                self.model.cpu()

            # Config inference settings
            self.inference_size = 320

            # Optional configs
            # self.model.conf = 0.50  # NMS confidence threshold
            # self.model.iou = 0.45  # NMS IoU threshold
            # self.model.agnostic = False  # NMS class-agnostic
            # self.model.multi_label = False  # NMS multiple labels per box
            # self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
            # self.model.max_det = 1000  # maximum number of detections per image
            # self.model.amp = False  # Automatic Mixed Precision (AMP) inference

        except Exception as e:
            print("Error during prediction: ", e)
    
    # @bentoml.Runnable.method(batchable=True, batch_dim=0)
    # def set_confidence_threshold(self, conf):
    #     self.model.conf = float(conf[0])
    #     return "Set confidence successfully"

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(self, input_imgs, conf):
        # Return predictions only
        self.model.conf = float(conf[0])
        results = self.model(input_imgs, size=self.inference_size)
        return results.pandas().xyxy

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def render(self, input_imgs, conf):
        # Return images with boxes and labels
        self.model.conf = float(conf[0])
        return self.model(input_imgs, size=self.inference_size).render()

SERVICE_NAME = f"{MODEL_NAME}_service"
yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30)
svc = bentoml.Service(SERVICE_NAME, runners=[yolo_v5_runner])

# @svc.api(input=Text(), output=Text())
# async def set_confidence(conf):
#     result = await yolo_v5_runner.set_confidence_threshold.async_run([conf])
#     return result


@svc.api(input=Multipart(input_img=Image(), conf=Text()), output=PandasDataFrame())
async def invocation(input_img, conf):
    batch_ret = await yolo_v5_runner.inference.async_run([input_img],[conf])
    return batch_ret[0]


# @svc.api(input=Text(), output=JSON())
# async def invocation_dir_path(input_img_path):
#     list_files = [file for file in os.listdir(input_img_path) if file.endswith((".jpg", ".jpeg", ".png"))]
#     results = []
#     for image_filename in list_files:
#         input_img = cv2.imread(os.path.join(input_img_path,image_filename))
#         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

#         batch_ret = await yolo_v5_runner.inference.async_run([input_img])
        
#         results.append({
#             "filename" : image_filename,
#             "data" : batch_ret[0]
#         })
#     return results


# @svc.api(input=Image(), output=Image())
# async def render_image(input_img):
#     batch_ret = await yolo_v5_runner.render.async_run([input_img])
#     return batch_ret[0]


@svc.api(input=Multipart(input_img=Image(), conf=Text()), output=Text())
async def render(input_img, conf):
    from PIL import Image

    batch_ret = await yolo_v5_runner.render.async_run([input_img],[conf])

    array_bytes = Image.fromarray(batch_ret[0])

    buffer = BytesIO()
    array_bytes.save(buffer, format="PNG")
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return base64_string


# @svc.api(input=Text(), output=JSON())
# async def render_dir_path(input_img_path):
#     from PIL import Image

#     list_files = [file for file in os.listdir(input_img_path) if file.endswith((".jpg", ".jpeg", ".png"))]
#     results = []
#     for image_filename in list_files:
#         input_img = cv2.imread(os.path.join(input_img_path,image_filename))
#         input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

#         batch_ret = await yolo_v5_runner.render.async_run([input_img])

#         array_bytes = Image.fromarray(batch_ret[0])
#         buffer = BytesIO()
#         array_bytes.save(buffer, format="PNG")

#         image_bytes = buffer.getvalue()
#         base64_encoded = base64.b64encode(image_bytes)
#         base64_string = base64_encoded.decode('utf-8')
#         results.append({
#             "filename" : image_filename,
#             "base64" : base64_string
#         })
#     return results