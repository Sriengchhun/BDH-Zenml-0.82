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
import bentoml
import base64
import torch.nn as nn
import torch
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    densenet121, DenseNet121_Weights
)
from bentoml.io import Image, JSON, Multipart, Text
from PIL import Image as PILImage, ImageDraw
from io import BytesIO

from torchvision import transforms


MODEL_NAME = os.getenv("MODEL_NAME_Class")
CLASS_NAMES = os.getenv("CLASS_NAMES")
NeuralNetwork = os.getenv("NeuralNetwork")

runner = bentoml.pytorch.get(MODEL_NAME).to_runner()
svc = bentoml.Service(f"{MODEL_NAME}_service", runners=[runner])

def preprocess_image(image: PILImage.Image):
    if NeuralNetwork == 'ResNet50':
        weights = ResNet50_Weights.IMAGENET1K_V2
        transform = weights.transforms()

    elif NeuralNetwork == 'VGG16':
        weights = VGG16_Weights.IMAGENET1K_V1
        transform = weights.transforms()

    elif NeuralNetwork == 'DenseNet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1
        transform = weights.transforms()

    else:
        raise ValueError(f"Neural Network Architecture '{NeuralNetwork}' not supported.")

    return transform(image).unsqueeze(0)

# def preprocess_image(image: PILImage.Image):
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),    
#         transforms.ToTensor(),          
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     return transform(image).unsqueeze(0)


def image_to_base64(image: PILImage.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@svc.api(input=Multipart(input_image=Image(), conf=Text()), output=JSON())
async def classify(input_image: PILImage.Image, conf):

    input_tensor = preprocess_image(input_image)
    result = await runner.async_run(input_tensor)
    
    softmax = torch.nn.functional.softmax(result, dim=1)
    confidence, label = torch.max(softmax, 1)
    class_names = eval(CLASS_NAMES)
    class_name = class_names[label.item()]

    if float(confidence) >= float(conf) :
        result = {"label": class_name, "confidence": float(confidence.item())}

    else:
        result = {"label": None, "confidence": None}    
    
    return result


@svc.api(input=Multipart(input_image=Image(), conf=Text()), output=JSON())
async def classify_and_return_base64(input_image:PILImage.Image, conf):

    input_tensor = preprocess_image(input_image)
    result = await runner.async_run(input_tensor)
    
    softmax = torch.nn.functional.softmax(result, dim=1)
    confidence, label = torch.max(softmax, 1)
    class_names = eval(CLASS_NAMES)
    class_name = class_names[label.item()]

    draw_image = input_image.convert("RGB")

    if float(confidence) >= float(conf) :
        draw = ImageDraw.Draw(draw_image)
        draw.text((10, 10), f"Label: {label.item()}, Confidence: {confidence.item():.2f}", fill=(255, 0, 0))
        base64_image = image_to_base64(draw_image)

        result = {"label": class_name, "confidence": float(confidence.item()), "image_base64": base64_image}
    
    else:
        result = {"label": None, "confidence": None, "image_base64": base64_image}
    
    return result




