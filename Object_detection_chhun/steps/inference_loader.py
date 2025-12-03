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
from typing import Dict, List

import cv2
from zenml.post_execution import get_pipeline
from zenml.steps import Output, step

from zenml.client import Client
from constants import INFERENCE_TEST_IMAGES

@step
def inference_loader() -> Output(images_path=List):
    """Loads the trained models from previous training pipeline runs."""
    client = Client()
    last_run = client.get_pipeline("yolov5_pipeline").last_run

    images_directory = INFERENCE_TEST_IMAGES
    image_files = [file for file in os.listdir(images_directory) if file.endswith((".jpg", ".jpeg", ".png"))]
    
    # Create a list to store the paths of the images
    images = []

    # Loop through the image files and append their paths to the list
    for image_file in image_files:
        images.append(os.path.join(images_directory, image_file))
    
    return images