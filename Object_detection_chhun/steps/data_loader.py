from zenml.steps import BaseParameters, Output, step

import os
import cv2
import numpy as np
import datetime
import requests
import shutil
import yaml

from typing import Any, Dict, Annotated
# from genericpath import isdir
from sklearn.model_selection import train_test_split
from constants import DATASET_PATH

def create_datasets(class_ids, task, datasets_path):
    skip_path = []
    data_path = task["data"]["image"]

    try:
        if data_path.endswith(('jpg', 'png', 'jpeg')) and os.path.isfile(data_path):
            image_path = data_path
            original_name = os.path.basename(image_path)
            basename = original_name.rsplit('.', 1)[0]

            if not os.path.exists(datasets_path):
                os.makedirs(datasets_path + "/labels")
                os.makedirs(datasets_path + "/images")

            annotations = task["annotations"]
            if annotations != []:
                annotations_result = annotations[0]["result"]
                print(image_path)

                if annotations_result != None:
                    # YOLO formatted lines
                    yolo_lines = []

                    for result in annotations_result:
                        bbox = result['value']

                        image_width = result['original_width']
                        image_height = result['original_height']

                        pixel_x = image_width * bbox['x'] / 100.0
                        pixel_y = image_height * bbox['y'] / 100.0
                        pixel_width = image_width * bbox['width'] / 100.0
                        pixel_height = image_height * bbox['height'] / 100.0

                        x = pixel_x / image_width * 100.0 
                        y = pixel_y / image_height * 100.0
                        width = pixel_width / image_width * 100.0
                        height = pixel_height / image_height *100.0


                        # Calculate the center x, center y, width, height in normalized format
                        x_center = (x + width / 2) / 100
                        y_center = (y + height / 2) / 100
                        width_normalized = width / 100
                        height_normalized = height / 100
                        
                        # Get the label and check if it's already in the class_ids mapping
                        label = bbox['rectanglelabels'][0]
                        if label not in class_ids:
                            # class_ids[label] = len(class_ids)  # Assign a new class ID
                            raise AssertionError(f"Impossible class labels is {label}, Possible class labels is {class_ids}")
                        
                        # Get the class id
                        class_id = class_ids[label]
                        
                        # Format the line for YOLO
                        line = f"{class_id} {x_center} {y_center} {width_normalized} {height_normalized}"
                        yolo_lines.append(line)

                    # Write to a text file
                    output_image = datasets_path + "/images/" + original_name
                    os.symlink(image_path, output_image)

                    output_file = datasets_path + "/labels/" + basename + ".txt"
                    with open(output_file, "w") as file:
                        for line in yolo_lines:
                            file.write(line + "\n")
        else:
            print(f"Dataset not found, skip file: {data_path}")
            skip_path.append(data_path)

    except Exception as e:
        print(e)
        print(f"Skip : {data_path}")
    

def task_datasets(project_id, img_ids, authorization_token, datasets_path):
    headers = {'Authorization': f'Token {authorization_token}'}
    url_class = f"https://label.aidery.io/api/projects/project-labels/{project_id}"
    response_class = requests.get(url_class,headers=headers)

    if response_class.status_code == 200:
        labels = response_class.json()
        class_ids = {name: str(index) for index, name in enumerate(labels)}

        if img_ids == '[0]':
            url_task = f"https://label.aidery.io/api/projects/{project_id}/tasks/?page_size=-1"
            response_task = requests.get(url_task, headers=headers)
            
            if response_task.status_code == 200:
                img_ids = []
                for task in response_task.json():
                    img_id = task["id"]
                    annotations = task["annotations"]
                    
                    if annotations != [] and annotations[0]["result"] != "null" and annotations[0]["result"] != []:
                        img_ids.append(img_id)
        else:
            img_ids = eval(img_ids)

        print(len(img_ids))

        if len(img_ids) < 10:
            raise ValueError(f"The dataset must contain more than 10 images.")

        # กำหนดอัตราส่วนการแบ่งข้อมูล
        train_ratio = 0.8
        validation_ratio = 0.1
        test_ratio = 0.1

        # แบ่งข้อมูลออกเป็น train + validation และ test set
        train_data, test_data = train_test_split(img_ids, test_size=test_ratio, random_state=42)

        # แบ่ง train_data ออกเป็น train และ validation set
        train_data, validation_data = train_test_split(train_data, test_size=validation_ratio/(train_ratio + validation_ratio), random_state=42)

        print("Train data:", train_data)
        print("Validation data:", validation_data)
        print("Test data:", test_data)

        os.makedirs(datasets_path)
        
        swapped_dict = {int(value): key for key, value in class_ids.items()}
        data = {
            'path' : datasets_path,
            'train' : 'train',
            'val' : 'val',
            'test' : 'test',
            'names': swapped_dict
            }

        classes_yaml = f"{datasets_path}/data.yaml"
        with open(classes_yaml, 'w') as file:
            yaml.dump(data, file, sort_keys=False)

        for train_data in train_data:
            url_task_ids= f"https://label.aidery.io/api/tasks/{train_data}?project={project_id}"
            response_task = requests.get(url_task_ids, headers=headers)

            if response_task.status_code == 200:
                task = response_task.json()
                create_datasets(class_ids,task, datasets_path + "/train")
                
        for validation_data in validation_data:
            url_task_ids= f"https://label.aidery.io/api/tasks/{validation_data}?project={project_id}"
            response_task = requests.get(url_task_ids, headers=headers)

            if response_task.status_code == 200:
                task = response_task.json()
                create_datasets(class_ids,task, datasets_path + "/val")

        for test_data in test_data:
            url_task_ids= f"https://label.aidery.io/api/tasks/{test_data}?project={project_id}"
            response_task = requests.get(url_task_ids, headers=headers)

            if response_task.status_code == 200:
                task = response_task.json()
                create_datasets(class_ids,task, datasets_path + "/test")
                
    return response_task.status_code


@step
def data_loader(
    project_id,
    img_ids,
    authorization_token,
) -> str:
    print(f"time : {datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')}")
    
    """Loads data"""
    images: dict(str, list(np.ndarray, list)) = {}
    train_images: dict(str, list(np.ndarray, list)) = {}
    valid_images: dict(str, list(np.ndarray, list)) = {}
    test_images: dict(str, list(np.ndarray, list)) = {}

    datasets_path = DATASET_PATH + str(project_id)
    # datasets_path = "/app/Object_detection/datasets_test"
    print(f'dataset_path in dataloader == {datasets_path}')

    if os.path.isdir(datasets_path):
        shutil.rmtree(datasets_path)

    response = task_datasets(
        project_id,
        img_ids,
        authorization_token,
        datasets_path,
    )

    if response == 200:
        return datasets_path
