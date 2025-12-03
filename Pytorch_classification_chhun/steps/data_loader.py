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
import numpy as np
import shutil
import datetime

from typing import Any
from sklearn.model_selection import train_test_split
from constants import DATASET_PATH

from zenml.steps import BaseParameters, Output, step


def create_new_datasets(input_path, datasets_path, input_classes):

    test_ratio = 0.2
    print("input:", input_classes)

    # เรียกดู class จากชื่อโฟลเดอร์
    classes = []
    for json_data in eval(input_classes):
        if os.path.isdir(os.path.join(input_path, json_data["sub_folderID"])):
            classes.append([json_data["sub_folderID"], json_data["name"]])
    
    print(f"Classes: {classes}")

    if not classes:
        print("Classes not found in the datasets path.")
        raise FileNotFoundError("Classes not found in the datasets path.")

    class_names = []
    for sub_folderID, name in classes:
        
        # อ่านไฟล์ทั้งหมด เพื่อเตรียมแบ่งชุดข้อมูล
        print(f"sub_folderID: {sub_folderID}, name: {name}")
        dir_path = os.path.join(input_path, sub_folderID, "originals")
        print(dir_path)

        skip_files = [file for file in os.listdir(dir_path) if not os.path.isfile(os.path.join(dir_path, file))]
        files = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
        
        if not files:
            print(f"No files found in directory: {sub_folderID}")
            raise FileNotFoundError(f"No files found in directory: {sub_folderID}")

        # แบ่งชุดข้อมูล
        print(f'Class: {name}, images: {len(files)}')

        if len(files) < 10:
            raise ValueError(f"The dataset must contain more than 10 images.")

        train_list, test_list = train_test_split(files, test_size=test_ratio, random_state=42)
        
        print(f'Data not found, Skip file: {len(skip_files)}', skip_files)
        print(f'Train list for {name}: {len(train_list)}', train_list)
        print(f'Test list for {name}: {len(test_list)}', test_list)

        # สร้างโฟลเดอร์สำหรับ train และ test
        train_output_path = os.path.join(datasets_path, "train", name)
        test_output_path = os.path.join(datasets_path, "test", name)
        
        os.makedirs(train_output_path)
        os.makedirs(test_output_path)


        try:
            # สร้าง symbolic links สำหรับ train set
            for train_data in train_list:
                src = os.path.join(dir_path, train_data)
                # print("train: ",src)
                new_path = os.path.join(train_output_path, train_data)
                os.symlink(src, new_path)

            # สร้าง symbolic links สำหรับ test set
            for test_data in test_list:
                src = os.path.join(dir_path, test_data)
                # print("test: ",src)
                new_path = os.path.join(test_output_path, test_data)
                os.symlink(src, new_path)
        
        except Exception as e:
            print(e)
            # print(f"Skip : {image_path}")
    
        class_names.append(name)

    return class_names


@step
def data_loader(
    project_id,
    input_path,
    input_classes,
) -> str:
    
    print(f"time : {datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')}")
    print(input_path) 
    
    """Loads data"""
    datasets_path = DATASET_PATH + str(project_id)

    if os.path.isdir(datasets_path):
        shutil.rmtree(datasets_path)

    classes = create_new_datasets(
        input_path,
        datasets_path,
        input_classes,
    )
    print(f'datasets_path = {datasets_path}')
    print("classes: ", classes)
    return datasets_path