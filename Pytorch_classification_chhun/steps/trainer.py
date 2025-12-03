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
import time
import math
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import bentoml
import random
from typing import Tuple, Annotated
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    densenet121, DenseNet121_Weights
)
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step


from sklearn.metrics import (
        accuracy_score, 
        confusion_matrix,
        f1_score,
        precision_score, 
        recall_score
    )

logger = get_logger(__name__)

def get_model_and_weights(NeuralNetwork: str, nn_weights, num_classes: int):
    if NeuralNetwork == 'ResNet50':
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif NeuralNetwork == 'VGG16':
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif NeuralNetwork == 'DenseNet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Neural Network Architecture '{NeuralNetwork}' not supported.")

    return model, weights.transforms()


def get_model_and_weights_custom(NeuralNetwork: str, weights, num_classes: int):
    if NeuralNetwork == 'ResNet50':
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(weights))
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif NeuralNetwork == 'VGG16':
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=None)
        model.load_state_dict(torch.load(weights))
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif NeuralNetwork == 'DenseNet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=None)
        model.load_state_dict(torch.load(weights))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Neural Network Architecture '{NeuralNetwork}' not supported.")

    return model, weights.transforms()


@step(
    enable_cache=False
)
def trainer(
    mode: str,
    model_id: int,
    project_id: int,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    neural_network_architecture: str,
    weights: str,
    datasets_path:str,
# ) -> Output(model_name=str, model_path=str, model=torch.nn.Module):
) -> Tuple[
    Annotated[str, "model_name"],
    Annotated[str, "model_path"],
    Annotated[torch.nn.Module, "model"],]:

    try:

        # ''' update evaluation to db'''
        # # setting path postgresqlDB
        # sys.path.append('/app/')
        # from postgresqlDB import ConnectPostgresqlDB

        # db = ConnectPostgresqlDB()
        # db.queue_on_train(params.model_id)

        save_dir = f"/app/Pytorch_classification_chhun/train/{model_id}"
        os.makedirs(save_dir, exist_ok=True)

        dataset = torchvision.datasets.ImageFolder(root=datasets_path + '/train')
        class_names = dataset.classes
        print("classes:", class_names)

        num_classes = len(class_names)

        if mode == "Default":
            model, transform = get_model_and_weights(neural_network_architecture, weights, num_classes)
        else:
            model, transform = get_model_and_weights_custom(neural_network_architecture, weights, num_classes)

        trainset = torchvision.datasets.ImageFolder(root=datasets_path + '/train', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=datasets_path + '/test', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        program_starts = time.time()
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for iteration, data in enumerate(trainloader, 0):
                train_starts = time.time()
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {iteration + 1:5d}] loss: {running_loss / 2000:.3f}')

                json_process = {
                    "epoch": f'{(epoch + 1)}/{epochs}',
                    "iteration": f'{iteration + 1}/{len(trainloader)}',
                    "loss": f"{running_loss/len(inputs):.6f}",
                    "time/iteration": f"{time.time() - train_starts:.6f}"
                }

                print(json_process)

                # db.update_plot_graph(params.model_id, json.dumps(json_process))

                running_loss = 0.0

        program_finished = time.time() - program_starts
        print('Finished Training: ', program_finished)


        model_path = os.path.join(save_dir, f"best.pth")
        torch.save(model.state_dict(), model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # save bento model
        bentoml.pytorch.save_model(
            model_name,
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        )

        # prediction
        model, _ = get_model_and_weights(neural_network_architecture, weights, num_classes)
        print(model_path)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        predicted_labels = []
        predicted_name_labels = []
        all_labels = [] 
        all_name_labels = [] 

        # เก็บข้อมูลภาพและผลลัพธ์
        all_images = []
        all_border_colors = []
        all_markers = []
        all_text_colors = []
        all_pred_labels = []
        all_true_labels = []

        with torch.no_grad():
            for idx, data in enumerate(testloader):
                images, labels = data
                outputs = model(images)
                print("outputs = ", outputs)

                _, predicted = torch.max(outputs, 1)
                print("predicted = ", predicted)

                predicted = predicted.cpu().numpy()  # แปลงเป็น NumPy
                labels = labels.cpu().numpy()

                predicted_labels.extend(predicted)  # แปลงเป็น numpy และเก็บไว้
                all_labels.extend(labels)  # เก็บ GroundTruth

                predicted_name_labels.extend([class_names[pred] for pred in predicted])
                all_name_labels.extend([class_names[label] for label in labels])

                for i in range(len(images)):
                    img = images[i].numpy().transpose((1, 2, 0))  # แปลงเป็น HWC
                    img = (img - img.min()) / (img.max() - img.min())  # Normalize

                    # แปลงค่าคลาสเป็นชื่อ
                    pred_label = class_names[predicted[i]]
                    true_label = class_names[labels[i]]

                    # เช็คว่าทำนายถูกหรือผิด
                    if predicted[i] != labels[i]:  # ทำนายผิด
                        border_color = "red"
                        marker = "X"
                        text_color = "red"
                    else:  # ทำนายถูก
                        border_color = "green"
                        marker = "O"
                        text_color = "green"

                    
                    all_images.append(img)
                    all_border_colors.append(border_color)
                    all_markers.append(marker)
                    all_text_colors.append(text_color)
                    all_pred_labels.append(pred_label)
                    all_true_labels.append(true_label)

            # แยกภาพที่ทายถูก (✅) และผิด (❌)
            correct_images = [(img, border, mark, color, pred, true)
                            for img, border, mark, color, pred, true
                            in zip(all_images, all_border_colors, all_markers,
                                    all_text_colors, all_pred_labels, all_true_labels)
                            if pred == true]

            wrong_images = [(img, border, mark, color, pred, true)
                            for img, border, mark, color, pred, true
                            in zip(all_images, all_border_colors, all_markers,
                                all_text_colors, all_pred_labels, all_true_labels)
                            if pred != true]

            # เลือกภาพที่จะแสดง (สูงสุด 10 รูป)
            num_to_sample = min(10, len(correct_images) + len(wrong_images))
            max_wrong = min(len(wrong_images), num_to_sample // 2)
            max_correct = num_to_sample - max_wrong

            selected_wrong = random.sample(wrong_images, max_wrong) if max_wrong > 0 else []
            # selected_correct = random.sample(correct_images, max_correct) if max_correct > 0 else []
            if max_correct > 0:
                selected_correct = random.sample(correct_images, min(len(correct_images), max_correct))
            else:
                selected_correct = []

            selected_images = selected_wrong + selected_correct
            random.shuffle(selected_images)

            # ปรับ Grid Layout อัตโนมัติ
            num_images = len(selected_images)
            cols = min(5, num_images)
            rows = math.ceil(num_images / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

            if num_images == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # วาดภาพ
            for ax, (image, border_color, marker, text_color, pred_label, true_label) in zip(axes, selected_images):
                ax.imshow(image)
                ax.axis("off")
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(5)

                ax.text(5, 20, f"{marker} {pred_label} / {true_label}",
                        fontsize=16, color=text_color, fontweight="bold",
                        bbox=dict(facecolor='white', alpha=0.75))

            # ซ่อนช่องที่เกินมา
            for ax in axes[len(selected_images):]:
                ax.axis("off")

            # บันทึกภาพรวมเพียง 1 ภาพ
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, "prediction_summary.jpg")
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # รีเซ็ตอาร์เรย์เพื่อเตรียมชุดใหม่
            all_images = []

        print(predicted_name_labels)
        print(predicted_labels)

        f1 = f1_score(all_labels, predicted_labels, average='micro')  # สามารถเปลี่ยนค่า average ได้
        print(f'F1 Score: {f1:.6f}')

        accuracy = accuracy_score(all_labels, predicted_labels)
        print(f'Accuracy: {accuracy:.6f}')

        precision = precision_score(all_labels, predicted_labels, zero_division=0, average='micro')
        print(f'Precision: {precision:.6f}')

        recall = recall_score(all_labels, predicted_labels, average='micro')
        print(f'Recall: {recall:.6f}')

        conf_matrix = confusion_matrix(all_name_labels, predicted_name_labels, labels=class_names)
        print(conf_matrix)

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        json_data = {
            "classes": class_names,
            "total_time": program_finished,
            "f1_score": f'{f1:.6f}',
            "accuracy": f'{accuracy:.6f}',
            "precision": f'{precision:.6f}',
            "recall": f'{recall:.6f}',
            "confusion_matrix": conf_matrix.tolist()
        }

        print(json_data)

        # db.update_evaluation(params.model_id, json.dumps(json_data))

        # วนลูปทุกไดเรกทอรีย่อย
        for root, _, files in os.walk(datasets_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # เช็คว่าเป็น symbolic link และเป็นไฟล์ภาพหรือไม่
                if os.path.islink(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    os.unlink(file_path)
                    # print(f"Unlinked: {file_path}")


    except Exception as e:
        raise Exception("Error", f"{e}")
    
    
    os.environ["MODEL_NAME_Class"] = model_name
    os.environ["CLASS_NAMES"] = str(class_names)
    os.environ["NeuralNetwork"] = str(neural_network_architecture)
    # db.queue_on_deployment(params.model_id)

    return model_name, model_path, model
