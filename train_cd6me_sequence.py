import os

# Suppress all warnings
import warnings

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from cd6me_apex_dataset import CD6ME_Apex_Dataset
from cd6me_models import TinyVIT
from cd6me_sequence_dataset import DatasetConfig, LandmarksDataset, RGBDataset
from datasets_utility import FocalLoss, PairedDataset
from geometric_models import GCN_TCN
from models_utilities import EarlyStopper, count_parameters

warnings.filterwarnings("ignore")


def standardization(values):
    x_mean = values.mean(dim=0, keepdim=True)  # Mean across all nodes
    x_std = values.std(dim=0, keepdim=True)  # Std across all nodes
    x_std = torch.clamp(x_std, min=1e-8)  # Prevent division by zero
    standardized_values = (values - x_mean) / x_std
    return standardized_values


batch_size = 16
num_classes = 12
epochs = 50
multi_task_learning_flag = False
frame_level_analysis_flag = False
temporal_uniform_sampling = False

threshold = 0.5

transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        # transforms.Resize((384, 384)), # for case when landmark is outputed as 384.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # NoisyUniformTemporalSubsample(8)
    ]
)

transform_of = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # NoisyUniformTemporalSubsample(8)
    ]
)


# Local Paths
df = pd.read_csv("metadata_csv/metadata_csv_statistics.csv")
dataset_path = "/home/hq/Documents/data"

weights_path = "/home/hq/Documents/Weights/CD6ME"
mag_weights = "/media/hq/ExtraSpace/Weights/Magnification/generator_212000.pth"
technique_name = "Q-Seq-11"

technique_name = "random"
number_of_frames = 40


running_loss = 0.0
correct = 0
total = 0

aus = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]

dataset = ["casme", "casme2", "samm", "fourd", "mmew", "casme3a"]

for dataset_fold in dataset:
    early_stopper = EarlyStopper(patience=5, min_delta=0)
    weights_name_mlflow = "{}/{}/{}_{}".format(
        weights_path, technique_name, number_of_frames, dataset_fold
    )
    weights_name = weights_name_mlflow + ".pth"
    weights_name_motion = weights_name_mlflow + "_motion.pth"
    weights_name_motion_classifier = weights_name_mlflow + "_motion_classifier.pth"

    if os.path.exists(weights_path + "/" + technique_name) == False:
        os.makedirs(weights_name_mlflow)

    mlflow_experiment_name = "{}/{}".format(weights_name_mlflow, "mlflow")
    if os.path.exists(mlflow_experiment_name) == False:
        os.makedirs(mlflow_experiment_name)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run(run_name=mlflow_experiment_name)

    df = pd.read_csv(
        "metadata_csv/metadata_csv_statistics.csv"
    )  # this line needs to be added to prevent overwriting
    LandmarksDataset_Config = DatasetConfig(
        dataset_path=dataset_path,
        num_frames=16,
        test_dataset_fold=dataset_fold,
        mode="train",
        data_type="Landmarks",
        transform=transform,
        mag_factor=10,
        sampling_rate=16,
    )

    RGBDataset_Config = DatasetConfig(
        dataset_path=dataset_path,
        num_frames=16,
        test_dataset_fold=dataset_fold,
        mode="train",
        data_type="RGB",
        transform=transform,
        mag_factor=10,
        sampling_rate=16,
    )

    cd6me_train_rgb = RGBDataset(config=RGBDataset_Config, dataframe=df)

    cd6me_train_landmarks = LandmarksDataset(
        config=LandmarksDataset_Config, dataframe=df
    )

    cd6me_train_motion = CD6ME_Apex_Dataset(
        dataset_path=dataset_path,
        dataframe=df,
        test_dataset_fold=dataset_fold,
        mode="train",
        input_type="optical_flow",
        transform=transform_of,
    )

    train_paired_dataset = PairedDataset(
        datasetA=cd6me_train_landmarks,
        datasetB=cd6me_train_rgb,
        datasetC=cd6me_train_motion,
    )

    train_dataloader = DataLoader(
        train_paired_dataset, batch_size=batch_size, shuffle=True, num_workers=False
    )

    model = GCN_TCN(node_features=130, num_classes=num_classes).cuda()
    model_motion_patches = TinyVIT(num_classes=num_classes).cuda()
    model_motion_classifier = TinyVIT(num_classes=num_classes).cuda()

    focal_criterion = FocalLoss(gamma=2.0, loss_type="bce")
    focal_criterion_emote = FocalLoss(gamma=2.0, loss_type="ce")

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # adam优化器
    scheduler = MultiStepLR(
        optimizer,
        milestones=np.arange(start=0, stop=epochs, step=20).tolist(),
        gamma=0.9,
    )

    optimizer_motion = optim.Adam(
        model_motion_patches.parameters(), lr=0.001
    )  # adam优化器
    scheduler_motion = ExponentialLR(optimizer_motion, gamma=0.9)

    optimizer_motion_classifier = optim.Adam(
        model_motion_classifier.parameters(), lr=0.001
    )  # adam优化器
    scheduler_motion_classifier = ExponentialLR(optimizer_motion_classifier, gamma=0.9)

    count_parameters(model)
    count_parameters(model_motion_patches)
    # count_parameters(infusenet)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        running_loss_motion = 0.0
        running_loss_motion_logits = 0.0
        running_loss_motion_classifier = 0.0
        running_loss_broker = 0.0
        running_loss_coral = 0.0

        w1 = 0.0
        w2 = 0.0
        correct = 0
        total = 0
        pred = []
        gt = []

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            optimizer_motion.zero_grad()
            optimizer_motion_classifier.zero_grad()

            # for paired dataset
            topological_data, texture_data, motion_data = data

            # vid mamba input: batch size, channel, frame num, x, y
            images, labels, edges, emotion, cluster_indices, batch_indices = (
                topological_data[0],
                topological_data[1],
                topological_data[2],
                topological_data[3],
                topological_data[4],
                topological_data[5],
            )
            images_texture, labels_texture = texture_data[0], texture_data[1]
            images_motion, labels_motion, face_region, aus_active, au_context = (
                motion_data[0],
                motion_data[1],
                motion_data[2],
                motion_data[3],
                motion_data[4],
            )

            images = standardization(values=images)
            images = images.float().cuda()
            labels = labels.cuda()
            labels = labels.float()
            emotion = emotion.long().cuda()
            edges = edges.long().cuda()
            batch_indices = batch_indices.long().cuda()
            cluster_indices = cluster_indices.flatten().cuda()

            images_texture = images_texture.float().cuda()
            labels_texture = labels_texture.cuda()
            images_motion = images_motion.float().cuda()

            _, x_feat_128, p_att_128_logits, _ = model_motion_patches(
                images_motion, graph_context=None
            )
            outputs_motion, motion_classifier_feat_128, _, _ = model_motion_classifier(
                images_motion, graph_context=None
            )

            outputs_topology, x4_mean = model(
                images,
                edges,
                motion_context=x_feat_128,
                cluster_indices=cluster_indices,
                batch_indices=batch_indices,
            )

            outputs = outputs_topology

            if len(outputs) == 1:
                outputs = outputs.logits

            loss = focal_criterion(outputs, labels)
            loss_motion_logits = focal_criterion(p_att_128_logits, labels)
            loss_motion_classifier = focal_criterion(outputs_motion, labels)

            loss = loss + loss_motion_logits + loss_motion_classifier

            outputs_topology = F.sigmoid(outputs_topology)
            outputs_motion = F.sigmoid(outputs_motion)
            sigmoid_outputs = 1 * outputs_topology + 1 * outputs_motion
            predictions = torch.where(sigmoid_outputs > threshold, 1, 0)

            loss.backward()
            optimizer.step()
            optimizer_motion.step()
            optimizer_motion_classifier.step()

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            # Accumulate individual losses
            running_loss += loss.item()
            running_loss_motion_logits += loss_motion_logits.item()
            running_loss_motion_classifier += loss_motion_classifier.item()

            pred += [predictions.cpu().numpy().tolist()]
            gt += [labels.cpu().numpy().tolist()]

        if multi_task_learning_flag:
            gt = np.hstack(gt)
            pred = np.hstack(pred)

        else:
            gt = np.vstack(gt)
            pred = np.vstack(pred)
        f1_per_epoch = f1_score(gt, pred, average="macro")
        acc_per_epoch = f1_score(gt, pred, average="micro")

        print(f1_score(gt, pred, average=None))
        # print(pred)f1

        # Calculate average losses
        avg_total_loss = running_loss / len(train_dataloader)
        # avg_motion_loss = running_loss_motion / len(train_dataloader)
        avg_motion_loss_logits = running_loss_motion_logits / len(train_dataloader)
        avg_motion_loss_classifier = running_loss_motion_classifier / len(
            train_dataloader
        )

        # Log individual losses to MLflow
        mlflow.log_metric("training_loss_total", avg_total_loss, step=epoch)
        mlflow.log_metric(
            "training_loss_motion_logits", avg_motion_loss_logits, step=epoch
        )
        mlflow.log_metric(
            "training_loss_motion_classifier", avg_motion_loss_classifier, step=epoch
        )

        mlflow.log_metric("MacroF1", f1_per_epoch, step=epoch)
        mlflow.log_metric("Accuracy", acc_per_epoch, step=epoch)

        scheduler.step()
        scheduler_motion.step()
        scheduler_motion_classifier.step()

        # Enhanced print statement with individual losses
        print(
            f"Epoch {epoch+1}, F1-score: {f1_per_epoch:.4f}, Accuracy: {acc_per_epoch:.4f}%, "
            f"Total Loss: {avg_total_loss:.4f}, "
            f"Motion Loss Logits: {avg_motion_loss_logits:.4f}, "
            f"Motion Loss Classifier: {avg_motion_loss_classifier:.4f}, "
        )

    torch.save(model.state_dict(), weights_name)
    torch.save(model_motion_patches.state_dict(), weights_name_motion)
    torch.save(model_motion_classifier.state_dict(), weights_name_motion_classifier)

    # log model
    mlflow.pytorch.log_model(model, "weights_name")
    mlflow.end_run()
