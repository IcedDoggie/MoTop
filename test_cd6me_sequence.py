import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from cd6me_apex_dataset import CD6ME_Apex_Dataset
from cd6me_models import TinyVIT
from cd6me_sequence_dataset import (
    DatasetConfig,
    LandmarksDataset,
    RGBDataset,
)
from datasets_utility import PairedDataset
from geometric_models import GCN_TCN
from rule_set_aus_to_emotions import au_emotion_recognizer



def standardization(values):
    x_mean = values.mean(dim=0, keepdim=True)  # Mean across all nodes
    x_std = values.std(dim=0, keepdim=True)  # Std across all nodes
    x_std = torch.clamp(x_std, min=1e-8)  # Prevent division by zero
    standardized_values = (values - x_mean) / x_std
    return standardized_values


batch_size = 8
num_classes = 12
epochs = 1
multi_task_learning_flag = False
frame_level_analysis_flag = False
temporal_uniform_sampling = False

threshold = 0.5

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整图像大小到48x48
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化,
        # NoisyUniformTemporalSubsample(8)
    ]
)
transform_of = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 调整图像大小到48x48
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化,
    ]
)



# Local Paths
df = pd.read_csv("metadata_csv/cross_dataset_seq.csv")
dataset_path = "/home/hq/Documents/data"

# weights_path = "/home/hq/Documents/Weights/CD6ME"
weights_path = "/media/hq/ExtraSpace/Weights/CD6ME"

mag_weights = "/media/hq/ExtraSpace/Weights/Magnification/generator_212000.pth"


technique_name = 'random'
number_of_frames = 40


results_table = PrettyTable()
results_table.field_names = ["Dataset Fold", "Accuracy", "Macro F1"]

au_results_table = PrettyTable()
au_results_table.field_names = [
    "AU1",
    "AU2",
    "AU4",
    "AU5",
    "AU6",
    "AU7",
    "AU9",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "Average",
]

f1_accumulator = []

running_loss = 0.0
correct = 0
total = 0

aus = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]


gt_cm = []
pred_cm = []

dataset = ["casme", "casme2", "samm", "fourd", "mmew", "casme3a"]

for dataset_fold in dataset:
    weights_name_mlflow = "{}/{}/{}_{}".format(
        weights_path, technique_name, number_of_frames, dataset_fold
    )
    # weights_name_motion = "{}/{}/{}_{}_motion".format(
    #     weights_path, motion_technique_name, number_of_frames, dataset_fold
    # )
    weights_name = weights_name_mlflow + ".pth"
    weights_name_motion = weights_name_mlflow + "_motion.pth"
    weights_name_motion_classifier = weights_name_mlflow + "_motion_classifier.pth"
    weights_name_motion_patches = weights_name_mlflow + "_motion_patches.pth"

    df = pd.read_csv("metadata_csv/metadata_csv_statistics.csv")

    LandmarksDataset_Config = DatasetConfig(
        dataset_path="/home/hq/Documents/data",
        num_frames=16,
        test_dataset_fold=dataset_fold,
        mode="test",
        data_type="Landmarks",
        transform=transform,
        mag_factor=10,
        sampling_rate=16,
    )

    RGBDataset_Config = DatasetConfig(
        dataset_path="/home/hq/Documents/data",
        num_frames=16,
        test_dataset_fold=dataset_fold,
        mode="test",
        data_type="RGB",
        transform=transform,
        mag_factor=10,
        sampling_rate=16,
    )

    cd6me_test_rgb = RGBDataset(config=RGBDataset_Config, dataframe=df)

    cd6me_test_landmarks = LandmarksDataset(
        config=LandmarksDataset_Config, dataframe=df
    )
    cd6me_test_geometry = Landmarks3DDataset(
        config=LandmarksDataset_Config, dataframe=df
    )
    cd6me_test_smirk = SMIRKDataset(config=LandmarksDataset_Config, dataframe=df)

    cd6me_test_motion = CD6ME_Apex_Dataset(
        dataset_path=dataset_path,
        dataframe=df,
        test_dataset_fold=dataset_fold,
        mode="test",
        input_type="optical_flow",
        transform=transform_of,
    )

    test_paired_dataset = PairedDataset(
        datasetA=cd6me_test_landmarks,
        datasetB=cd6me_test_rgb,
        datasetC=cd6me_test_motion,
    )

    # cd6me_test = CD6ME_Sequence_Dataset(dataset_path=dataset_path, num_frames=number_of_frames, dataframe=df, test_dataset_fold=dataset_fold, mode='test', data_type='OpticalFlow', temporal_uniform_sampling=temporal_uniform_sampling, transform=transform)
    test_dataloader = DataLoader(
        test_paired_dataset, batch_size=batch_size, shuffle=False, num_workers=False
    )

    model_motion = TinyVIT(num_classes=12)
    dict_s = torch.load(weights_name_motion_classifier)
    model_motion.load_state_dict(dict_s, strict=True)
    model_motion.eval()

    # weights_name_motion = weights_name_motion.replace('P-Seq-25', 'P-Seq-13')
    model_motion_feat = TinyVIT(num_classes=12)
    dict_s = torch.load(weights_name_motion)
    model_motion_feat.load_state_dict(dict_s, strict=True)
    model_motion_feat.eval()

    # weights_name = weights_name.replace('P-Seq-25', 'P-Seq-13')
    model = GCN_TCN(node_features=130, num_classes=12)
    dict_s = torch.load(weights_name)
    model.load_state_dict(dict_s, strict=False)
    model.eval()

    model_motion = model_motion.cuda()
    model = model.cuda()
    model_motion_feat = model_motion_feat.cuda()

    correct = 0
    total = 0
    pred = []
    gt = []

    for i, data in enumerate(test_dataloader):

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
        images_motion, labels_motion, face_region, aus_active, _ = (
            motion_data[0],
            motion_data[1],
            motion_data[2],
            motion_data[3],
            motion_data[4],
        )

        images = images.float().cuda()
        ori_images = images.clone()
        labels = labels.cuda()
        labels = labels.float()
        edges = edges.long().cuda()
        emotion = emotion.long().cuda()
        batch_indices = batch_indices.long().cuda()
        cluster_indices = cluster_indices.flatten().cuda()

        images = standardization(values=images)

        images_motion = images_motion.float().cuda()
        images_texture = images_texture.float().cuda()

        _, x_feat_128, p_att_128_logits, _ = model_motion_feat(
            images_motion, graph_context=None
        )
        outputs_motion, _, _, _ = model_motion(images_motion, graph_context=None)


        outputs_topology, _ = model(
            images,
            edges,
            motion_context=x_feat_128,
            cluster_indices=cluster_indices,
            batch_indices=batch_indices,
        )
        # outputs = (outputs_topology + outputs_motion + x_mag) / 3
        outputs_topology = F.sigmoid(outputs_topology)
        outputs_motion = F.sigmoid(outputs_motion)
        sigmoid_outputs = 1 * outputs_topology + 1 * outputs_motion

        predictions = torch.where(sigmoid_outputs > threshold, 1, 0)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        pred += [predictions.cpu().numpy().tolist()]
        gt += [labels.cpu().numpy().tolist()]


    gt = np.vstack(gt)
    pred = np.vstack(pred)
    f1_per_epoch = f1_score(gt, pred, average="macro")
    acc_per_epoch = f1_score(gt, pred, average="micro")

    au_emotion_recognizer(dataset_fold, pred_aus=pred)


    gt_cm.append(gt)
    pred_cm.append(pred)
    print(f1_score(gt, pred, average=None))

    # aus scores accumulator (this will go into benchmarking sheet)
    f1_accumulator += [f1_score(gt, pred, average=None)]

    results_table.add_row([dataset_fold, acc_per_epoch, f1_per_epoch])


f1_accumulator = np.vstack(f1_accumulator).mean(axis=0)
mean_f1_accumulator = np.mean(f1_accumulator)
f1_accumulator = np.append(f1_accumulator, mean_f1_accumulator)
au_results_table.add_row(f1_accumulator)
print(results_table)
print(au_results_table)
