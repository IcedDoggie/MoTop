import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from natsort import natsorted
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import functional as F
from torch.utils.data import Dataset


def custom_collate_fn_auxiliary_data(batch):
    """
    batch: list of tensors with shape (N_i, C, H, W)
    returns: single tensor of shape (sum(N_i), C, H, W)
    """
    main_frames, main_labels, aux_frames, aux_labels = zip(*batch)
    main_frames = torch.stack(main_frames, dim=0)  # Shape: (B, N, C, H, W)
    aux_frames = torch.cat(aux_frames, dim=0)  # Shape: (B, N, C, H, W)
    main_labels = torch.stack(main_labels, dim=0)  # Shape: (B, N)
    aux_labels = torch.cat(aux_labels, dim=0)  # Shape: (B
    main_frames = torch.cat(
        (main_frames, aux_frames), dim=0
    )  # Concatenate along the batch dimension
    main_labels = torch.cat((main_labels, aux_labels), dim=0)  #
    return main_frames, main_labels


def uniform_temporal_sampling(frames: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Uniformly samples num_samples frames from the input tensor.
    frames: Tensor of shape [T, ...] where T is the number of frames.
    Returns: Tensor of shape [num_samples, ...]
    """
    T = frames.shape[0]
    if T == num_samples:
        return frames
    elif T < num_samples:
        # If not enough frames, repeat last frame
        indices = torch.linspace(0, T - 1, steps=num_samples).long()
        indices = torch.clamp(indices, max=T - 1)
        return frames[indices]
    else:
        # Uniformly sample indices
        indices = torch.linspace(0, T - 1, steps=num_samples).long()
        return frames[indices]



def custom_collate_fn(batch):
    meshes, edges, labels = zip(*batch)
    meshes = torch.stack(meshes)
    # edges is a tuple of numpy arrays or tensors of different shapes
    # Keep as a list, or pad if needed
    labels = torch.tensor(labels)
    return meshes, list(edges), labels


def pad_edges(edges):
    dim_to_pad = 0

    max_len = max(edge.shape[dim_to_pad] for edge in edges)
    # max_len = 10002
    padded = []
    for edge in edges:
        pad_width = ((0, max_len - edge.shape[dim_to_pad]), (0, 0))
        padded_edge = np.pad(edge, pad_width, mode="constant", constant_values=0)
        padded.append(torch.tensor(padded_edge))
    return np.stack(padded)


def map_samm_emotion_to_5class(emotion):
    emotion = emotion.lower()
    if emotion == "happiness":
        return "happiness"
    elif emotion == "disgust":
        return "disgust"
    elif emotion == "surprise":
        return "surprise"
    elif emotion == "sadness":
        return "sadness"
    else:
        return "others"


def window_frame_extraction(frame, frame_list):
    apex_idx = frame_list.index(frame[0])  # Get the index of the apex frame
    if apex_idx > 1 and apex_idx < len(frame_list) - 2:
        start = max(apex_idx - 2, 0)
        end = min(apex_idx + 3, len(frame_list))
    elif apex_idx < 2:
        apex_idx = 2  # force the apex to be at least 2 frames in
        start = max(apex_idx - 2, 0)
        end = min(apex_idx + 3, len(frame_list))
    elif apex_idx > len(frame_list) - 3:
        apex_idx = len(frame_list) - 3  # force the apex to be at least 2 frames in
        start = max(apex_idx - 2, 0)
        end = min(apex_idx + 3, len(frame_list))

    return start, end


def get_loso_splits(annotation_file):
    """
    Creates Leave-One-Subject-Out (LOSO) splits.
    """
    if type(annotation_file) == str:
        df = pd.read_excel(annotation_file)
    else:
        df = annotation_file
    logo = LeaveOneGroupOut()

    # emotion_classes_omittance = ['Other']
    # df = df.loc[~df['Estimated Emotion'].isin(emotion_classes_omittance)]

    subjects = df["Subject"].unique()
    subjects = sorted(subjects)
    splits = {}

    for train_idx, test_idx in logo.split(df, groups=df["Subject"]):
        test_subject = df.iloc[test_idx]["Subject"].unique()[0]
        splits[test_subject] = (df.iloc[train_idx], df.iloc[test_idx])

    return splits


class PairedDataset(Dataset):
    def __init__(self, rgb_dataset, threeD_dataset):
        self.rgb_dataset = rgb_dataset
        self.threeD_dataset = threeD_dataset

        # Ensure both datasets have the same length
        assert len(self.threeD_dataset) == len(
            self.rgb_dataset
        ), "The two datasets must have the same length."

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, index):
        rgb_data = self.rgb_dataset[index]
        threeD_data = self.threeD_dataset[index]

        # Return both data as a tuple
        return rgb_data, threeD_data


def calculate_class_weights(dataset, num_classes=5):
    """
    Calculate class weights based on dataset distribution

    Args:
        dataset: Your dataset object
        num_classes: Number of classes (5 for CASME2)

    Returns:
        class_weights: Tensor of weights for each class
    """
    # Collect all labels from the dataset
    all_labels = []

    # Method 1: If dataset has direct access to labels
    if hasattr(dataset, "data_rows"):
        # For datasets with data_rows attribute
        all_labels = dataset.data_rows["EstimatedEmotion_5class_Quantized"].tolist()
    elif hasattr(dataset, "annotations"):
        # For datasets with annotations attribute
        all_labels = dataset.annotations["EstimatedEmotion_5class_Quantized"].tolist()
    else:
        # Method 2: Iterate through dataset (slower but more general)
        print("Collecting labels by iterating through dataset...")
        for i in range(len(dataset)):
            try:
                if len(dataset[i]) == 2:  # (data, label)
                    _, label = dataset[i]
                elif len(dataset[i]) == 3:  # (nodes, edges, label)
                    _, _, label = dataset[i]
                else:
                    label = dataset[i][-1]  # Last element is usually label

                if torch.is_tensor(label):
                    label = label.item()
                all_labels.append(label)
            except Exception as e:
                print(f"Error getting label for index {i}: {e}")
                continue

    # Convert to numpy array
    all_labels = np.array(all_labels)

    # Count class distribution
    class_counts = Counter(all_labels)
    print("Class distribution:")
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        percentage = (count / len(all_labels)) * 100
        print(f"  Class {class_id}: {count} samples ({percentage:.2f}%)")

    # Calculate weights using sklearn
    class_weights_sklearn = compute_class_weight(
        class_weight="balanced", classes=np.unique(all_labels), y=all_labels
    )

    # Ensure we have weights for all classes (in case some are missing)
    class_weights = np.ones(num_classes)
    unique_classes = np.unique(all_labels)
    for i, class_id in enumerate(unique_classes):
        class_weights[class_id] = class_weights_sklearn[i]

    # Convert to torch tensor
    class_weights_tensor = torch.FloatTensor(class_weights)

    print("Calculated class weights:")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f}")

    return class_weights_tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", loss_type="ce"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, logits, targets):
        if self.loss_type == "ce":
            # Cross-entropy loss expects long targets
            targets_long = targets.long()
            ce_loss = F.cross_entropy(logits, targets_long, reduction="none")
            pt = torch.exp(-ce_loss)

            # Apply alpha weighting
            if self.alpha is not None:
                if self.alpha.type() != logits.data.type():
                    self.alpha = self.alpha.type_as(logits.data)
                # Convert targets to long for gather operation
                at = self.alpha.gather(0, targets_long.data.view(-1))
                ce_loss = ce_loss * at

        elif self.loss_type == "bce":
            # Binary cross-entropy with logits for multi-label
            ce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

            # For BCE, we need to handle alpha differently since targets are multi-dimensional
            if self.alpha is not None:
                if self.alpha.type() != logits.data.type():
                    self.alpha = self.alpha.type_as(logits.data)

                # Expand alpha to match target dimensions
                alpha_expanded = self.alpha.unsqueeze(0).expand_as(targets)

                # Apply different weights for positive and negative samples
                # For positive samples (targets == 1), use alpha
                # For negative samples (targets == 0), use (1 - alpha) or just 1
                alpha_t = torch.where(
                    targets == 1, alpha_expanded, 1.0 - alpha_expanded
                )
                ce_loss = ce_loss * alpha_t

            # For BCE, pt calculation is different
            sigmoid_p = torch.sigmoid(logits)
            pt = torch.where(targets == 1, sigmoid_p, 1 - sigmoid_p)

        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def visualize_patches_around_landmarks(input_patches):
    patches_tensor = input_patches[0]
    nodes, x, y, c = patches_tensor.shape
    for i in range(nodes):
        patch = patches_tensor[i]
        # patch = np.transpose(patch, (2, 0, 1))  # to H, W, C
        plt.imshow(patch)

    a = 1
    return 1


# /media/hq/Seagate Backup Plus Drive/opticalflow
def extract_patches_around_landmarks(image, landmarks, patch_size=8):
    """
    Extract patches around landmarks from an image

    Args:
        image: PIL Image or numpy array (H, W, C)
        landmarks: numpy array of shape (14, 2) - landmark coordinates
        patch_size: int - size of patches to extract (8x8)

    Returns:
        patches: numpy array of shape (14, patch_size, patch_size, C)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    H, W = image.shape[:2]
    C = image.shape[2] if len(image.shape) == 3 else 1

    patches = []
    half_patch = patch_size // 2

    for i, (x, y) in enumerate(landmarks):
        # Convert to integer coordinates
        x, y = int(round(x)), int(round(y))

        # Calculate patch boundaries
        x_min = max(0, x - half_patch)
        x_max = min(W, x + half_patch)
        y_min = max(0, y - half_patch)
        y_max = min(H, y + half_patch)

        # Extract patch
        if len(image.shape) == 3:
            patch = image[y_min:y_max, x_min:x_max, :]
        else:
            patch = image[y_min:y_max, x_min:x_max]

        # Pad if necessary to ensure patch_size x patch_size
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            if len(image.shape) == 3:
                padded_patch = np.zeros((patch_size, patch_size, C), dtype=image.dtype)
                pad_y = (patch_size - patch.shape[0]) // 2
                pad_x = (patch_size - patch.shape[1]) // 2
                padded_patch[
                    pad_y : pad_y + patch.shape[0], pad_x : pad_x + patch.shape[1], :
                ] = patch
            else:
                padded_patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
                pad_y = (patch_size - patch.shape[0]) // 2
                pad_x = (patch_size - patch.shape[1]) // 2
                padded_patch[
                    pad_y : pad_y + patch.shape[0], pad_x : pad_x + patch.shape[1]
                ] = patch
            patch = padded_patch

        patches.append(patch)

    return np.array(patches)


def replicate_patches_over_frames(patches, num_frames=3):
    """
    Replicate patches over multiple frames

    Args:
        patches: numpy array of shape (14, patch_size, patch_size, C)
        num_frames: int - number of frames to replicate over (3)

    Returns:
        replicated_patches: numpy array of shape (num_frames, 14, patch_size, patch_size, C)
    """
    replicated = np.tile(patches[np.newaxis, ...], (num_frames, 1, 1, 1, 1))
    return replicated


def bilinear_temporal_sampling(frames, num_sampling=16):
    """
    frames: torch.Tensor of shape (T, C, H, W)
    Returns: torch.Tensor of shape (num_sampling, C, H, W)
    """
    T = frames.shape[0]
    if T >= num_sampling:
        return frames
    # Prepare for interpolation: (1, C, T, H, W)
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    # Interpolate along the temporal dimension
    frames_upsampled = F.interpolate(
        frames,
        size=(num_sampling, frames.shape[3], frames.shape[4]),
        mode="trilinear",
        align_corners=False,
    )
    # Back to (num_sampling, C, H, W)
    frames_upsampled = frames_upsampled.squeeze(0).permute(1, 0, 2, 3)
    return frames_upsampled


def label_face_region(row):
    # Check for presence of upper and lower face AUs
    upper_aus = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9"]
    lower_aus = ["AU10", "AU12", "AU14", "AU15", "AU17"]
    has_upper = any(row[au] == 1 for au in upper_aus if au in row)
    has_lower = any(row[au] == 1 for au in lower_aus if au in row)
    if has_upper and has_lower:
        return "bothface"
    elif has_upper:
        return "upperface"
    elif has_lower:
        return "lowerface"
    else:
        return "none"


def extract_numbers(string):
    # Find all occurrences of numbers in the string
    return re.findall(r"\d+", string)[0]


def find_jpg_files(root_folder):
    jpg_files = []
    for root, dirs, files in os.walk(root_folder):
        if len(files) > 0:
            files = natsorted(files)
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


def extract_onset_apex_offset_frames(dataset_path, fold_df, i):
    subject = fold_df.iloc[i].subject
    video = fold_df.iloc[i].material
    emote = fold_df.iloc[i].emotion
    dataset = fold_df.iloc[i].dataset
    onset = str(fold_df.iloc[i]["onset"])
    apex = str(fold_df.iloc[i]["apex"])
    offset = str(fold_df.iloc[i]["offset"])

    if dataset == "casme" and "sub" not in subject:
        subject = "sub" + subject
    elif dataset == "casme2" and "sub" not in subject:
        subject = "sub" + subject

    if dataset == "casme3a":
        dataset = "CASME3"
    elif dataset == "fourd":
        dataset = "4DMicro"
    else:
        dataset = dataset.upper()

    if dataset == "CASME":
        # 'reg_EP01_5-113.jpg'
        onset = "{}-{}.jpg".format(video, onset)
        apex = "{}-{}.jpg".format(video, apex)
        offset = "{}-{}.jpg".format(video, offset)
        images_per_vid = find_jpg_files(
            os.path.join(dataset_path, dataset, "Cropped", str(subject), str(video))
        )
    if dataset == "CASME2":
        # reg_img46.jpg
        onset = "reg_img{}.jpg".format(onset)
        apex = "reg_img{}.jpg".format(apex)
        offset = "reg_img{}.jpg".format(offset)
        images_per_vid = find_jpg_files(
            os.path.join(dataset_path, dataset, "Cropped", str(subject), str(video))
        )
    if dataset == "CASME3":
        # 0.jpg
        onset = "{}.jpg".format(onset)
        apex = "{}.jpg".format(apex)
        offset = "{}.jpg".format(offset)
        images_per_vid = find_jpg_files(
            os.path.join(
                dataset_path, dataset, "ME_A_cropped", str(subject), str(video)
            )
        )
    if dataset == "4DMicro":
        # 0.jpg
        onset = "{}.jpg".format(onset)
        apex = "{}.jpg".format(apex)
        offset = "{}.jpg".format(offset)
        images_per_vid = find_jpg_files(
            os.path.join(dataset_path, dataset, "Cropped", str(subject), str(video))
        )
    if dataset == "MMEW":
        # 0.jpg
        onset = "{}.jpg".format(onset)
        apex = "{}.jpg".format(apex)
        offset = "{}.jpg".format(offset)
        images_per_vid = find_jpg_files(
            os.path.join(dataset_path, dataset, "Cropped", emote, str(video))
        )
    if dataset == "SAMM":
        # 0.jpg
        onset = "{}.jpg".format(onset)
        apex = "{}.jpg".format(apex)
        offset = "{}.jpg".format(offset)
        images_per_vid = find_jpg_files(
            os.path.join(dataset_path, dataset, "SAMM_CROP", str(subject), str(video))
        )

    # todo, use df to extract onset apex and offset frames
    try:
        onset_frame = [x for x in images_per_vid if onset in x][0]
        apex_frame = [x for x in images_per_vid if apex in x][0]
        offset_frame = [x for x in images_per_vid if offset in x][0]
    except:
        # cases where naming convention is not consistent in CASME ( 69.jpg and 069.jpg)
        try:
            onset = str(fold_df.iloc[i]["onset"])
            apex = str(fold_df.iloc[i]["apex"])
            offset = str(fold_df.iloc[i]["offset"])
            onset = "{}-{}.jpg".format(video, onset.zfill(3))
            apex = "{}-{}.jpg".format(video, apex.zfill(3))
            offset = "{}-{}.jpg".format(video, offset.zfill(3))
            onset_frame = [x for x in images_per_vid if onset in x][0]
            apex_frame = [x for x in images_per_vid if apex in x][0]
            offset_frame = [x for x in images_per_vid if offset in x][0]
        except:
            a = 1
    return onset_frame, apex_frame, offset_frame
    # return images_per_vid


class PairedDataset(Dataset):
    def __init__(self, datasetA, datasetB, datasetC):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC

        # Ensure both datasets have the same length
        assert len(self.datasetA) == len(
            self.datasetB
        ), "The two datasets must have the same length."
        assert len(self.datasetB) == len(
            self.datasetC
        ), "The two datasets must have the same length."

    def __len__(self):
        return len(self.datasetA)

    def __getitem__(self, index):
        datumA = self.datasetA[index]
        datumB = self.datasetB[index]
        datumC = self.datasetC[index]

        # Return both data as a tuple
        return datumA, datumB, datumC


def normalize_landmarks_to_image_coordinates(landmarks, target_size=224):
    """
    Normalize landmarks from their original coordinate space to [0, target_size] range

    Args:
        landmarks: tensor of shape (batch_size, num_frames, num_landmarks, 2 or 3)
        target_size: target image size (default 224 for standard image models)

    Returns:
        normalized_landmarks: landmarks scaled to [0, target_size] range
    """
    # Get the original shape
    original_shape = landmarks.shape

    # Flatten spatial dimensions for easier processing
    landmarks_flat = landmarks.view(
        -1, original_shape[-1]
    )  # (batch*frames*landmarks, coords)

    # Calculate min and max for each coordinate dimension
    min_coords = landmarks_flat.min(dim=0, keepdim=True)[0]  # (1, coords)
    max_coords = landmarks_flat.max(dim=0, keepdim=True)[0]  # (1, coords)

    # Calculate range for normalization
    coord_range = max_coords - min_coords
    coord_range = torch.clamp(coord_range, min=1e-8)  # Prevent division by zero

    # Normalize to [0, 1] then scale to [0, target_size]
    landmarks_normalized = (landmarks_flat - min_coords) / coord_range
    landmarks_scaled = landmarks_normalized * target_size

    # Reshape back to original shape
    landmarks_scaled = landmarks_scaled.view(original_shape)

    return landmarks_scaled
