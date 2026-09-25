from typing import List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict


def rename_module_prefix(state_dict, old_name, new_name):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace(old_name, new_name)
        # encoder.embeddings.patch_embeddings.projection.weight
        # model.videomae.embeddings.patch_embeddings.projection.weight
        new_state_dict[new_key] = value
    return new_state_dict


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total_params: {total_params}, Trainable_params: {trainable_params}")
    return total_params, trainable_params


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_accuracy = float("-inf")
        self.save_weights_flag = False

    def early_stop(self, training_accuracy):
        if training_accuracy > self.min_training_accuracy:
            self.min_training_accuracy = training_accuracy
            self.counter = 0
            self.save_weights_flag = True
        elif training_accuracy < (self.min_training_accuracy - self.min_delta):
            self.counter += 1
            self.save_weights_flag = False
            if self.counter >= self.patience:
                return True
        return False


class MultiLabelF1Score(nn.Module):
    """Multi-label F1-Score

    Computes F1-Score for multi-label cases. First, thresholds the output from a
    network to obtain binary predictions.

    Parameters
    ----------
    average : str, default None
        Determines how the different multi-labels are averaged together. See
        Scikit-learn documentation for more.
    threshold : float, default=0.0
        Determines at which value outputs from a network are thresholded for
        binary outputs.

    """

    def __init__(self, average: Optional[str] = None, threshold: float = 0.0):
        super().__init__()
        self.average = average
        self.threshold = threshold

    def forward(self, labels: torch.Tensor, outputs: torch.Tensor) -> List[float]:
        predictions = torch.where(outputs > self.threshold, 1, 0)
        if self.average is None:
            return f1_score(labels, predictions, average=None)
        # Each label separately to get f1 for each label
        if self.average is not None:
            return [
                f1_score(labels[:, i], predictions[:, i], average=self.average)
                for i in range(labels.shape[-1])
            ]
