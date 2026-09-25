import collections
import logging
import re
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from datasets_utility import *




@dataclass
class DatasetConfig:
    """Configuration class for dataset parameters"""

    dataset_path: str
    num_frames: int
    test_dataset_fold: str
    mode: str
    data_type: str
    transform: Optional[callable] = None
    mag_factor: int = 3
    sampling_rate: int = 16


def frame_distance_visualisation(frame_count_list, dbs=None, max_per_db_display=None):
    """
    Visualize frame distances grouped by dataset string (dbs).
    - frame_count_list: list/array of shape (N, 2) per sample (apex-onset, offset-apex)
    - dbs: optional list of dataset identifiers (len == N); if None, all samples grouped under "all"
    - max_per_db_display: limit points plotted per db for performance
    """
    import math

    frame_counts = np.vstack(frame_count_list)  # shape (N, 2)
    n = frame_counts.shape[0]

    if dbs is None:
        dbs = ["all"] * n
    dbs = np.array(dbs)

    unique_dbs = np.unique(dbs)
    n_plots = len(unique_dbs)
    ncols = 2
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )
    axes = axes.flatten()

    for i, db in enumerate(unique_dbs):
        idxs = np.where(dbs == db)[0]
        if idxs.size == 0:
            continue

        counts_db = frame_counts[idxs]
        # limit number of points for readability
        if max_per_db_display is not None and counts_db.shape[0] > max_per_db_display:
            rng = np.linspace(0, counts_db.shape[0] - 1, max_per_db_display, dtype=int)
            counts_db = counts_db[rng]
            plot_x = np.arange(len(rng))
        else:
            plot_x = np.arange(counts_db.shape[0])

        ax = axes[i]
        # ax.scatter(plot_x, counts_db[:, 0], c="red", label="apex - onset", s=20, alpha=0.8)
        # if second diff available, plot it
        if counts_db.shape[1] > 1:
            ax.scatter(
                plot_x,
                counts_db[:, 1],
                c="blue",
                label="offset - apex",
                s=20,
                alpha=0.8,
            )

        ax.set_title(f"{db} (n={len(idxs)})")
        ax.set_xlabel("Sample index (per-db)")
        ax.set_ylabel("Frame distance (frames)")
        ax.grid(alpha=0.25)
        ax.legend()

    # hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def standardization(values):
    x_mean = values.mean(dim=0, keepdim=True)  # Mean across all nodes
    x_std = values.std(dim=0, keepdim=True)  # Std across all nodes
    x_std = torch.clamp(x_std, min=1e-8)  # Prevent division by zero
    standardized_values = (values - x_mean) / x_std
    return standardized_values


class BaseSequenceDataset(Dataset, ABC):
    """Abstract base class for all sequence datasets"""

    def __init__(self, config: DatasetConfig, dataframe):
        self.config = config
        self.dataframe = self._prepare_dataframe(dataframe)
        self._processed_emotions = self._process_and_map_emotions()
        self.dataframe["processed_emotion"] = self._processed_emotions
        self.fold = self._get_fold()
        self.logger = logging.getLogger(self.__class__.__name__)


        a = 1

    def _process_and_map_emotions(self):
        emotion = self.dataframe["emotion"].str.lower()
        emotion_mapping = {
            "tense": "negative",
            "happiness": "positive",
            "repression": "negative",
            "disgust": "negative",
            "surprise": "surprise",
            "contempt": "negative",
            "comtempt": "negative",
            "fear": "negative",
            "sadness": "negative",
            "others": "others",
            "anger": "negative",
            "other": "others",
            "contempt": "negative",
            "negative": "negative",
            "surprise+repression": "surprise",
            "surprise+positive": "surprise",
            "positive": "positive",
            "surprise+negative": "surprise",
            "positive+repression": "positive",
            "negative+repression": "negative",
            "happy": "positive",
            "sad": "negative",
        }
        emotion = emotion.map(emotion_mapping)
        a = 1
        return emotion

    def _prepare_dataframe(self, dataframe):
        """Common dataframe preprocessing"""
        df = dataframe.copy()

        # Add 'sub' prefix for CASME datasets
        df.loc[df["dataset"] == "casme", "subject"] = (
            "sub" + df["subject"].loc[df["dataset"] == "casme"]
        )
        df.loc[df["dataset"] == "casme2", "subject"] = (
            "sub" + df["subject"].loc[df["dataset"] == "casme2"]
        )

        # Select relevant columns
        aus = [
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
        ]
        basic_columns = [
            "subject",
            "material",
            "onset",
            "apex",
            "offset",
            "apexf",
            "AU",
            "emotion",
            "n_frames",
            "dataset",
            "filtered_AU",
            "AU_exists",
        ]
        df = df[basic_columns + aus]

        # Add face region labels
        df["face_region_label"] = df.apply(label_face_region, axis=1)
        face_region_map = {"oneface": 0, "bothface": 1}
        df["face_region_quantized"] = df["face_region_label"].map(face_region_map)

        return df

    def _get_fold(self):
        """Get train/test fold based on mode"""
        if self.config.mode == "train":
            return self.dataframe.loc[
                self.dataframe["dataset"] != self.config.test_dataset_fold
            ]

        else:
            return self.dataframe.loc[
                self.dataframe["dataset"] == self.config.test_dataset_fold
            ]

    def _convert_to_landmark_paths(
        self, landmarks_names, folder_renamed, file_ext=".npy"
    ):
        """Convert video frame paths to corresponding landmark file paths"""
        # landmark_mag = "mag" + str(self.config.mag_factor)
        landmark_mag = ""
        if "SAMM" in landmarks_names[0]:
            landmarks_names = [
                x.replace("SAMM_CROP", landmark_mag + folder_renamed)
                for x in landmarks_names
            ]
            landmarks_names = [x.replace(".jpg", file_ext) for x in landmarks_names]
        elif "CASME3" in landmarks_names[0]:
            landmarks_names = [
                x.replace("ME_A_cropped", landmark_mag + folder_renamed)
                for x in landmarks_names
            ]
            landmarks_names = [x.replace(".jpg", file_ext) for x in landmarks_names]
        else:
            landmarks_names = [
                x.replace("Cropped", landmark_mag + folder_renamed)
                for x in landmarks_names
            ]
            landmarks_names = [x.replace(".jpg", file_ext) for x in landmarks_names]
        return landmarks_names

    # @abstractmethod
    def _load_data(self) -> Tuple[List, List]:
        """Load data specific to dataset type"""
        vid_list = []
        label_list = []
        frame_count_list = []

        for i in range(len(self.fold)):
            frames = extract_onset_apex_offset_frames(
                self.config.dataset_path, self.fold, i
            )
            frame_count = []
            for s in frames:
                s_file = s.split("/")[-1]
                if "CASME3" in s:
                    a = 1
                if "CASME" in s:
                    frame_count += [int(re.findall(r"\d+", str(s_file))[-1])]
                elif "SAMM" in s:
                    frame_count += [int(re.findall(r"\d+", str(s_file))[-1])]
                else:
                    frame_count += [int(re.findall(r"\d+", str(s_file))[0])]
            frame_count = np.diff(frame_count)

            labels = self.fold.iloc[i][self._get_aus()].values

            vid_list.append(frames)
            label_list.append(labels)
            frame_count_list.append(frame_count)

        return vid_list, label_list, frame_count_list

    def __len__(self):
        return len(self.vid_list)


class RGBDataset(BaseSequenceDataset):
    """Dataset for RGB video data"""

    def __init__(self, config: DatasetConfig, dataframe):
        super().__init__(config, dataframe)
        self.vid_list, self.label_list, self.frame_count_list = self._load_data()
        self.transform = self.config.transform

    def _get_aus(self):
        return [
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
        ]

    def __getitem__(self, idx):
        imgs = self.vid_list[idx]

        # numpy
        imgs_arr = [
            np.array(Image.open(img).convert("RGB").resize((224, 224))) for img in imgs
        ][0]

        imgs = [self.transform(Image.open(img).convert("RGB")) for img in imgs]
        img = torch.stack(imgs)  # (T, C, H, W)
        label = self.label_list[idx]
        label = [int(x) for x in label]
        label = torch.tensor(label)
        return img, label, imgs_arr

    def __len__(self):
        return len(self.vid_list)


class LandmarksDataset(BaseSequenceDataset):
    """Dataset for 2D landmarks data"""

    def __init__(self, config: DatasetConfig, dataframe):
        super().__init__(config, dataframe)
        self.vid_list, self.label_list, self.frame_count_list = self._load_data()

        dbs = [vid[0].split("/")[5] for vid in self.vid_list]

        # frame_distance_visualisation(self.frame_count_list, dbs=dbs)

        self.transform = self.config.transform
        statistical_edge_flag = True
        self.filtered_au = self.dataframe["filtered_AU"].values

        # Add AU co-occurrence edges here if needed
        # adding co-occurring AUs edges
        self.au_co_occurring_edges = []
        for au in self.filtered_au:
            if au is not None and "+" in au:
                landmark_aus = au.split("+")
                for landmark_au in landmark_aus:
                    landmark_aus_location = self._aus_to_landmark_indices(landmark_au)
                    if "L" in landmark_au:
                        landmark_au = landmark_au.replace("L", "")
                    if "R" in landmark_au:
                        landmark_au = landmark_au.replace("R", "")
                    au_landmarks = landmark_aus_location[landmark_au]
                # Create all possible connections between landmarks for this AU
                for i in range(len(au_landmarks)):
                    for j in range(i + 1, len(au_landmarks)):
                        self.au_co_occurring_edges.append(
                            [au_landmarks[i], au_landmarks[j]]
                        )
        self.au_co_occurring_edges = np.vstack(self.au_co_occurring_edges)
        self.au_co_occurring_edges = np.unique(self.au_co_occurring_edges, axis=0)

    def _get_aus(self):
        return [
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
        ]

    def __getitem__(self, idx):
        return self._process_landmarks_sample(idx)

    def _load_landmarks(self, landmarks_paths):
        """Load landmarks from given paths"""
        images = [np.load(img, allow_pickle=True) for img in landmarks_paths]
        images = np.stack(images)
        # images[:, :, 1] = images[:, :, 1].max() - images[:, :, 1]  # flip y axis
        # filter off nodes from face boundary
        images = images[:, 17:, :]

        selected_indices = []
        omitted_indices = []
        cluster_indices = []

        # Left eyebrow: select 3 points (start, middle, end)
        left_brow_indices = [0, 2, 4]  # Corresponds to original [17, 19, 21]
        selected_indices.extend(left_brow_indices)

        # Right eyebrow: select 3 points (start, middle, end)
        right_brow_indices = [5, 7, 9]  # Corresponds to original [22, 24, 26]
        selected_indices.extend(right_brow_indices)

        # Lower nose tip: select 3 points
        nose_tip_indices = [14, 16, 18]  # Corresponds to original [31, 32, 33]
        selected_indices.extend(nose_tip_indices)

        # # Mouth boundary: select 5 points (corners and key points)
        mouth_indices = [31, 40, 37, 32, 36]
        # mouth_indices = [31, 34, 37, 40, 43]  # Corresponds to original [48, 51, 54, 57, 60]
        # # Alternative mouth selection for better boundary representation:
        # # mouth_indices = [31, 33, 37, 41, 43]  # left corner, top, center, bottom, right corner
        selected_indices.extend(mouth_indices)

        # nose bridge indices
        nose_bridge_indices = [10, 11, 12, 13]  # Corresponds to
        omitted_indices.extend(nose_bridge_indices)
        # only remove nose bridge
        # only remove nose bridge - create boolean mask
        keep_mask = np.ones(images.shape[1], dtype=bool)
        keep_mask[omitted_indices] = False

        # Extract selected landmarks
        landmarks_14 = images[:, selected_indices, :]
        landmarks_51 = images.copy()
        landmarks_47 = images[:, keep_mask, :]

        # cluster indices for 47 landmarks
        cluster_indices = [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
        ]
        # cluster_indices = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
        #                    7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        # cluster_indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        #                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

        images = landmarks_47
        a = 1
        return images, cluster_indices

    def _aus_to_landmark_indices(self, aus: List[str]) -> List[int]:
        aus_landmark_location = {
            "1": [3, 4, 5, 6],
            "2": [0, 1, 8, 9, 2, 7],
            "4": [3, 4, 5, 6],
            "5": [15, 16, 17, 18, 21, 22, 23, 24],
            "6": [19, 20, 25, 26],
            "7": [19, 20, 25, 26],
            "9": [10, 11, 12, 13, 14],
            "10": [28, 40, 41, 42, 32, 29, 30, 31, 44, 45, 46],
            "12": [27, 39, 43, 33],
            "14": [27, 39, 43, 33],
            "15": [27, 39, 43, 33],
            "17": [38, 37, 36, 35, 34],
        }
        return aus_landmark_location

    def _extract_patches(self, landmarks_names, landmarks):
        """Extract patches around landmarks from original images"""
        landmark_mag = "mag" + str(self.config.mag_factor)
        # Load corresponding RGB images for patch extraction
        if "SAMM" in landmarks_names[0]:
            rgb_image_paths = [
                x.replace(landmark_mag + "_Cropped_Landmarks", "SAMM_CROP").replace(
                    ".npy", ".jpg"
                )
                for x in landmarks_names
            ]
        elif "CASME3" in landmarks_names[0]:
            rgb_image_paths = [
                x.replace(landmark_mag + "_Cropped_Landmarks", "ME_A_cropped").replace(
                    ".npy", ".jpg"
                )
                for x in landmarks_names
            ]
        else:
            rgb_image_paths = [
                x.replace(landmark_mag + "_Cropped_Landmarks", "Cropped").replace(
                    ".npy", ".jpg"
                )
                for x in landmarks_names
            ]

        # Extract patches around landmarks for each frame
        all_patches = []
        for frame_idx, (rgb_path, landmarks_2d) in enumerate(
            zip(rgb_image_paths, landmarks)
        ):
            # Load RGB image
            rgb_image = Image.open(rgb_path).convert("RGB")
            rgb_array = np.array(rgb_image)

            # Extract 8x8 patches around landmarks
            patches = extract_patches_around_landmarks(
                rgb_array, landmarks_2d, patch_size=64
            )
            transformed_patches = []
            for p in range(patches.shape[0]):
                patch = Image.fromarray(patches[p])
                patch = self.transform(patch)
                transformed_patches += [patch]
            transformed_patches = np.stack(transformed_patches)
            all_patches.append(transformed_patches)

        all_patches = np.array(all_patches)  # Shape: (3, 14, 8, 8, 3)
        return all_patches

    def _14_landmark_edges(self):
        landmark_edges = np.array(
            [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [12, 9],
                [9, 10],
                [10, 11],
                [11, 13],
                [2, 7],
                [3, 7],
                [7, 12],
                [7, 13],
            ]
        )
        return landmark_edges

    def _51_landmark_edges(self):
        landmark_edges = np.array(
            [
                # Add edges for 51 landmarks here
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [15, 16],
                [16, 17],
                [17, 18],
                [15, 20],
                [20, 19],
                [19, 18],
                [21, 22],
                [22, 23],
                [23, 24],
                [21, 26],
                [26, 25],
                [25, 24],
                [4, 10],
                [5, 14],
                [10, 11],
                [11, 12],
                [12, 13],
                [13, 14],
                [12, 29],
                [12, 31],
                [29, 30],
                [30, 31],
                [28, 40],
                [40, 41],
                [41, 42],
                [42, 32],
                [29, 28],
                [28, 39],
                [39, 27],
                [27, 38],
                [38, 37],
                [37, 36],
                [36, 35],
                [35, 34],
                [34, 44],
                [44, 45],
                [45, 46],
                [46, 38],
                [34, 33],
                [33, 43],
                [43, 32],
                [32, 31],
            ]
        )
        return landmark_edges

    def _51_landmark_edges_with_au_co_occurrence(self):
        landmark_edges = self._51_landmark_edges()
        landmark_edges = np.vstack([landmark_edges, self.au_co_occurring_edges])

        return landmark_edges

    def _landmarks_with_au_co_occurrence_edges(self):
        landmark_edges = self.au_co_occurring_edges
        return landmark_edges

    def _get_all_possible_edges(self):
        edges = []
        one_state = self.ref_landmark
        for i in range(len(one_state)):
            central_node = i
            for j in range(len(one_state) - 1):
                if j != central_node:
                    # Bidirectional edges: central -> node and node -> central
                    edges.append([central_node, j])
                    # edges.append([j, central_node])
        landmark_edges = edges
        landmark_edges = np.vstack(landmark_edges)

        return landmark_edges

    def _generate_edges(self):
        """Generate edges based on landmark connections"""
        edges = []
        # specific edges
        # landmark_edges = self._14_landmark_edges()
        landmark_edges = self._51_landmark_edges()
        # landmark_edges = self._51_landmark_edges_with_au_co_occurrence()
        # landmark_edges = self._landmarks_with_au_co_occurrence_edges()
        # landmark_edges = self._get_all_possible_edges()

        edges.append(landmark_edges.T)
        edges = torch.Tensor(edges[0])

        return edges

    def _generate_incidence_matrices(self, edges):
        N = 47  # number of nodes
        incidence_matrices = []
        hyperedges = []
        adj = collections.defaultdict(set)

        for u, v in edges.T:
            adj[u].add(v)
            # adj[v].add(u) # this is needed if bidirectional

        min_degree = 0  # hyperedge if node connects to ≥ 3 others
        hub_nodes = [i for i in adj if len(adj[i]) >= min_degree]

        for i in hub_nodes:
            he = {i} | adj[i]  # center + neighbors
            hyperedges.append(list(he))
        hyperedges = list(map(list, set(tuple(sorted(he)) for he in hyperedges)))
        H = torch.zeros(N, len(hyperedges))

        for e, nodes in enumerate(hyperedges):
            nodes = [int(n) for n in nodes]
            H[nodes, e] = 1

        hyperedges_2dform = []
        N, num_hyperedges = H.shape
        # Create all pairwise connections within this hyperedge
        for he_idx in range(num_hyperedges):
            # Get nodes that belong to this hyperedge
            nodes_in_hyperedge = torch.where(H[:, he_idx] == 1)[0]

            if len(nodes_in_hyperedge) >= 2:
                # Create all pairwise connections within this hyperedge
                for i in range(len(nodes_in_hyperedge)):
                    for j in range(i + 1, len(nodes_in_hyperedge)):
                        node_i = nodes_in_hyperedge[i].item()
                        node_j = nodes_in_hyperedge[j].item()
                        hyperedges_2dform.append([node_i, node_j])

        hyperedges_2dform = np.stack(hyperedges_2dform).T

        return H, hyperedges_2dform

    def _emotion_discrete(self, emotion):
        emotion_map = {
            "negative": 0,
            "positive": 1,
            "surprise": 2,
            "others": 3,
        }
        return emotion_map.get(emotion, 3)  # Default to 'others' if not found

    def _process_landmarks_sample(self, idx):
        """Process landmarks sample with patches extraction"""
        landmarks_names = self.vid_list[idx]
        # only get onset and apex
        # landmarks_names = landmarks_names[0:2]

        label = self.label_list[idx]
        label = [int(x) for x in label]
        au_exists = self.fold.iloc[idx]["AU_exists"]
        self.filtered_au = self.fold.iloc[idx]["filtered_AU"]
        emotion = self.fold.iloc[idx]["processed_emotion"]
        emotion = self._emotion_discrete(emotion)

        # Convert paths to landmark files
        landmarks_paths = self._convert_to_landmark_paths(
            landmarks_names, folder_renamed="Cropped_Landmarks", file_ext=".npy"
        )
        # landmarks_paths = self._convert_to_landmark_paths(
        #     landmarks_names, folder_renamed="_Cropped_Landmarks_fullseq", file_ext=".npy"
        # )
        # landmarks_paths = self._convert_to_landmark_paths(landmarks_names)

        # Load landmarks
        landmarks, cluster_indices = self._load_landmarks(landmarks_paths)
        # # flipping landmarks
        # # FLIP LANDMARKS VERTICALLY - Add this line
        # landmarks[:, :, 1] = landmarks[:, :, 1].max() - landmarks[:, :, 1]  # flip y axis

        self.ref_landmark = landmarks[0]
        # scale = 1
        scale = 2
        # scale = 5
        # scale = 10
        landmarks_diff = []
        for i in range(len(landmarks) - 1):
            idx_0 = landmarks[i]
            idx_1 = landmarks[i + 1]
            diff = idx_0 + scale * (idx_1 - idx_0)
            landmarks_diff.append(diff)
        landmarks_diff = np.array(landmarks_diff)
        landmarks = landmarks_diff

        # Generate edges
        edges = self._generate_edges()
        extra_edges = self.au_co_occurring_edges.T

        # batch information for pooling in batches
        facial_aware_indices = [x + idx for x in cluster_indices]
        # facial_aware_indices = [idx]
        a = 1

        return (
            torch.tensor(landmarks),
            torch.tensor(label),
            torch.tensor(edges),
            torch.tensor(emotion),
            torch.tensor(cluster_indices),
            torch.tensor(facial_aware_indices),
            # [landmarks_names[0]]
            # torch.tensor(extra_edges),
        )


