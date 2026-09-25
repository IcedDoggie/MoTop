import os

import cv2
import numpy as np
import torch

# from meb import core, datasets, utils
from PIL import Image
from torch.utils.data import Dataset

from datasets_utility import *


class CD6ME_Apex_Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        dataframe,
        test_dataset_fold,
        mode,
        input_type,
        apex_window=None,
        transform=None,
    ):
        # input_type
        # 1. optical_flow
        # 2. rgb
        self.input_type = input_type

        self.dataframe = dataframe
        if (
            "sub"
            not in self.dataframe.loc[
                self.dataframe["dataset"] == ("casme"), "subject"
            ].iloc[0]
        ):
            self.dataframe.loc[self.dataframe["dataset"] == ("casme"), "subject"] = (
                "sub"
                + self.dataframe["subject"].loc[self.dataframe["dataset"] == ("casme")]
            )  # add sub to df for casme
            self.dataframe.loc[self.dataframe["dataset"] == ("casme2"), "subject"] = (
                "sub"
                + self.dataframe["subject"].loc[self.dataframe["dataset"] == ("casme2")]
            )  # add sub to df for casme2
        # labels
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
        self.transform = transform

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
        columns_to_take = basic_columns + aus

        self.dataframe = self.dataframe[columns_to_take]

        self.dataframe["face_region_label"] = self.dataframe.apply(
            label_face_region, axis=1
        )
        # Map face_region_label to numeric values
        # face_region_map = {'upperface': 0, 'lowerface': 1, 'bothface': 2, 'none': 3}
        face_region_map = {"upperface": 0, "lowerface": 0, "none": 0, "bothface": 1}

        self.dataframe["face_region_quantized"] = self.dataframe[
            "face_region_label"
        ].map(face_region_map)
        # Filter out rows where face_region_label is 'none'
        # self.dataframe = self.dataframe[self.dataframe['face_region_label'] != 'none']

        # list of dbs: casme, casme2, casme3a, samm, fourd, mmew
        self.train_fold = self.dataframe.loc[
            self.dataframe["dataset"] != test_dataset_fold
        ]
        self.test_fold = self.dataframe.loc[
            self.dataframe["dataset"] == test_dataset_fold
        ]

        if mode == "train":
            self.fold = self.train_fold
        else:
            self.fold = self.test_fold
        self.mode = mode

        # now we sample the images path first
        self.imgs_list = []
        self.label_list = []
        self.dataset_list = []
        self.emotion_list = []
        self.label_face_region_list = []
        if input_type == "rgb":
            for i in range(len(self.fold)):
                dataset = self.fold.iloc[i].dataset
                subject = self.fold.iloc[i].subject
                video = self.fold.iloc[i].material
                labels = self.fold.iloc[i][aus].values
                apex = self.fold.iloc[i].apex
                onset = self.fold.iloc[i].onset
                face_region = self.fold.iloc[i].face_region_quantized

                if apex_window != None:
                    local_random_state = np.random.RandomState(0)
                    # np.random.seed(0)
                    rand_num_window = 0
                    while rand_num_window == 0:
                        rand_num_window = local_random_state.randint(
                            low=apex_window[0], high=apex_window[1], size=100
                        )  # we need a fix seed but at the same time make sure we don't get the exact apex location
                        rand_num_window = rand_num_window[
                            rand_num_window != 0
                        ]  # this is to remove entries with 0
                        rand_num_window = rand_num_window[
                            0
                        ]  # only take the first one since the above lines meant to remove 0
                    apex += rand_num_window

                if dataset == "casme3a":
                    dataset = "CASME3"
                elif dataset == "fourd":
                    dataset = "4DMicro"
                else:
                    dataset = dataset.upper()

                # images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM' + str(num_frames), str(subject), str(video)))
                images_per_vid = find_jpg_files(
                    os.path.join(
                        dataset_path, dataset, "Cropped", str(subject), str(video)
                    )
                )

                if (
                    len(images_per_vid) == 0
                ):  #  temporary fix for folders that have nested nature.
                    emote = self.fold.iloc[i].emotion
                    # images_per_vid = find_jpg_files(os.path.join(dataset_path, dataset, 'TIM' + str(num_frames), emote, str(video)))
                    images_per_vid = find_jpg_files(
                        os.path.join(
                            dataset_path, dataset, "Cropped", emote, str(video)
                        )
                    )

                # toggling dataset
                if dataset == "CASME":
                    onset = [
                        x
                        for x in images_per_vid
                        if int(onset) == int(x.split("-")[-1].split(".jpg")[0])
                    ]
                    apex = [
                        x
                        for x in images_per_vid
                        if int(apex) == int(x.split("-")[-1].split(".jpg")[0])
                    ]
                    # images_per_vid = [x for x in images_per_vid if int(apex) == int(x.split('-')[-1].split('.jpg')[0])]
                    if len(apex) == 0:
                        apex = self.fold.iloc[i].apex
                        apex = [
                            x
                            for x in images_per_vid
                            if int(apex) == int(x.split("-")[-1].split(".jpg")[0])
                        ]

                elif dataset == "SAMM":
                    onset = [
                        x
                        for x in images_per_vid
                        if onset
                        == int(x.split("/")[-1].split(".jpg")[0].split("_")[-1])
                    ]
                    apex = [
                        x
                        for x in images_per_vid
                        if apex == int(x.split("/")[-1].split(".jpg")[0].split("_")[-1])
                    ]
                    # images_per_vid = [x for x in images_per_vid if int(apex) == int(x.split('-')[-1].split('.jpg')[0])]
                    if len(apex) == 0:
                        apex = self.fold.iloc[i].apex
                        apex = [
                            x
                            for x in images_per_vid
                            if apex
                            == int(x.split("/")[-1].split(".jpg")[0].split("_")[-1])
                        ]

                elif dataset == "4DMicro":
                    onset = [
                        x
                        for x in images_per_vid
                        if onset
                        == int(x.split("/")[-1].split(".jpg")[0].split("_")[-1])
                    ]
                    apex = [
                        x
                        for x in images_per_vid
                        if apex == int(x.split("/")[-1].split(".jpg")[0].split("_")[-1])
                    ]
                    # images_per_vid = [x for x in images_per_vid if int(apex) == int(x.split('-')[-1].split('.jpg')[0])]
                    if len(apex) == 0:
                        apex = self.fold.iloc[i].apex
                        apex = [
                            x
                            for x in images_per_vid
                            if apex
                            == int(x.split("/")[-1].split(".jpg")[0].split("_")[-1])
                        ]

                elif dataset == "CASME2":
                    onset = [
                        x
                        for x in images_per_vid
                        if onset
                        == int(x.split("/")[-1].split(".jpg")[0].split("reg_img")[-1])
                    ]
                    apex = [
                        x
                        for x in images_per_vid
                        if apex
                        == int(x.split("/")[-1].split(".jpg")[0].split("reg_img")[-1])
                    ]
                    # images_per_vid = [x for x in images_per_vid if int(apex) == int(x.split('-')[-1].split('.jpg')[0])]
                    if len(apex) == 0:
                        apex = self.fold.iloc[i].apex
                        apex = [
                            x
                            for x in images_per_vid
                            if apex
                            == int(
                                x.split("/")[-1].split(".jpg")[0].split("reg_img")[-1]
                            )
                        ]

                else:
                    onset = [
                        x
                        for x in images_per_vid
                        if onset == int(x.split("/")[-1].split(".jpg")[0])
                    ]
                    apex = [
                        x
                        for x in images_per_vid
                        if apex == int(x.split("/")[-1].split(".jpg")[0])
                    ]
                    # images_per_vid = [x for x in images_per_vid if int(apex) == int(x.split('-')[-1].split('.jpg')[0])]
                    if len(apex) == 0:
                        apex = self.fold.iloc[i].apex
                        apex = [
                            x
                            for x in images_per_vid
                            if apex == int(x.split("/")[-1].split(".jpg")[0])
                        ]

                # self.imgs_list += [images_per_vid]
                self.imgs_list += [[onset, apex]]
                self.label_list += [labels]
                self.dataset_list += [dataset]
                self.emotion_list += [self.fold.iloc[i].emotion]
                self.label_face_region_list += [self.fold.iloc[i].face_region_quantized]

        if input_type == "magnification":
            for i in range(len(self.fold)):
                dataset = self.fold.iloc[i].dataset
                subject = self.fold.iloc[i].subject
                video = self.fold.iloc[i].material
                labels = self.fold.iloc[i][aus].values
                apex = self.fold.iloc[i].apex

                if dataset == "casme3a":
                    dataset = "CASME3"
                elif dataset == "fourd":
                    dataset = "4DMicro"
                else:
                    dataset = dataset.upper()

                if dataset == "MMEW":
                    emote = self.fold.iloc[i].emotion
                    folder_path = os.path.join(
                        dataset_path, dataset, "mag5_onset-apex", emote, str(video)
                    )
                    images_per_vid = os.listdir(folder_path)
                elif dataset == "CASME3":
                    folder_path = os.path.join(
                        dataset_path,
                        dataset,
                        "mag5_onset-apex",
                        str(subject),
                        str(video),
                        "color",
                    )
                    images_per_vid = os.listdir(folder_path)
                else:
                    folder_path = os.path.join(
                        dataset_path,
                        dataset,
                        "mag5_onset-apex",
                        str(subject),
                        str(video),
                    )
                    images_per_vid = os.listdir(folder_path)

                images_per_vid = [folder_path + "/" + images_per_vid[0]]

                self.imgs_list += [images_per_vid]
                self.label_list += [labels]
                self.dataset_list += [self.fold.iloc[i].dataset]
                self.label_face_region_list += [self.fold.iloc[i].face_region_quantized]

        if self.input_type == "optical_flow":
            dataset = self.fold["dataset"].unique()

            for f in dataset:
                if f == "casme2":
                    optical_flow_filename = "casme2_uv_frames_secrets_of_OF.npy"
                elif f == "casme":
                    optical_flow_filename = "casme_uv_frames_secrets_of_OF.npy"
                elif f == "samm":
                    optical_flow_filename = "samm_uv_frames_secrets_of_OF.npy"
                elif f == "fourd":
                    optical_flow_filename = "4d_uv_frames_secrets_of_OF.npy"
                elif f == "casme3a":
                    optical_flow_filename = "casme3_uv_frames_secrets_of_OF.npy"
                elif f == "mmew":
                    optical_flow_filename = "mmew_uv_frames_secrets_of_OF.npy"

                if f == "casme3a":
                    f = "CASME3"
                elif f == "fourd":
                    f = "4DMicro"
                else:
                    f = f.upper()

                optical_flow_npy = os.path.join(dataset_path, f, optical_flow_filename)
                optical_flow_npy = np.load(optical_flow_npy)
                resized_flow = []

                for img in optical_flow_npy:
                    img = np.transpose(img, (1, 2, 0))  # to H, W, C
                    img = cv2.resize(
                        img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
                    )
                    img = np.transpose(img, (2, 0, 1))  # to C, H, W
                    resized_flow += [img]

                resized_flow = np.stack(resized_flow)

                # # for onset-apex, apex-offset optical flow only
                # if len(resized_flow.shape) == 4:
                #     resized_flow = resized_flow.reshape((len(resized_flow)//2, 2, 3, 224, 224))

                self.imgs_list += [resized_flow]
            for i in range(len(self.fold)):
                labels = self.fold.iloc[i][aus].values
                self.dataset_list += [self.fold.iloc[i].dataset]
                self.label_list += [labels]
                self.emotion_list += [self.fold.iloc[i].emotion]
                self.label_face_region_list += [self.fold.iloc[i].face_region_quantized]

        if self.input_type == "optical_flow":
            if len(self.imgs_list) == 0:
                a = 1
            self.imgs_list = np.vstack(self.imgs_list)
        else:
            # self.imgs_list = np.stack(self.imgs_list)
            self.imgs_list = np.stack(self.imgs_list)[
                :, :, 0
            ]  # for case when we have onset and apex
        self.label_list = np.stack(self.label_list)
        self.dataset_list = np.stack(self.dataset_list)
        self.label_face_region_list = np.stack(self.label_face_region_list)

        a = 1

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if self.input_type == "rgb" or self.input_type == "magnification":
            # img = self.imgs_list[idx][0]
            # img = Image.open(img).convert('RGB')
            # img = self.transform(img)

            # for getting two images
            onset = self.imgs_list[idx][0]
            onset = Image.open(onset).convert("RGB")
            if self.transform != None:
                onset = self.transform(onset)

            apex = self.imgs_list[idx][1]
            apex = Image.open(apex).convert("RGB")
            if self.transform != None:
                apex = self.transform(apex)

            img = [onset, apex]

        elif len(self.imgs_list[idx].shape) == 4:
            img_onset_apex = self.imgs_list[idx][0]
            img_onset_apex = torch.Tensor(img_onset_apex)
            if self.transform != None:
                img_onset_apex = self.transform(img_onset_apex)

            img_apex_offset = self.imgs_list[idx][1]
            img_apex_offset = torch.Tensor(img_apex_offset)
            if self.transform != None:
                img_apex_offset = self.transform(img_apex_offset)

            img = [img_onset_apex, img_apex_offset]

        else:
            img = self.imgs_list[idx]
            img = torch.Tensor(img)
            if self.transform != None:
                img = self.transform(img)

        label = self.label_list[idx]
        label = [int(x) for x in label]
        labels = torch.tensor(label)
        dataset = self.dataset_list[idx]
        filename = self.imgs_list[idx][1]
        emotion = self.emotion_list[idx]
        au_exists = self.fold.iloc[idx]["AU_exists"]
        filtered_au = self.fold.iloc[idx]["filtered_AU"]

        face_region = self.label_face_region_list[idx]
        face_region = torch.tensor(face_region)

        return img, labels, face_region, au_exists, filtered_au
