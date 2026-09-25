import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def pred_emotion_from_au(pred, emotions, emotion_to_aus, au_to_idx, dataset):
    predicted_emotions = []
    predicted_indices = []

    for row in pred:
        # compute score per emotion as sum of predicted AU presence for that emotion's AU set
        scores = []
        for emo in emotions:
            emo_aus = emotion_to_aus.get(emo, [])
            idxs = [au_to_idx[a] for a in emo_aus if a in au_to_idx]
            if len(idxs) == 0:
                scores.append(0)
            else:
                # handle non-binary probabilities by summing; works for binary or soft outputs
                scores.append(float(row[idxs].sum()))

        # choose best emotion (break ties by order in emotions)
        best_idx = int(np.argmax(scores))
        # if all scores zero, assign 'others'
        if sum(scores) == 0:
            if dataset == "casme2":
                best_idx = emotions.index("others")
            elif dataset == "samm":
                best_idx = emotions.index("other")

        predicted_indices.append(best_idx)
        predicted_emotions.append(emotions[best_idx])

    predicted_emotions = np.array(predicted_emotions)
    predicted_indices = np.array(predicted_indices)
    return predicted_indices, predicted_emotions


def au_emotion_recognizer(dataset, pred_aus):
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
    casme2_emotions = ["happiness", "disgust", "repression", "surprise", "others"]
    samm_emotions = ["anger", "happiness", "contempt", "surprise", "other"]
    casme3_emotions = [
        "happiness",
        "disgust",
        "fear",
        "anger",
        "sadness",
        "surprise",
        "others",
    ]
    fourd_emotions = ["negative", "positive", "surprise", "repression", "others"]
    mmew_emotions = [
        "happiness",
        "surprise",
        "anger",
        "disgust",
        "fear",
        "sadness",
        "others",
    ]
    smic_emotions = ["positive", "negative", "surprise"]

    if dataset == "casme2" or dataset == "casme":
        emotions = casme2_emotions
    elif dataset == "samm":
        emotions = samm_emotions
    elif dataset == "casme3a":
        emotions = casme3_emotions
    elif dataset == "fourd":
        emotions = fourd_emotions
    elif dataset == "mmew":
        emotions = mmew_emotions
    elif dataset == "smic":
        emotions = smic_emotions

    if dataset == "smic":
        df = pd.read_excel("/home/hq/Documents/data/SMIC/smic.xlsx")
        emotion_gt = df["emotion"].values
    else:
        df = pd.read_csv("metadata_csv/metadata_csv_statistics.csv")
        # casme2
        # dataset = 'casme2'
        dataset_fold = df.loc[df["dataset"] == dataset].reset_index()
        dataset_fold["emotion"] = dataset_fold[
            "emotion"
        ].str.lower()  # ensure emotion labels are lowercase for matching
        data_entry_to_omit = dataset_fold.loc[
            ~dataset_fold["emotion"].isin(emotions)
        ].index.values
        dataset_fold = dataset_fold.loc[dataset_fold["emotion"].isin(emotions)]
        emotion_gt = dataset_fold["emotion"].values
        # drop from pred_aus
        pred_aus = np.delete(pred_aus, data_entry_to_omit, axis=0)

    # utilises aus to predict emotions based on rules derived from the metadata statistics
    # based on casme2 table
    if dataset == "casme2" or dataset == "casme":
        happiness_aus = ["AU6", "AU12"]
        disgust_aus = ["AU9", "AU10", "AU14"]
        repression_aus = ["AU14", "AU15", "AU17"]
        surprise_aus = ["AU1", "AU2"]
        others_aus = ["AU4", "AU5", "AU7"]
        # map emotion -> AU list
        emotion_to_aus = {
            "happiness": happiness_aus,
            "disgust": disgust_aus,
            "repression": repression_aus,
            "surprise": surprise_aus,
            "others": others_aus,
        }

    elif dataset == "samm":
        happiness_aus = ["AU6", "AU12", "AU15"]
        surprise_aus = ["AU1", "AU2"]
        anger_aus = ["AU4", "AU7"]
        contempt_aus = ["AU14", "AU17"]
        others_aus = ["AU5", "AU9", "AU10"]
        # map emotion -> AU list
        emotion_to_aus = {
            "happiness": happiness_aus,
            "anger": anger_aus,
            "contempt": contempt_aus,
            "surprise": surprise_aus,
            "other": others_aus,
        }
    elif dataset == "casme3a":
        happiness_aus = ["AU6", "AU12"]
        disgust_aus = ["AU9", "AU10"]
        fear_aus = ["AU14"]
        anger_aus = ["AU4", "AU7"]
        sadness_aus = ["AU17"]
        surprise_aus = ["AU1", "AU2"]
        others_aus = ["AU5", "AU15"]
        # map emotion -> AU list
        emotion_to_aus = {
            "happiness": happiness_aus,
            "disgust": disgust_aus,
            "fear": fear_aus,
            "anger": anger_aus,
            "sadness": sadness_aus,
            "surprise": surprise_aus,
            "others": others_aus,
        }

    elif dataset == "fourd":
        positive_aus = ["AU12"]
        negative_aus = ["AU1", "AU4", "AU7", "AU9", "AU10"]
        surprise_aus = ["AU2"]
        repression_aus = ["AU14", "AU15", "AU17"]
        others_aus = ["AU5", "AU6"]
        # map emotion -> AU list
        emotion_to_aus = {
            "positive": positive_aus,
            "negative": negative_aus,
            "surprise": surprise_aus,
            "repression": repression_aus,
            "others": others_aus,
        }
    elif dataset == "mmew":
        happiness_aus = ["AU6", "AU12"]
        surprise_aus = ["AU1", "AU2", "AU5"]
        anger_aus = ["AU7", "AU9"]
        disgust_aus = ["AU4", "AU10"]
        fear_aus = ["AU14"]
        sadness_aus = ["AU17"]
        others_aus = ["AU15"]
        # map emotion -> AU list
        emotion_to_aus = {
            "happiness": happiness_aus,
            "surprise": surprise_aus,
            "anger": anger_aus,
            "disgust": disgust_aus,
            "fear": fear_aus,
            "sadness": sadness_aus,
            "others": others_aus,
        }

    elif dataset == "smic":
        positive_aus = ["AU6", "AU12"]
        negative_aus = ["AU4", "AU5", "AU7", "AU9", "AU10", "AU14", "AU15", "AU17"]
        surprise_aus = ["AU1", "AU2"]
        # map emotion -> AU list
        emotion_to_aus = {
            "positive": positive_aus,
            "negative": negative_aus,
            "surprise": surprise_aus,
        }

    # map AU name -> column index in pred_aus (assumes order matches `aus` list)
    au_to_idx = {au: i for i, au in enumerate(aus)}

    # ensure pred_aus is 2D (N, num_aus)
    pred = np.asarray(pred_aus)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]

    _, predicted_emotions = pred_emotion_from_au(
        pred, emotions, emotion_to_aus, au_to_idx, dataset
    )

    acc = accuracy_score(emotion_gt, predicted_emotions)

    f1 = f1_score(emotion_gt, predicted_emotions, average="macro")

    # cm = confusion_matrix(emotion_gt, predicted_emotions, normalize='true')
    # cm_plot = ConfusionMatrixDisplay(cm, display_labels=np.unique(emotion_gt))
    # cm_plot.plot()
    a = 1

    print(f"Dataset: {dataset}")
    print(f"Accuracy: {acc}")
    print(f"Macro F1 score: {f1}")

    return acc, f1, emotion_gt


def au_emotion_recognizer_no_gt(dataset, pred_aus, granularity="coarse"):
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
    smic_emotions = ["positive", "negative", "surprise"]
    casme2_emotions = ["happiness", "disgust", "repression", "surprise", "others"]

    if dataset == "smic":
        emotions = smic_emotions
    if dataset == "smic":
        df = pd.read_excel("/home/hq/Documents/data/SMIC/smic.xlsx")

    if granularity == "coarse":
        emotions = smic_emotions
        positive_aus = ["AU6", "AU12"]
        negative_aus = ["AU4", "AU5", "AU7", "AU9", "AU10", "AU14", "AU15", "AU17"]
        surprise_aus = ["AU1", "AU2"]
        # map emotion -> AU list
        emotion_to_aus = {
            "positive": positive_aus,
            "negative": negative_aus,
            "surprise": surprise_aus,
        }
    else:
        emotions = casme2_emotions
        happiness_aus = ["AU6", "AU12"]
        disgust_aus = ["AU9", "AU10", "AU14"]
        repression_aus = ["AU14", "AU15", "AU17"]
        surprise_aus = ["AU1", "AU2"]
        others_aus = ["AU4", "AU5", "AU7"]
        # map emotion -> AU list
        emotion_to_aus = {
            "happiness": happiness_aus,
            "disgust": disgust_aus,
            "repression": repression_aus,
            "surprise": surprise_aus,
            "others": others_aus,
        }

    # map AU name -> column index in pred_aus (assumes order matches `aus` list)
    au_to_idx = {au: i for i, au in enumerate(aus)}

    # ensure pred_aus is 2D (N, num_aus)
    pred = np.asarray(pred_aus)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    _, predicted_emotions = pred_emotion_from_au(
        pred, emotions, emotion_to_aus, au_to_idx, dataset
    )

    return predicted_emotions
