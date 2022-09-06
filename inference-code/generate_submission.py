import gc
import itertools
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

infer_df = pd.read_pickle("../outputs/infer_df.pkl")
length_sorted_idx = np.load("../outputs/sorted_idx.npy")


def perform_localization(char_probs, threshold):
    """convert character wise prediction to location spans

    :param char_probs: character wise predictions
    :type char_prob: list
    :param threshold: threshold for label decision, defaults to 0.5
    :type threshold: float, optional
    :return: locations
    :rtype: list
    """
    # print(threshold)
    results = np.where(char_probs >= threshold)[0]
    results = [list(g) for _, g in itertools.groupby(
        results, key=lambda n, c=itertools.count(): n - next(c))]
    results = [[min(r), max(r)+1] for r in results]
    return results


def postprocess_localization(text, span_offsets):
    """remove spaces at the beginning of label span prediction

    :param span: input text (patient history)
    :type text: str
    :param span_offset: prediction span offsets
    :type offset_mapping: list
    :return: updated span offsets 
    :rtype: list
    """
    to_return = []
    for start, end in span_offsets:
        match = list(re.finditer('\S+', text[start:end]))
        if len(match) == 0:
            to_return.append([start, end])
        else:
            span_start, _ = match[0].span()
            to_return.append([start + span_start, end])
    return to_return


def extract_confidence(char_probs, locations):
    results = []
    for start, end in locations:
        results.append(round(np.mean(char_probs[start:end]), 4))
    return results


def compute_annotation_from_location(pn_history, locations):
    all_texts = []
    for char_start_idx, char_end_idx in locations:
        all_texts.append(pn_history[char_start_idx:char_end_idx])
    return all_texts


# DeBERTa Large
preds_del_mpl_1 = pd.read_pickle("../outputs/preds_del_mpl_1.pkl")
preds_del_mpl_2 = pd.read_pickle("../outputs/preds_del_mpl_2.pkl")
preds_del_kd_1 = pd.read_pickle("../outputs/preds_del_kd_1.pkl")
preds_del_kd_2 = pd.read_pickle("../outputs/preds_del_kd_2.pkl")


# DeBERTa XLarge
preds_dexl_mpl_1 = pd.read_pickle("../outputs/preds_dexl_mpl_1.pkl")
preds_dexl_mpl_2 = pd.read_pickle("../outputs/preds_dexl_mpl_2.pkl")

# DeBERTa V2 XLarge
preds_dexlv2_mpl_1 = pd.read_pickle("../outputs/preds_dexlv2_mpl_1.pkl")
preds_dexlv2_sft_1 = pd.read_pickle("../outputs/preds_dexlv2_sft_1.pkl")

# DeBERTa V3 Large
preds_delv3_mpl_1 = pd.read_pickle("../outputs/preds_delv3_mpl_1.pkl")
preds_delv3_mpl_2 = pd.read_pickle("../outputs/preds_delv3_mpl_2.pkl")
preds_delv3_mpl_3 = pd.read_pickle("../outputs/preds_delv3_mpl_3.pkl")
preds_delv3_mpl_4_marked = pd.read_pickle("../outputs/preds_delv3_mpl_4_marked.pkl")
preds_delv3_mpl_5_marked = pd.read_pickle("../outputs/preds_delv3_mpl_5_marked.pkl")

# DeBERTa V3 Large: Public
pred_ext = pd.read_pickle("../outputs/preds_delv3_public.pkl")


def print_line():
    print("=="*40)


df_preds = [
    preds_del_mpl_1,
    preds_del_mpl_2,
    preds_del_kd_1,
    preds_del_kd_2,


    preds_dexl_mpl_1,
    preds_dexl_mpl_2,

    preds_dexlv2_mpl_1,
    preds_dexlv2_sft_1,

    preds_delv3_mpl_1,
    preds_delv3_mpl_2,
    preds_delv3_mpl_3,
    preds_delv3_mpl_4_marked,
    preds_delv3_mpl_5_marked,

    pred_ext[["id", "pred_0"]].rename(columns={"pred_0": "char_probs"}),
    pred_ext[["id", "pred_1"]].rename(columns={"pred_1": "char_probs"}),
    pred_ext[["id", "pred_2"]].rename(columns={"pred_2": "char_probs"}),
    pred_ext[["id", "pred_3"]].rename(columns={"pred_3": "char_probs"}),
    pred_ext[["id", "pred_4"]].rename(columns={"pred_4": "char_probs"}),
]

for th_df in df_preds:
    print_line()
    print(th_df.head())
    print_line()

weights = [
    1.0,  # DEL
    1.0,  # DEL
    1.0,  # DEL
    1.0,  # DEL

    1.0,  # DEXL
    1.0,  # DEXL

    1.0,  # DEXL V2
    1.0,  # DEXL V2

    1.0,  # DEXL V3
    1.0,  # DEXL V3
    1.0,  # DEXL V3
    1.0,  # DEXL V3
    1.0,  # DEXL V3

    1.0,  # EXT
    1.0,  # EXT
    1.0,  # EXT
    1.0,  # EXT
    1.0,  # EXT
]

assert len(weights) == len(df_preds)

total_weight = np.sum(weights)

probs = []
for df in df_preds:
    cur_probs = df['char_probs'].values
    probs.append(cur_probs)

num_examples = len(infer_df)
num_models = len(probs)
mean_scores = []

for i in range(num_examples):
    s = None
    for j in range(num_models):
        if s is not None:
            s += probs[j][i]*weights[j]
        else:
            s = probs[j][i]*weights[j]
    s = s/total_weight
    mean_scores.append(s)

infer_df['char_probs'] = mean_scores


try:
    del pred_ext
    gc.collect()
except:
    pass


infer_df["boxes_at_50"] = infer_df['char_probs'].apply(
    lambda x: perform_localization(x, 0.5)
)

infer_df["boxes_at_50"] = infer_df[['pn_history', 'boxes_at_50']].apply(
    lambda x: postprocess_localization(x[0], x[1]), axis=1
)

infer_df['confidence_at_50'] = infer_df[["char_probs", "boxes_at_50"]].apply(
    lambda x: extract_confidence(x[0], x[1]), axis=1
)


# Post Processing
def _process_2(boxes, annotations, pn_history):
    """
    post processing function for feature 2: chest pressure
    """
    updated_boxes = set()
    boxes = deepcopy(boxes)
    pn_history = pn_history.lower()

    if len(boxes) == 1:
        if (annotations[0].lower().strip() == "chest pain") | (annotations[0].lower().strip() == "pressure"):
            return []

    for box, anno in zip(boxes, annotations):
        start, end = box

        text_segment = pn_history[start:end]
        # remove pain
        match = list(re.finditer(' pain', text_segment))

        if len(match) > 0:  # HA -> Heart attack
            span_start, span_end = match[0].span()

            new_box_1 = [start, start + span_start]
            new_box_2 = [start+span_end, end]

            updated_boxes.add(tuple(new_box_1))
            updated_boxes.add(tuple(new_box_2))
        else:
            updated_boxes.add(tuple(box))

    updated_boxes = sorted([list(b) for b in updated_boxes], key=lambda x: x[0])
    updated_boxes = list(filter(lambda x: x[0] != x[1], updated_boxes))

    return updated_boxes


def _process_10(boxes, annotations):
    """
    post processing function for feature 10 # past few months
    """
    remove_kws = ["week", "year", "past month", "1 month", "one month"]

    filtered_boxes = set()
    boxes = deepcopy(boxes)
    for box, anno in zip(boxes, annotations):
        anno = anno.lower()
        for kw in remove_kws:
            if kw in anno:
                break
            else:
                filtered_boxes.add(tuple(box))
    filtered_boxes = sorted([list(b) for b in filtered_boxes], key=lambda x: x[0])

    return filtered_boxes


def _process_303(boxes, annotations):
    """
    post processing function for feature 303
    """
    remove_kws = ["pain killer", "analgesic", "voltaren", "pain meds"]
    filtered_boxes = set()
    boxes = deepcopy(boxes)
    for box, anno in zip(boxes, annotations):
        anno = anno.lower()
        for kw in remove_kws:
            if kw in anno:
                break
            else:
                filtered_boxes.add(tuple(box))
    filtered_boxes = sorted([list(b) for b in filtered_boxes], key=lambda x: x[0])

    return filtered_boxes


def _process_309(boxes, annotations):
    """
    post processing function for feature 309
    """
    boxes = deepcopy(boxes)

    remove_kws = ["2 weeks", "2weeks"]
    filtered_boxes = set()
    boxes = deepcopy(boxes)
    for box, anno in zip(boxes, annotations):
        anno = anno.lower()
        for kw in remove_kws:
            if kw in anno:
                break
            else:
                filtered_boxes.add(tuple(box))
    filtered_boxes = sorted([list(b) for b in filtered_boxes], key=lambda x: x[0])
    return filtered_boxes


def _process_311(boxes, annotations):
    """
    post processing function for feature 303
    """
    boxes = deepcopy(boxes)
    if len(boxes) == 1:
        if annotations[0].lower() == "denies":
            return []
    return boxes


def post_process(feature_num, boxes, annotations, pn_history):
    try:
        if len(boxes) == 0:
            return boxes

        if feature_num == 2:
            return _process_2(boxes, annotations, pn_history)

        if feature_num == 10:
            return _process_10(boxes, annotations)

        if feature_num == 303:
            return _process_303(boxes, annotations)

        if feature_num == 309:
            return _process_309(boxes, annotations)

        if feature_num == 311:
            return _process_311(boxes, annotations)
        else:
            return boxes

    except Exception as e:
        print(e)
        return boxes


infer_df['annotation_pred'] = infer_df[['pn_history', 'boxes_at_50']].apply(
    lambda x: compute_annotation_from_location(x[0], x[1]), axis=1
)

infer_df["filtered_boxes"] = infer_df[["feature_num", "boxes_at_50", "annotation_pred", "pn_history"]].apply(
    lambda x: post_process(x[0], x[1], x[2], x[3]), axis=1
)
infer_df["location_preds_str"] = infer_df["filtered_boxes"].apply(lambda x: ";".join([f"{elem[0]} {elem[1]}" for elem in x]))

infer_df = infer_df.rename(columns={'location_preds_str': 'location'})
infer_df = infer_df.iloc[length_sorted_idx].copy()
print(infer_df.head())
submission_df = infer_df[["id", "location"]].reset_index(drop=True)
print(submission_df.head())

# save submission
settings_path = str(Path(__file__).parents[1] / "settings.json")
with open(settings_path, "r") as f:
    root_config = json.load(f)

sub_dir = root_config["SUBMISSION_DIR"]
submission_df.to_csv(os.path.join(sub_dir, "submission.csv"), index=False)
#----------- DONE ------------------------#
