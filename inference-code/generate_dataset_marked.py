import argparse
import gc
import json
import os
import re
import time

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument('--config_path', type=str, required=True)
args = ap.parse_args()


def print_line():
    print("=="*40)


def get_dataset(df, config):
    """
    crate dataset for NBME
    """
    text_col, feature_col = config["text_col"], config["feature_col"]
    keep_cols = [text_col, feature_col, 'feature_num']

    df = df[keep_cols].copy()
    this_dataset = Dataset.from_pandas(df)

    print_line()
    if config["use_deberta_v3_tokenizer"]:
        print("Using deberta v3/v2 tokenizer")
        tokenizer = DebertaV2TokenizerFast.from_pretrained(config["base_model_path"])
    else:
        print("Using auto tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trim_offsets=False)
    print_line()

    # add markers
    print("adding marker tokens in tokenizer")
    num_cases = 10
    MARKERS = [f"[QA_CASE={i}]" for i in range(num_cases)]
    print(MARKERS)

    print(f"# vocab before adding markers: {len(tokenizer)}")
    tokenizer.add_tokens(MARKERS)
    print(f"# vocab after adding markers: {len(tokenizer)}")

    def tokenize_function(examples):
        tz = tokenizer(
            examples[feature_col],
            examples[text_col],
            padding=False,
            truncation='only_second',
            max_length=config["max_length"],
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )
        return tz

    def _get_sequence_ids(input_ids):
        sequence_ids = [0]*len(input_ids)
        special_token_ids, flag = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())), False

        for i, input_id in enumerate(input_ids):
            if input_id == tokenizer.sep_token_id:
                flag = True

            if flag:
                sequence_ids[i] = 1

            if input_id in special_token_ids:
                sequence_ids[i] = None
        return sequence_ids

    def add_sequence_ids(examples):
        sequence_ids = []
        input_ids = examples["input_ids"]
        for cur_input_ids in input_ids:
            cur_seq_ids = _get_sequence_ids(cur_input_ids)
            sequence_ids.append(cur_seq_ids)
        return {"sequence_ids": sequence_ids}

    def process_offsets(examples):
        """
        unstrip token offsets
        """
        processed_offsets = []
        focus_seq_pair_id = config["text_sequence_identifier"]

        for offsets, seq_ids in zip(examples["offset_mapping"], examples['sequence_ids']):
            updated_offsets = []
            prev_offset = [0, 0]

            for pos, offset in enumerate(offsets):
                seq_id = seq_ids[pos]
                if seq_id != focus_seq_pair_id:
                    updated_offsets.append(offset)
                    prev_offset = offset
                else:
                    if prev_offset[-1] != offset[0]:
                        offset[0] = prev_offset[-1]
                    updated_offsets.append(offset)
                    prev_offset = offset
            processed_offsets.append(updated_offsets)
        return {'offset_mapping_unstripped': processed_offsets}

    this_dataset = this_dataset.map(tokenize_function, batched=True)
    this_dataset = this_dataset.map(add_sequence_ids, batched=True)
    this_dataset = this_dataset.map(process_offsets, batched=True)

    try:
        this_dataset = this_dataset.remove_columns(column_names=["offset_mapping"])
    except Exception as e:
        print(e)

    try:
        this_dataset = this_dataset.remove_columns(column_names=["__index_level_0__"])
    except Exception as e:
        print(e)

    return this_dataset

################################################################
# Execution
################################################################


with open(args.config_path, "r") as f:
    config = json.load(f)

print_line()
print(f"The following config will be used for dataset creation:")
for k, v in config.items():
    print(f"{k:<50}:{v}")
print_line()

test_df = pd.read_pickle("../outputs/infer_df.pkl")


def add_marker(text, case_num):
    MARKER = f"[QA_CASE={case_num}]"
    return MARKER + " " + text


print("adding markers at the beginning of feature text")
test_df["feature_text"] = test_df[["feature_text", "case_num"]].apply(
    lambda x: add_marker(x[0], x[1]), axis=1
)


def process_feature_text(text):
    return re.sub('-', ' ', text)


test_df["feature_text"] = test_df["feature_text"].apply(lambda x: process_feature_text(x))

print(test_df.sample())

nbme_dataset = get_dataset(test_df, config)
nbme_dataset.save_to_disk(config["dataset_save_path"])

print("Dataset creation completed!")
print_line()
###################### DONE ####################################
