import ast
import logging
import os
import re
import traceback
from copy import deepcopy

import numpy as np
import portion as Por
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast

# os.environ["WANDB_DISABLED"] = "true"


def merge_intervals(input_list):
    """merge overlapping interval in a given list of intervals

    :param input_list: list of intervals
    :type input_list: List[List[int, int]]
    :return: merged intervals
    :rtype: List[List[int, int]]
    """
    if len(input_list) == 0:
        return input_list
    input_list = deepcopy(input_list)

    intervals = [Por.closed(a, b) for a, b in input_list]
    merge = Por.Interval(*intervals)
    to_return = []
    for m in merge:
        to_return.append([m.lower, m.upper])
    return to_return


def get_word_offsets(text):
    """get word offsets from a given text

    :param text: _description_
    :type text: _type_
    :return: _description_
    :rtype: _type_
    """
    matches = re.finditer("\S+", text)
    spans, words = [], []

    for match in matches:
        span, word = match.span(), match.group()
        spans.append(span)
        words.append(word)
    assert tuple(words) == tuple(text.split())
    return np.array(spans)


def strip_offset_mapping(text, offset_mapping):
    """process offset mapping produced by huggingface tokenizers
    by stripping spaces from the tokens

    :param text: input text that is tokenized
    :type text: str
    :param offset_mapping: offsets returned from huggingface tokenizers
    :type offset_mapping: list
    :return: processed offset mapping
    :rtype: list
    """
    to_return = []
    for start, end in offset_mapping:
        match = list(re.finditer('\S+', text[start:end]))
        if len(match) == 0:
            to_return.append((start, end))
        else:
            span_start, span_end = match[0].span()
            to_return.append((start + span_start, start + span_end))
    return to_return


def get_sequence_ids(input_ids, tokenizer):
    """if a pair of texts are given to HF tokenizers, the first text
    has sequence id of 0 and second text has sequence id 1. This function
    derives sequence ids for a given tokenizer based on token input ids

    :param input_ids: token input id sequence
    :type input_ids: List[int]
    :param tokenizer: HF tokenizer
    :type tokenizer: PreTrainedTokenizer
    :return: sequence ids
    :rtype: List
    """
    sequence_ids = [0]*len(input_ids)

    switch = False
    special_token_ids = set(
        tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map.values()
        )
    )
    for i, input_id in enumerate(input_ids):
        if input_id == tokenizer.sep_token_id:
            switch = True
        if switch:
            sequence_ids[i] = 1
        if input_id in special_token_ids:
            sequence_ids[i] = None
    return sequence_ids


def create_additional_labels(df):
    """create additional labels for multi-task learning setup

    :param df: input training dataframe
    :type df: pd.DataFrame
    :return: processed dataframe with additional labels and label names
    :rtype: tuple(pd.DataFrame, List[str])
    """

    temp_columns, additional_labels = [], []
    df['anno_len'] = df["annotation"].apply(len)
    temp_columns.append("anno_len")

    df["is_relevant"] = df["anno_len"].apply(lambda x: int(x > 0))
    additional_labels.append("is_relevant")

    df["is_multi_relevant"] = df["anno_len"].apply(lambda x: int(x > 1))
    additional_labels.append("is_multi_relevant")

    df["has_composite_spans"] = df["location"].apply(
        lambda x: int(np.sum([1 for elem in x if ";" in elem]) > 0)
    )
    additional_labels.append("has_composite_spans")

    df["merged_label_spans"] = df["label_spans"].apply(merge_intervals)
    temp_columns.append("merged_label_spans")

    df["has_overlapping_spans"] = (
        df['label_spans'].apply(len) != df["merged_label_spans"].apply(len)
    ).astype(int)
    additional_labels.append("has_overlapping_spans")

    df = df.drop(columns=temp_columns)

    return df, additional_labels


class NbmeTokenClassificationDataset:
    """Dataset class for NBME token classification task
    """

    def __init__(self, config):
        self.config = config

        # column names
        self.text_col = self.config["text_col"]
        self.feature_col = self.config["feature_col"]
        self.label_col = self.config["label_col"]
        self.annotation_col = self.config["annotation_col"]

        # sequence number for patient history texts
        self.focus_seq = self.config["text_sequence_identifier"]

        # load tokenizer
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config 
        """
        if ('v3' in self.config["base_model_path"].lower()) | ('v2' in self.config["base_model_path"].lower()):
            print("using fast deberta v3 tokenizer")
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(self.config["base_model_path"])
        else:
            print("using auto tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model_path"], trim_offsets=False)

    def tokenize_function(self, examples):
        if self.focus_seq == 0:
            tz = self.tokenizer(
                examples[self.text_col],
                examples[self.feature_col],
                padding=False,
                truncation='only_second',
                max_length=self.config["max_length"],
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

        elif self.focus_seq == 1:
            tz = self.tokenizer(
                examples[self.feature_col],
                examples[self.text_col],
                padding=False,
                truncation='only_second',
                max_length=self.config["max_length"],
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

        else:
            raise ValueError("bad text_sequence_identifier in config")
        return tz

    def add_sequence_ids(self, examples):
        sequence_ids = []
        input_ids = examples["input_ids"]

        for tok_ids in input_ids:
            sequence_ids.append(get_sequence_ids(tok_ids, self.tokenizer))
        return {"sequence_ids": sequence_ids}

    def process_token_offsets(self, examples):
        stripped_offsets, unstripped_offsets = [], []
        prev_offset = None

        for offsets, seq_ids, feature_text, pn_history in zip(
            examples["offset_mapping"],
            examples['sequence_ids'],
            examples[self.feature_col],
            examples[self.text_col]
        ):
            current_stripped, current_unstripped = [],  []

            for pos, offset in enumerate(offsets):
                start, end = offset
                seq_id = seq_ids[pos]

                if seq_id is None:
                    current_stripped.append(offset)
                    current_unstripped.append(offset)
                    prev_offset = offset
                    continue

                elif seq_id == self.focus_seq:
                    focus_text = pn_history[start:end]
                else:
                    focus_text = feature_text[start:end]

                # strip offsets
                match = list(re.finditer('\S+', focus_text))
                if len(match) == 0:
                    current_stripped.append((start, end))
                else:
                    span_start, span_end = match[0].span()
                    current_stripped.append((start + span_start, start + span_end))

                # upstrip offsets
                if prev_offset[-1] != offset[0]:
                    offset[0] = prev_offset[-1]
                current_unstripped.append(offset)
                prev_offset = offset

            stripped_offsets.append(np.array(current_stripped))
            unstripped_offsets.append(np.array(current_unstripped))

        return {
            'offset_mapping_stripped': stripped_offsets,
            "offset_mapping_unstripped": unstripped_offsets
        }

    def generate_labels(self, examples):
        labels = []
        for offsets, inputs, seq_ids, locations in zip(
            examples["offset_mapping_stripped"],
            examples["input_ids"],
            examples["sequence_ids"],
            examples[self.label_col]
        ):
            this_label = [0.0]*len(inputs)
            for idx, (seq_id, offset) in enumerate(zip(seq_ids, offsets)):
                if seq_id != self.focus_seq:  # ignore this token
                    this_label[idx] = -1.0
                    continue

                token_start_char_idx, token_end_char_idx = offset
                for label_start_char_idx, label_end_char_idx in locations:
                    # case 1: location char start is inside token
                    if token_start_char_idx <= label_start_char_idx < token_end_char_idx:
                        this_label[idx] = 1.0

                    # case 2: location char end is inside token
                    if token_start_char_idx < label_end_char_idx <= token_end_char_idx:
                        this_label[idx] = 1.0

                    # case 3: token in between location
                    if label_start_char_idx < token_start_char_idx < label_end_char_idx:
                        this_label[idx] = 1.0

                    # break the loop if token is already detected positive
                    if this_label[idx] > 0:
                        break

            labels.append(this_label)
        return {"labels": labels}

    def pre_execution_setups(self, df, mode):
        possible_modes = ["train", "infer"]
        assert mode in possible_modes, f"Error! Supported modes: {possible_modes}"

        if mode == 'train':
            required_cols = [
                self.text_col,
                self.feature_col,
                self.label_col,
                self.annotation_col
            ]
        else:
            required_cols = [self.text_col, self.feature_col]

        for col in required_cols:
            assert col in df.columns, f"Error! column {col} is required!"
        keep_cols = required_cols + self.new_cols
        df = df[keep_cols].copy()
        return df

    def feature_engineering(self, df):
        """derive new features out of input dataframe

        :param df: input dataframe
        :type df: pd.DataFrame
        :return: dataframe with added columns after feature engineering
        :rtype: pd.DataFrame
        """
        self.new_cols = []
        return df

    def get_dataset(self, df, mode='train'):
        """main api for creating the NBME dataset

        :param df: input dataframe
        :type df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        self.new_cols = []
        if mode == 'train':
            df = self.feature_engineering(df)
        df = self.pre_execution_setups(df, mode)

        # create the dataset
        nbme_dataset = Dataset.from_pandas(df)
        nbme_dataset = nbme_dataset.map(self.tokenize_function, batched=True)
        nbme_dataset = nbme_dataset.map(self.add_sequence_ids, batched=True)
        nbme_dataset = nbme_dataset.map(self.process_token_offsets, batched=True)

        if mode == "train":
            nbme_dataset = nbme_dataset.map(self.generate_labels, batched=True)
        try:
            nbme_dataset = nbme_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return nbme_dataset


class NbmeMTLDatasetNeo(NbmeTokenClassificationDataset):
    def __init__(self, config):
        super().__init__(config)

    def generate_labels(self, examples):
        labels = []
        for offsets, inputs, seq_ids, locations in zip(
            examples["offset_mapping_stripped"],
            examples["input_ids"],
            examples["sequence_ids"],
            examples[self.label_col]
        ):
            # print("hello MTL")
            this_label = np.zeros(shape=(3, len(inputs)))
            for idx, (seq_id, offset) in enumerate(zip(seq_ids, offsets)):
                if seq_id != self.focus_seq:  # ignore this token
                    this_label[:, idx] = -1.0
                    continue

                token_start_char_idx, token_end_char_idx = offset
                for label_start_char_idx, label_end_char_idx in locations:
                    # case 1: location char start is inside token
                    if token_start_char_idx <= label_start_char_idx < token_end_char_idx:
                        this_label[0, idx] = 1.0
                        this_label[1, idx] = 1.0  # detection

                    # case 2: location char end is inside token
                    if token_start_char_idx < label_end_char_idx <= token_end_char_idx:
                        this_label[0, idx] = 1.0
                        this_label[2, idx] = 1.0  # termination

                    # case 3: token in between location
                    if label_start_char_idx < token_start_char_idx < label_end_char_idx:
                        this_label[0, idx] = 1.0

                    # break the loop if token is already detected positive
                    if this_label[0, idx] > 0:
                        break

            labels.append(this_label)
        return {"labels": labels}



    

class NbmeMTLDatasetNeoMarked(NbmeMTLDatasetNeo):
    def __init__(self, config):
        super().__init__(config)

    def load_tokenizer(self):
        """load tokenizer as per config 
        """
        if ('v3' in self.config["base_model_path"].lower()) | ('v2' in self.config["base_model_path"].lower()):
            print("using fast deberta v3 tokenizer")
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(self.config["base_model_path"])
        else:
            print("using auto tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_model_path"], trim_offsets=False)
            
        # add markers
        num_cases = 10
        MARKERS = [f"[QA_CASE={i}]" for i in range(num_cases)]
        print(MARKERS)
        
        print(f"# vocab before adding markers: {len(self.tokenizer)}")
        self.tokenizer.add_tokens(MARKERS)
        print(f"# vocab after adding markers: {len(self.tokenizer)}")


class NbmeMTLDatasetV1Pseudo(NbmeMTLDatasetNeo):
    def __init__(self, config):
        super().__init__(config)

    def pre_execution_setups(self, df, mode):
        possible_modes = ["train", "infer"]
        assert mode in possible_modes, f"Error! Supported modes: {possible_modes}"

        if mode == 'train':
            required_cols = [
                self.text_col,
                self.feature_col,
                self.label_col,
                self.annotation_col
            ]
        else:
            required_cols = [self.text_col, self.feature_col]

        for col in required_cols:
            assert col in df.columns, f"Error! column {col} is required!"
        keep_cols = required_cols + self.new_cols + ['confidence']
        df = df[keep_cols].copy()
        return df


class NbmeRelevanceDataset(NbmeTokenClassificationDataset):
    def __init__(self, config):
        super().__init__(config)

    def feature_engineering(self, df):
        """derive new features out of input dataframe

        :param df: input dataframe
        :type df: pd.DataFrame
        :return: dataframe with added columns after feature engineering
        :rtype: pd.DataFrame
        """
        df, self.new_cols = create_additional_labels(df)
        return df

    def generate_labels(self, examples):
        label_cols = self.new_cols
        labels = np.zeros(shape=(len(examples['input_ids']), len(label_cols)))
        for col_idx, col in enumerate(label_cols):
            labels[:, col_idx] = examples[col]
        return {"labels": labels}


# Masked Language Model Dataset
def get_mlm_dataset(notes_df, config):
    nbme_dataset = Dataset.from_pandas(notes_df)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_checkpoint"],
        trim_offsets=True
    )

    def tokenize_function(examples):
        result = tokenizer(examples[config["text_col"]])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_datasets = nbme_dataset.map(
        tokenize_function, batched=True, remove_columns=[config["text_col"]]
    )

    chunk_size = config["chunk_size"]

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    test_pct = config["test_pct"]
    max_train_examples = config["max_train_examples"]
    max_test_examples = int(config["max_train_examples"]*test_pct)

    test_size = int(len(lm_datasets) * test_pct)
    train_size = len(lm_datasets) - test_size

    test_size = min(test_size, max_test_examples)
    train_size = min(train_size, max_train_examples)

    downsampled_dataset = lm_datasets.train_test_split(
        train_size=train_size, test_size=test_size, seed=config["seed"]
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config["mask_probability"]
    )

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    try:
        downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    except Exception as e:
        traceback.print_exc()

    downsampled_dataset["test"] = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )

    downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    return downsampled_dataset
