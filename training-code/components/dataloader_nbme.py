import os
import pdb
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (DataCollatorForTokenClassification,
                          DataCollatorWithPadding)

from components.dataset_nbme import (NbmeMTLDatasetNeo,
                                     NbmeRelevanceDataset,
                                     NbmeTokenClassificationDataset)


@dataclass
class DataCollatorForMTLNeo(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received
    Multitask learning targets will be added to batch and padded
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -1
    return_tensors = "pt"

    def torch_call(self, features):
        label_name = "labels"
        labels = None

        if label_name in features[0].keys():
            labels = [feature[label_name] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:  # e.g. in eval mode
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        # main labels
        num_labels, padded_labels = len(labels[0]), []

        for this_example in labels:
            padding_length = sequence_length - len(this_example[0])
            padding_matrix = (self.label_pad_token_id)*np.ones(shape=(num_labels, padding_length))
            this_example = (np.concatenate([np.array(this_example), padding_matrix], axis=1).T).tolist()
            padded_labels.append(this_example)
        batch[label_name] = padded_labels

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        # cast lables to float for bce loss
        batch[label_name] = batch[label_name].to(torch.float32)
        return batch


@dataclass
class DataCollatorForMTLV1Pseudo(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received
    Multitask learning targets will be added to batch and padded
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -1
    return_tensors = "pt"

    def torch_call(self, features):
        label_name = "labels"
        required_fields = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']

        labels = None
        if label_name in features[0].keys():
            labels = [feature[label_name] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:  # e.g. in eval mode
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        # main labels
        num_labels, padded_labels = len(labels[0]), []

        for this_example in labels:
            padding_length = sequence_length - len(this_example[0])
            padding_matrix = (self.label_pad_token_id)*np.ones(shape=(num_labels, padding_length))
            this_example = (np.concatenate([np.array(this_example), padding_matrix], axis=1).T).tolist()
            padded_labels.append(this_example)
        batch[label_name] = padded_labels

        batch = {k: (torch.tensor(v, dtype=torch.int64) if k != label_name else torch.tensor(v, dtype=torch.float32))
                 for k, v in batch.items()}

        # cast lables to float for bce loss
        batch[label_name] = batch[label_name].to(torch.float32)
        return batch


# Get Dataloaders
class NbmeTokenClassificationDL:
    def __init__(self, config):
        self.config = config
        self.Loaders = namedtuple('Loaders', ['train', 'valid', 'infer'])
        self.dc = self.get_dataset_creator()
        self.data_collator = self.get_data_collator()

    def get_dataset_creator(self):
        return NbmeTokenClassificationDataset(self.config)

    def get_data_collator(self):
        collator = DataCollatorForTokenClassification(
            tokenizer=self.dc.tokenizer,
            label_pad_token_id=-1
        )

        def collate_fn(batch):
            batch = collator(batch)
            if 'labels' in batch.keys():
                batch["labels"] = batch["labels"].to(torch.float32)
            return batch

        return collate_fn

    def _get_train_loaders(self, df):
        # split train-validation
        # save input
        df.to_pickle(os.path.join(self.config["output_dir"], self.config["processed_nbme_df_path"]))

        train_input = df[df['kfold'].isin(self.config["train_folds"])].copy()
        valid_input = df[df['kfold'].isin(self.config["valid_folds"])].copy()

        # train_input.to_pickle(os.path.join(self.config["output_dir"], self.config["train_df_path"]))
        # valid_input.to_pickle(os.path.join(self.config["output_dir"], self.config["valid_df_path"]))

        train_dataset = self.dc.get_dataset(train_input, mode='train')
        valid_dataset = self.dc.get_dataset(valid_input, mode='train')

        # save datasets
        train_dataset_path = os.path.join(
            self.config["output_dir"], self.config["train_dataset_path"]
        )
        train_dataset.save_to_disk(train_dataset_path)

        valid_dataset_path = os.path.join(
            self.config["output_dir"], self.config["valid_dataset_path"]
        )
        valid_dataset.save_to_disk(valid_dataset_path)

        # create data loaders
        loader_columns = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
        loader_columns += self.dc.new_cols

        train_dataset.set_format(type=None, columns=loader_columns)
        valid_dataset.set_format(type=None, columns=loader_columns)

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.config["train_batch_size"],
            collate_fn=self.data_collator,
            pin_memory=True,
            shuffle=True,
        )

        valid_dl = DataLoader(
            valid_dataset,
            batch_size=self.config["valid_batch_size"],
            collate_fn=self.data_collator,
            pin_memory=True,
            shuffle=False,
        )
        return self.Loaders(train_dl, valid_dl, None)

    def _get_infer_loaders(self, df):
        infer_dataset = self.dc.get_dataset(df, mode='infer')

        # save dataset
        infer_dataset_path = os.path.join(
            self.config["output_dir"], self.config["infer_dataset_path"]
        )
        infer_dataset.save_to_disk(infer_dataset_path)

        infer_dataset.set_format(
            type=None,
            columns=['input_ids', 'attention_mask', 'token_type_ids']
        )

        infer_dl = DataLoader(
            infer_dataset,
            batch_size=self.config["infer_batch_size"],
            collate_fn=self.data_collator,
            pin_memory=True,
            shuffle=False,
        )
        return self.Loaders(None, None, infer_dl)

    def get_loaders(self, df, mode='train'):
        if mode == 'train':
            return self._get_train_loaders(df)
        elif mode == 'infer':
            return self._get_infer_loaders(df)
        else:
            raise


class NbmeMTLDLNeo(NbmeTokenClassificationDL):
    def __init__(self, config):
        super().__init__(config)

    def get_dataset_creator(self):
        return NbmeMTLDatasetNeo(self.config)

    def get_data_collator(self):
        return DataCollatorForMTLNeo(tokenizer=self.dc.tokenizer, label_pad_token_id=-1)


class NbmeRelevanceDL(NbmeTokenClassificationDL):
    def __init__(self, config):
        super().__init__(config)

    def get_dataset_creator(self):
        return NbmeRelevanceDataset(self.config)

    def get_data_collator(self):
        collator = DataCollatorWithPadding(tokenizer=self.dc.tokenizer)

        def collate_fn(batch):
            batch = collator(batch)
            if "labels" in batch.keys():
                batch["labels"] = batch["labels"].to(torch.float32)
            return batch

        return collate_fn
