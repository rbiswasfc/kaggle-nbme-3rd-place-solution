import argparse
import gc
import itertools
import json
import os
import re
import time
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_from_disk
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoTokenizer,
                          DataCollatorForTokenClassification)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast


def print_line():
    print("=="*40)


def print_usage():
    print_line()
    print(f"cpu = {psutil.cpu_percent()}%")
    print(f"memory = {psutil.virtual_memory()[2]}%")
    print_line()


print_usage()


ap = argparse.ArgumentParser()
ap.add_argument('--config_path', type=str, required=True)
ap.add_argument("--save_path", type=str, required=True)
args = ap.parse_args()

################################################################
# Model
################################################################


class ReInit:
    def __init__(self, arch):
        self.arch = arch

    def reinit(self, base_model, num_reinit_layers):
        print(f'dummy reinit ...')


class NbmeModelMTLNeo(LightningModule):
    """The Multi-task NBME model class
    """

    def __init__(self, config, scorer=None):
        super(NbmeModelMTLNeo, self).__init__()
        self.save_hyperparameters(ignore='scorer')

        # base transformer
        self.base_model = AutoModel.from_pretrained(
            self.hparams.config["base_model_path"],
        )

        if config["gradient_checkpointing"]:
            self.base_model.gradient_checkpointing_enable()

        if config.get("resize_embedding", False):
            print("resizing model embeddings...")
            print(f"tokenizer length = {config['len_tokenizer']}")
            self.base_model.resize_token_embeddings(config["len_tokenizer"])

        hidden_size = self.base_model.config.hidden_size
        num_layers_in_head = self.hparams.config["num_layers_in_head"]

        # token classification head
        self.tok_classifier = nn.Linear(
            in_features=hidden_size * num_layers_in_head,
            out_features=self.hparams.config['mtl_tok_num_labels'],
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def forward(self, **kwargs):
        out = self.base_model(**kwargs, output_hidden_states=True)
        last_hidden_state = out["last_hidden_state"]
        all_hidden_states = out["hidden_states"]

        # token classification logits
        n = self.hparams.config["num_layers_in_head"]
        tok_output = torch.cat(all_hidden_states[-n:], dim=-1)

        # pass through 5 dropout layers and take average
        tok_output1 = self.dropout1(tok_output)
        tok_output2 = self.dropout2(tok_output)
        tok_output3 = self.dropout3(tok_output)
        tok_output4 = self.dropout4(tok_output)
        tok_output5 = self.dropout5(tok_output)
        tok_output = (tok_output1 + tok_output2 + tok_output3 + tok_output4 + tok_output5)/5

        tok_logits = self.tok_classifier(tok_output)

        return tok_logits

    def get_logits(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        tok_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return tok_logits

    def predict_step(self, batch, batch_idx):
        tok_logits = self.get_logits(batch)
        preds = torch.sigmoid(tok_logits)
        return preds


################################################################
# Utility Functions
################################################################

def get_char_probs(text, predictions, offsets, seq_ids):
    results = np.zeros(len(text))
    for pred, offset, seq_id in zip(predictions, offsets, seq_ids):
        if seq_id != config["text_sequence_identifier"]:
            continue
        char_start_idx = offset[0]
        char_end_idx = offset[1]
        results[char_start_idx:char_end_idx] = pred
    return results


def localization_postprocess(text, offset_mapping):
    to_return = []
    for start, end in offset_mapping:
        match = list(re.finditer('\S+', text[start:end]))
        if len(match) == 0:
            to_return.append((start, end))
        else:
            span_start, span_end = match[0].span()
            to_return.append((start + span_start, end))
    return to_return


def localization(char_prob, th):
    result = np.where(char_prob >= th)[0]
    result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
    result = [[min(r), max(r)+1] for r in result]
    return result


def extract_confidence(char_probs, locations):
    results = []
    for start, end in locations:
        results.append(round(np.mean(char_probs[start:end]), 4))
    return results


def prepare_submission(infer_dataset, preds, tuned_thresholds, default_threshold):
    info_df = pd.DataFrame()

    keep_cols = [
        "pn_history",
        "feature_text",
        "feature_num",
        "sequence_ids",
        "offset_mapping_unstripped",
    ]
    for col in keep_cols:
        info_df[col] = infer_dataset[col]
    info_df["token_preds"] = preds
    info_df["threshold"] = info_df['feature_num'].apply(lambda x: tuned_thresholds.get(int(x), default_threshold))

    info_df['char_probs'] = info_df[["pn_history", "token_preds", "offset_mapping_unstripped", "sequence_ids"]].apply(
        lambda x: get_char_probs(x[0], x[1], x[2], x[3]), axis=1)

    info_df['char_localization'] = info_df[["char_probs", "threshold"]].apply(
        lambda x: localization(x[0], x[1]), axis=1)

    info_df['boxes'] = info_df[["pn_history", "char_localization"]].apply(
        lambda x: localization_postprocess(x[0], x[1]), axis=1)

    info_df['box_scores'] = info_df[["char_probs", "boxes"]].apply(
        lambda x: extract_confidence(x[0], x[1]), axis=1)

    info_df["prediction_string"] = info_df["boxes"].apply(lambda x: ";".join([f"{elem[0]} {elem[1]}" for elem in x]))
    info_df = info_df[["char_probs", "boxes", "box_scores"]].copy()
    return info_df


################################################################
# Execution
################################################################
with open(args.config_path, "r") as f:
    config = json.load(f)

print_line()
print(f"The following config will be used for inference:")
for k, v in config.items():
    print(f"{k:<50}:{v}")
print_line()

checkpoint = config["checkpoint_path"]
################################################################
# Dataset & Data Loaders
################################################################

infer_ds = load_from_disk(config["dataset_path"])
original_dataset = deepcopy(infer_ds)

infer_ds.set_format(
    type=None,
    columns=['input_ids', 'attention_mask', 'token_type_ids']
)

if config["use_deberta_v3_tokenizer"]:
    print("using deberta v3 tokenizer")
    tokenizer = DebertaV2TokenizerFast.from_pretrained(config["base_model_path"])
else:
    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trim_offsets=False)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

infer_dl = DataLoader(
    infer_ds,
    batch_size=config["infer_batch_size"],
    collate_fn=data_collator,
    pin_memory=True,
    shuffle=False,
)

################################################################
# Predictions
################################################################
if config.get("process_ckpt", False):
    print("processing ckpt...")
    ckpt = torch.load(checkpoint)
    print("removing module from keys...")
    state_dict = ckpt['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    processed_state = {"state_dict": new_state_dict}

    del state_dict
    gc.collect()
    checkpoint = "../outputs/swa_model.pth.tar"
    torch.save(processed_state, checkpoint, _use_new_zipfile_serialization=False)
    del processed_state, new_state_dict
    gc.collect()
    print("done...")

model = NbmeModelMTLNeo.load_from_checkpoint(checkpoint, config=config, strict=False)
trainer = pl.Trainer(gpus=1)
# trainer = pl.Trainer()
preds = trainer.predict(model, dataloaders=infer_dl)
preds = [p.to('cpu').detach().numpy()[:, :, 0] for p in preds]
preds = list(chain(*preds))

################################################################
# Submission Dataframe Preparation
################################################################

tuned_thresholds = dict()

default_threshold = config["threshold"]
df_preds = prepare_submission(original_dataset, preds, tuned_thresholds, default_threshold)
df_preds.to_pickle(args.save_path)

################################################################
# Clean Ups
################################################################

del model, trainer, original_dataset, infer_ds, infer_dl, df_preds
torch.cuda.empty_cache()
gc.collect()

################################################################
print("sleeping for 2 seconds...")
time.sleep(2)
print("slept well!")
################################################################
print("resources available at the end of script execution..")
print_usage()
