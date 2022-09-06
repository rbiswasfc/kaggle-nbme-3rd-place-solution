import argparse
import ast
import copy
import gc
import itertools
import json
import math
import os
import pickle
import random
import re
import string
import sys
import time
from itertools import chain

import numpy as np
import pandas as pd
import psutil
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          DataCollatorForTokenClassification)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################
# Configs & Arguments
################################################################


def print_usage():
    print("=="*30)
    print(f"cpu = {psutil.cpu_percent()}%")
    print(f"memory = {psutil.virtual_memory()[2]}%")
    print("=="*30)


ap = argparse.ArgumentParser()
ap.add_argument('--save_path', type=str, required=True)
args = ap.parse_args()

main_dir = "../data/inference_data/"


class CFG:
    num_workers = 4
    path = "../prod-models/delv3/deberta-v3-large-5-folds-public/"
    config_path = path+'config.pth'
    model = "microsoft/deberta-v3-large"
    batch_size = 32
    fc_dropout = 0.2
    max_len = 512
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]


tokenizer = DebertaV2TokenizerFast.from_pretrained('../prod-models/backbones/NBME_DELV3_TAPT')
CFG.tokenizer = tokenizer

################################################################
# Utility Functions
################################################################


def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)

    return predictions


def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features


def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(
        text,
        feature_text,
        add_special_tokens=True,
        padding=False,
        truncation='only_first',
        max_length=CFG.max_len,
        return_offsets_mapping=False
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])

        return inputs


class ScoringModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg

        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]

        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))

        return output


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)

    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').tolist())

    predictions = list(chain(*preds))

    return predictions


################################################################
# Execution
################################################################
test_df = pd.read_pickle("../outputs/infer_df.pkl")

tmp = pd.read_csv(main_dir+'test.csv')
tmp_cols = tmp.columns
test = test_df[tmp_cols].copy()
del tmp, test_df
gc.collect()

submission = pd.read_csv(main_dir + 'sample_submission.csv')
features = pd.read_csv(main_dir+'features.csv')
patient_notes = pd.read_csv(main_dir+'patient_notes.csv')
features = preprocess_features(features)

test = test.merge(features, on=['feature_num', 'case_num'], how='left')
test = test.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

test_dataset = TestDataset(CFG, test)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         collate_fn=data_collator,
                         # num_workers=CFG.num_workers,
                         pin_memory=True,
                         drop_last=False)

predictions = []
for fold in CFG.trn_fold:
    print(f"getting predictions from fold {fold}")
    model = ScoringModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(CFG.path+f"{CFG.model.split('/')[1]}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    char_probs = get_char_probs(test['pn_history'].values, prediction, CFG.tokenizer)
    predictions.append(char_probs)
    del model, state, prediction, char_probs
    gc.collect()
    torch.cuda.empty_cache()

test['pred_0'] = predictions[0]
test['pred_1'] = predictions[1]
test['pred_2'] = predictions[2]
test['pred_3'] = predictions[3]
test['pred_4'] = predictions[4]

#------------------------------------------------------------------------------------------#


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


#------------------------------------------------------------------------------------------#
for fold in CFG.trn_fold:
    test['char_localization'] = test[f"pred_{fold}"].apply(lambda x: localization(x, 0.5))
    test[f'boxes_{fold}'] = test[["pn_history", "char_localization"]].apply(
        lambda x: localization_postprocess(x[0], x[1]), axis=1)
    test[f'box_scores_{fold}'] = test[[f"pred_{fold}", f"boxes_{fold}"]].apply(
        lambda x: extract_confidence(x[0], x[1]), axis=1)
#------------------------------------------------------------------------------------------#

test = test.drop(columns=["char_localization"])
test.to_pickle(args.save_path)

#------------------------------------------------------------------------------------------#
print("resources available at the end of script execution..")
print_usage()
#------------------------------------------------------------------------------------------#
