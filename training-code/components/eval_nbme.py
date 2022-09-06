import itertools
import json
import logging
import os
import re
import warnings
from copy import deepcopy
from itertools import chain, groupby

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast

from components.data_ingestion import ingest_data
from components.data_processing import DataProcessor
from components.dataloader_nbme import DataCollatorForMTLNeo
from components.metric_utils import span_micro_f1
from components.utils import load_config

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def compute_annotation_from_location(pn_history, locations):
    all_texts = []
    for char_start_idx, char_end_idx in locations:
        all_texts.append(pn_history[char_start_idx:char_end_idx])
    return all_texts


def perform_localization(char_probs, threshold=0.5):
    """convert character wise prediction to location spans

    :param char_probs: character wise predictions
    :type char_prob: list
    :param threshold: threshold for label decision, defaults to 0.5
    :type threshold: float, optional
    :return: locations
    :rtype: list
    """
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
            to_return.append((start, end))
        else:
            span_start, _ = match[0].span()
            to_return.append((start + span_start, end))
    return to_return


def extract_confidence(char_probs, locations):
    results = []
    for start, end in locations:
        results.append(round(np.mean(char_probs[start:end]), 4))
    return results


def token2char(text, quantities, offsets, seq_ids, focus_seq=1):
    """convert token prediction/truths to character wise predictions/truths

    :param text: patient notes text
    :type text: str
    :param quantities: token level variable values
    :type quantities: list
    :param offsets: token offsets without stripping
    :type offsets: list
    :param seq_ids: sequence id of the tokens
    :type seq_ids: list
    :param focus_seq: which sequence to focus on
    :type focus_seq: int
    :return: character probabilities
    :rtype: list
    """
    results = np.zeros(len(text))
    for q, offset, seq_id in zip(quantities, offsets, seq_ids):
        if seq_id != focus_seq:
            continue
        char_start_idx, char_end_idx = offset[0], offset[1]
        results[char_start_idx:char_end_idx] = q
    return results


class ValidationReporter:
    def __init__(self, config, model, valid_ds):
        """initialize the validation reporter class
        get the validation dataset, validation dataloader and model 

        :param config: config for the current run
        :type config: dict
        """
        self.config = config
        self.model = model
        self.ds = deepcopy(valid_ds)

        # create validation dataloader
        loader_columns = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
        valid_ds.set_format(type=None, columns=loader_columns)

        if 'v3' in config["base_model_path"].lower():
            print("using fast deberta v3 tokenizer")
            tokenizer = DebertaV2TokenizerFast.from_pretrained(config["base_model_path"])
        else:
            tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

        data_collator = DataCollatorForMTLNeo(tokenizer=tokenizer, label_pad_token_id=-1)
        self.dl = DataLoader(
            valid_ds,
            batch_size=config["valid_batch_size"],
            collate_fn=data_collator,
            pin_memory=True,
            shuffle=False,
        )

    def get_ground_truths(self):
        """fetch the ground truth labels
        """
        actuals = []
        for batch in self.dl:
            actuals.append(batch['labels'].to('cpu').detach().numpy()[:, :, 0])
        actuals = list(chain(*actuals))
        self._actuals = actuals

    def get_predictions(self):
        """get the model predictions on the validation data
        """
        trainer = pl.Trainer(gpus=1)
        preds = trainer.predict(self.model, dataloaders=self.dl)
        preds = [p.to('cpu').detach().numpy()[:, :, 0] for p in preds]
        preds = list(chain(*preds))
        self._preds = preds

    def get_char_truths(self, text, offsets):
        results = np.zeros(len(text))
        for char_start_idx, char_end_idx in offsets:
            results[char_start_idx:char_end_idx] = 1
        return results

    def generate_summary_df(self):
        """initialize the summary dataframe for current analysis
        """
        info_df = pd.DataFrame()

        required_cols = [
            "pn_history",
            "feature_text",
            "annotation",
            "label_spans",
            "sequence_ids",
            "offset_mapping_unstripped"
        ]

        for col in required_cols:
            info_df[col] = self.ds[col]

        self.get_ground_truths()
        self.get_predictions()

        info_df["token_preds"] = self._preds
        info_df["token_truths"] = self._actuals

        input_cols = ["pn_history", "token_preds", "offset_mapping_unstripped", "sequence_ids"]
        focus_seq = self.config["text_sequence_identifier"]

        info_df["char_probs"] = info_df[input_cols].apply(
            lambda x: token2char(x[0], x[1], x[2], x[3], focus_seq), axis=1
        )

        info_df["char_truths"] = info_df[["pn_history", "label_spans"]].apply(
            lambda x: self.get_char_truths(x[0], x[1]), axis=1
        )

        info_df = info_df.drop(columns=["token_preds", "token_truths"])
        return info_df


class ValidationResultProducer:
    @staticmethod
    def produce_results(df, threshold):
        df["location_preds"] = df["char_probs"].apply(
            lambda x: perform_localization(x, threshold)
        )
        df['location_preds'] = df[["pn_history", "location_preds"]].apply(
            lambda x: postprocess_localization(x[0], x[1]), axis=1
        )
        df['annotation_preds'] = df[["pn_history", "location_preds"]].apply(
            lambda x: compute_annotation_from_location(x[0], x[1]), axis=1
        )
        df['location_preds_confidence'] = df[["char_probs", "location_preds"]].apply(
            lambda x: extract_confidence(x[0], x[1]), axis=1)
        return df


class PerformanceEvaluator:
    def __init__(self, eval_df):
        self.df = eval_df
        self.performance = dict()

    def get_threshold_invariant_metrics(self):
        """measure the model performance in terms of AUC, AUC PR
        """
        flat_truths = list(chain(*self.df["char_truths"].values))
        flat_preds = list(chain(*self.df["char_probs"].values))

        self.performance["char_auc"] = roc_auc_score(flat_truths, flat_preds)
        self.performance["char_aucpr"] = average_precision_score(flat_truths, flat_preds)

    def evaluate_at_threshold(self, threshold):
        """evaluate model performance at certain threshold

        :param threshold: threshold value
        :type threshold: float
        :return: performance measures
        :rtype: tuple of precision, recall, f1 metrics
        """
        df_th = deepcopy(self.df)

        df_th["char_yp"] = df_th["char_probs"].apply(lambda x: (np.array(x) >= threshold).astype(int))

        flat_truths = list(chain(*df_th["char_truths"].values))
        flat_yps = list(chain(*df_th["char_yp"].values))

        measures = precision_recall_fscore_support(flat_truths, flat_yps, average='binary')
        char_precision = measures[0]
        char_recall = measures[1]
        char_f1 = measures[2]
        return char_precision, char_recall, char_f1

    def get_performance(self):
        self.get_threshold_invariant_metrics()

        step_size = 0.02
        n_steps = 20
        threshold_list = [0.30 + i*step_size for i in range(n_steps)]

        self.performance["char_precision_list"] = []
        self.performance["char_recall_list"] = []
        self.performance["char_f1_list"] = []
        self.performance["threshold_list"] = threshold_list

        for th in threshold_list:
            pr, rr, f1 = self.evaluate_at_threshold(th)
            self.performance["char_precision_list"].append(pr)
            self.performance["char_recall_list"].append(rr)
            self.performance["char_f1_list"].append(f1)

        best_f1_idx = np.argmax(self.performance["char_f1_list"])
        best_threshold = self.performance["threshold_list"][best_f1_idx]
        best_f1 = self.performance["char_f1_list"][best_f1_idx]

        self.performance["best_f1"] = best_f1
        self.performance["best_threshold"] = best_threshold
        return self.performance


class FeaturePerformanceAnalyzer:
    def __init__(self, config):
        nbme_data = ingest_data(config)
        nbme_data = DataProcessor.process_data(nbme_data)
        self.features_df = nbme_data.features_df
        self.text2num = dict(zip(self.features_df["feature_text"], self.features_df["feature_num"]))
        self.num2text = {v: k for k, v in self.text2num.items()}

    def get_feature_wise_performance_df(self, summary_df, model_threshold):
        all_features = self.text2num.keys()

        rows = []
        for feat in all_features:
            feat_num = self.text2num[feat]
            feat_df = summary_df[summary_df["feature_text"] == feat].copy()

            try:
                feat_evaluator = PerformanceEvaluator(feat_df)
                feat_performance = feat_evaluator.get_performance()

                feat_aucpr = feat_performance["char_aucpr"]
                feat_best_th = feat_performance["best_threshold"]
                feat_best_f1 = feat_performance["best_f1"]

                flat_truths = list(chain(*feat_df["char_truths"].values))
                flat_preds = list(chain(*feat_df["char_probs"].values))
                flat_yps = [1 if p >= model_threshold else 0 for p in flat_preds]
                res = precision_recall_fscore_support(flat_truths, flat_yps, average='binary')
                feat_f1 = res[2]

                this_row = [feat_num, feat, feat_aucpr, feat_f1, feat_best_f1, feat_best_th]
                # print(this_row)
                rows.append(this_row)
            except Exception as e:
                print(e)

        feature_report_df = pd.DataFrame(rows)
        feature_report_df.columns = ["feature_num", "feature_text", "aucpr", "f1",
                                     "best_f1", "best_threshold"]

        feature_report_df = feature_report_df.sort_values(by='aucpr')
        return feature_report_df


def scorer(preds, valid_dataset, threshold=0.5, focus_seq=1):
    """scorer for evaluation during training of models

    :param preds: model preds on valid dataset [char probs]
    :type preds: list
    :param valid_dataset: validation dataset
    :type valid_dataset: dataset
    :param threshold: threshold at which model is evaluated, defaults to 0.5
    :type threshold: float, optional
    :param focus_seq: patient notes sequence, defaults to 1
    :type focus_seq: int, optional
    :return: Leaderboard metric
    :rtype: float
    """
    info_df = pd.DataFrame()
    required_cols = [
        "pn_history",
        "sequence_ids",
        "offset_mapping_unstripped",
        "label_spans",
    ]

    for col in required_cols:
        info_df[col] = valid_dataset[col]

    info_df["token_preds"] = preds

    # convert token preds to char preds
    input_cols = ["pn_history", "token_preds", "offset_mapping_unstripped", "sequence_ids"]
    info_df["char_probs"] = info_df[input_cols].apply(
        lambda x: token2char(x[0], x[1], x[2], x[3], focus_seq), axis=1
    )

    #  location
    info_df["location_preds"] = info_df["char_probs"].apply(
        lambda x: perform_localization(x, threshold)
    )
    info_df['location_preds'] = info_df[["pn_history", "location_preds"]].apply(
        lambda x: postprocess_localization(x[0], x[1]), axis=1
    )
    lb = span_micro_f1(info_df["label_spans"].values, info_df["location_preds"].values)
    return lb
