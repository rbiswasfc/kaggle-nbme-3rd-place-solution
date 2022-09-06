import ast
import os
import re
import traceback
from itertools import chain

import pandas as pd

os.environ["WANDB_DISABLED"] = "true"


def probe_features(df):
    # TODO: implement soft probing
    return df


def add_marker(text, case_num):
    MARKER = f"[QA_CASE={case_num}]"
    return MARKER + " " + text


def add_prefix(text, case_num, feature_num):
    prefix = f"Patient {case_num} Query {feature_num} :"
    return prefix + " " + text


class DataProcessor:
    @staticmethod
    def process_feature_text(text):
        return re.sub('-', ' ', text)

    @staticmethod
    def apply_ast(df):
        columns = [
            "annotation",
            "location",
        ]

        for col in columns:
            try:
                if type(df[col].values[0]) != list:
                    df[col] = df[col].apply(ast.literal_eval)
            except Exception as e:
                print(e)
                traceback.print_exc()

        return df

    @staticmethod
    def location2annotation(inputs):
        """
        a helper function to compute annotation from the location column
        """
        locations, pn_history = inputs

        all_texts = []
        for loc in locations:
            offsets = loc.split(";")
            texts = []
            for offset in offsets:
                s, e = list(map(int, offset.split()))
                texts.append(pn_history[s:e])
            all_texts.append(" ".join(texts))

        return all_texts

    @staticmethod
    def location2spans(location):
        """
        a helper function to compute the label spans from the input location list
        """
        spans = [loc.split(";") for loc in location]
        spans = [list(map(int, s.split())) for s in chain(*spans)]
        return spans

    @staticmethod
    def process_data(nbme_data, soft_probing=False, add_markers=False):
        if add_markers:
            print("adding markers at the beginning of feature text")
            nbme_data.features_df["feature_text"] = nbme_data.features_df[["feature_text", "case_num"]].apply(
                lambda x: add_marker(x[0], x[1]), axis=1
            )

        # process train data
        nbme_data.train_df = DataProcessor.apply_ast(nbme_data.train_df)
        nbme_data.train_df['label_spans'] = nbme_data.train_df['location'].apply(
            DataProcessor.location2spans
        )
        if 'feature_text' not in nbme_data.train_df.columns:
            nbme_data.train_df = pd.merge(nbme_data.train_df, nbme_data.features_df,
                                          on=['feature_num', 'case_num'], how='left')

            nbme_data.train_df = pd.merge(nbme_data.train_df, nbme_data.notes_df,
                                          on=['pn_num', 'case_num'], how='left')

        nbme_data.train_df["feature_text"] = nbme_data.train_df["feature_text"].apply(
            lambda x: DataProcessor.process_feature_text(x)
        )

        nbme_data.features_df["feature_text"] = nbme_data.features_df["feature_text"].apply(
            lambda x: DataProcessor.process_feature_text(x)
        )

        if soft_probing:
            nbme_data.features_df = probe_features(nbme_data.features_df)

        # process test data
        nbme_data.test_df = pd.merge(
            nbme_data.test_df,
            nbme_data.notes_df,
            how='left',
            on=["pn_num", "case_num"]
        )
        nbme_data.test_df = pd.merge(
            nbme_data.test_df,
            nbme_data.features_df,
            how='left',
            on=["feature_num", "case_num"]
        )

        return nbme_data
