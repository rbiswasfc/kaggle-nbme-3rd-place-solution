import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# read settings
settings_path = str(Path(__file__).parents[1] / "settings.json")

with open(settings_path, "r") as f:
    root_config = json.load(f)

data_dir = root_config["TEST_DATA_DIR"]
output_dir = root_config["OUTPUT_DIR"]

features_df = pd.read_csv(os.path.join(data_dir, "features.csv"))
notes_df = pd.read_csv(os.path.join(data_dir, "patient_notes.csv"))

test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
test_df = pd.merge(test_df, notes_df, how='left', on=["pn_num", "case_num"])
test_df = pd.merge(test_df, features_df, how='left', on=["feature_num", "case_num"])

# sort test data based on input text length for faster inference
test_df["text_length"] = test_df[["pn_history", "feature_text"]].apply(
    lambda x: len(x[0]) + len(x[1]), axis=1
)
length_sorted_idx = np.argsort([-len_ for len_ in test_df["text_length"].tolist()])
sorted_df = test_df.sort_values(by='text_length', ascending=False)

sorted_df = sorted_df.drop(columns=['text_length'])
test_df = test_df.drop(columns=['text_length'])

sorted_df.to_pickle(os.path.join(output_dir, "infer_df.pkl"))
np.save(os.path.join(output_dir, "sorted_idx.npy"), length_sorted_idx)
