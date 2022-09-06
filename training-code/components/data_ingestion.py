import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class NbmeCompData:
    train_df: Optional[pd.DataFrame]
    features_df: Optional[pd.DataFrame]
    notes_df: Optional[pd.DataFrame]
    test_df: Optional[pd.DataFrame]
    submission_df: Optional[pd.DataFrame]


def ingest_data(config):
    """helper function for ingesting raw data for NBME competition

    :param config: config dict
    :type config: dict
    :return: nbme project data
    :rtype: NbmeCompData
    """
    # data_config = config["data_config"]
    data_dir = config["data_dir"]

    train_df = pd.read_csv(os.path.join(data_dir, config["train_path"]))
    features_df = pd.read_csv(os.path.join(data_dir, config["features_path"]))
    notes_df = pd.read_csv(os.path.join(data_dir, config["notes_path"]))
    test_df = pd.read_csv(os.path.join(data_dir, config["test_path"]))
    submission_df = pd.read_csv(os.path.join(data_dir, config["submission_path"]))

    project_data = NbmeCompData(
        train_df,
        features_df,
        notes_df,
        test_df,
        submission_df,
    )
    return project_data
