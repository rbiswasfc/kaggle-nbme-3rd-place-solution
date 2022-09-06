import json
import logging
import os
import random
import traceback
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from .constants import MAX_FOLD_ID


def get_secrets():
    """function to read in secret items e.g. passwords, tokens

    :return: secret dict
    :rtype: dict
    """
    secret_path = str(Path(__file__).parents[1] / "secrets.json")
    with open(secret_path) as f:
        return json.load(f)


def read_random_hash():
    """read current hash value

    :return: hash string
    :rtype: str
    """
    hash_path = str(Path(__file__).parents[1] / "random_hash.json")
    with open(hash_path, "r") as f:
        hash_dict = json.load(f)
    h = hash_dict["random_hash"]
    return h


def load_config(flatten=False):
    """function to load project configs

    :return: config dict
    :rtype: dict
    """

    RANDOM_HASH = read_random_hash()

    config = dict()

    model_config_path = Path(__file__).parents[1] / "configs/model_config.json"
    data_config_path = Path(__file__).parents[1] / "configs/data_config.json"
    scheduler_config_path = Path(__file__).parents[1] / "configs/scheduler_config.json"
    main_config_path = Path(__file__).parents[1] / "configs/main_config.json"

    with open(model_config_path) as f:
        config["model_config"] = json.load(f)

    with open(data_config_path) as f:
        config["data_config"] = json.load(f)

    with open(scheduler_config_path) as f:
        config["scheduler_config"] = json.load(f)

    with open(main_config_path) as f:
        config["main_config"] = json.load(f)

    config["main_config"]["random_hash"] = RANDOM_HASH

    # folder managements
    # "RBB", "DT", "MTL", "ST1", "F4"
    tags = config["main_config"]["tags"]

    BACKBONE = tags[0]
    ADAPTATION = tags[1]
    NBME_MODEL = tags[2]
    STAGE = tags[3]
    FOLD = tags[4]

    VALID_FOLDS = [int(FOLD[-1])]
    TRAIN_FOLDS = [i for i in range(MAX_FOLD_ID+1) if i not in VALID_FOLDS]
    RUN_SRC_FOLDER = "_".join([BACKBONE, ADAPTATION, NBME_MODEL, STAGE, FOLD])

    config["data_config"]["model_dir"] = os.path.join(
        config["data_config"]["model_dir"], RUN_SRC_FOLDER, RANDOM_HASH
    )

    config["data_config"]["output_dir"] = os.path.join(
        config["data_config"]["model_dir"], "outputs"
    )

    config["data_config"]["logs_dir"] = os.path.join(
        config["data_config"]["output_dir"], "e2e_log"
    )

    config["data_config"]["train_folds"] = TRAIN_FOLDS
    config["data_config"]["valid_folds"] = VALID_FOLDS

    if flatten:
        flat_config = dict()
        for outer_k, outer_v in config.items():
            for k, v in outer_v.items():
                flat_config[k] = v
        return flat_config
    else:
        return config


def seed_everything(seed=461):
    """set seed for the kernel run

    :param seed: seed, defaults to 461
    :type seed: int, optional
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_env_vars():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["WANDB_MODE"] = "online"


def setup_display():
    pd.options.display.max_colwidth = 250
    pd.options.display.max_rows = 100


def setup_wandb():
    wandb_api_key = get_secrets()["wandb_key"]
    try:
        wandb.login(key=wandb_api_key)
    except:
        print("Wandb login failed")
        traceback.print_exc()


def setup_wandb_run(config, experiment_group):
    tags = config["tags"]
    project = config["project"]
    run_id = "_".join(tags) + config["random_hash"]
    run = wandb.init(
        project=project,
        group=experiment_group,
        config=config,
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )
    return run


def check_create_dir(dir_path):
    """
    check if folder exists at a specific path, if not create the directory
    :param dir_path: path to the directory to be created
    :type dir_path: str
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def setup_folders(config):
    check_paths = [
        config["model_dir"],
        config["output_dir"],
        config["logs_dir"]
    ]

    for cp in check_paths:
        check_create_dir(cp)


def setup(config):
    """master function to set up different project related aspects

    :param config: config
    :type config: dict
    """
    setup_wandb()
    seed_everything(config["seed"])
    setup_env_vars()
    setup_display()
    setup_folders(config)


def setup_logger(logger, config):
    """
    configure logger for the project

    :param logger: a logger object
    :type logger: logging.Logger
    :param config: config dict
    :type config: dict
    :return: logger with desired settings
    :rtype: logging.Logger
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    logs_dir = config["logs_dir"]
    logs_file = os.path.join(logs_dir, "execution.log")

    file_handler = logging.FileHandler(logs_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def try_except(logger):
    """
    A decorator for exception handing
    :param logger: logger to used inside the docorator
    :type logger: logging.Logger
    :return a docorator
    :rtype: function
    """

    def main_decorator(func):
        """
        Function containing decorator logic
        :param func: function for which exception handing will be performed
        :type func: function
        :return: wrapped function
        :rtype: function
        """

        @ wraps(func)
        def wrapper(*args, **kwargs):
            try:
                value = func(*args, **kwargs)
                return value
            except Exception:
                # traceback.print_exc()
                logger.exception("Exception in running {}".format(func.__name__))
                return None

        return wrapper

    return main_decorator


def get_model_checkpoint_path(config):
    """parse model checkpoint path based on config dict

    :param config: config dict
    :type config: dict
    :return: model checkpoint path
    :rtype: str
    """
    model_dir = config["model_dir"]
    fold_id = "_".join(list(map(str, config["valid_folds"])))

    model_filename = f"nbme_model_fold_{fold_id}.ckpt"
    checkpoint_path = os.path.join(model_dir, model_filename)
    return checkpoint_path


if __name__ == "__main__":
    get_secrets()
