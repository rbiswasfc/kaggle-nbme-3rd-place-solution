import argparse
import json
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from datasets import load_from_disk
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import (AutoModel, DataCollatorForTokenClassification,
                          get_cosine_schedule_with_warmup, get_scheduler)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast

from components.data_ingestion import ingest_data
from components.data_processing import DataProcessor
from components.dataloader_nbme import DataCollatorForMTLNeo
from components.dataset_nbme import NbmeMTLDatasetNeo
from components.eval_nbme import scorer
from components.models_nbme import NbmeModelMTLNeo

#------------------------ GPU Utils ------------------------------------#


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


#------------------------ Config ---------------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument('--config_path', type=str, required=True)
args = ap.parse_args()

with open(args.config_path, "r") as f:
    config = json.load(f)

config["dropout"] = 0.1
config["num_layers_reinit"] = 0
config['scheduler_name'] = 'NoScheduler'
config["lr"] = 5e-6
config["weight_decay"] = 1e-3
config["epochs"] = 2
config["num_accumulate_grad"] = 1
config["val_check_interval"] = 0.5
config["patience"] = 4
config["encoder_lr"] = 5e-6
config["decoder_lr"] = 5e-6
config["model_dir"] = "../dev-models/del/"
# config["model_name"] = "A_DEL_MPL_1.ckpt"
config["output_dir"] = "../outputs/finetue_outputs"
config["awp_flag"] = False
config["awp_trigger"] = 1.0
config["freeze_lower_encoders"] = True
config["grad_accumulation"] = 1
config["batch_size"] = 32
config['max_length'] = 480
# config["student_path"] = "../dev-models/tmp/del-mpl-1/trained_student/student_last.pth.tar"

#------------------------ Data Loaders -------------------------------------#


def get_train_valid_dataloaders(config):
    """get train and valid dataloader 

    :param config: config
    :type config: dict
    :return: train and valid dataloaders
    :rtype: tuple(DataLoader, DataLoader)
    """
    nbme_data = ingest_data(config)
    nbme_data = DataProcessor.process_data(nbme_data)

    df = nbme_data.train_df
    if config["debug"]:
        print("DEBUG Mode: sampling 1024 examples from train data")
        df = df.sample(min(1024, len(df)))
    train_df = df[df['kfold'].isin(config['train_folds'])].copy()
    valid_df = df[df['kfold'].isin(config['valid_folds'])].copy()

    dataset_creator = NbmeMTLDatasetNeo(config)
    train_ds = dataset_creator.get_dataset(train_df, mode='train')
    valid_ds = dataset_creator.get_dataset(valid_df, mode='train')

    # save train dataset
    train_dataset_path = os.path.join(
        config["output_dir"], config["train_dataset_path"]
    )
    train_ds.save_to_disk(train_dataset_path)

    # save train dataset
    valid_dataset_path = os.path.join(
        config["output_dir"], config["valid_dataset_path"]
    )
    valid_ds.save_to_disk(valid_dataset_path)

    train_ds.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    )

    data_collator = DataCollatorForMTLNeo(tokenizer=dataset_creator.tokenizer, label_pad_token_id=-1)

    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=True,
    )

    valid_ds.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=config["batch_size"],
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=False,
    )
    return train_dl, valid_dl


train_dl, valid_dl = get_train_valid_dataloaders(config)

#------------------------ Scorer -----------------------------------------#

valid_ds_path = os.path.join(config["output_dir"], config["valid_dataset_path"])
valid_ds = load_from_disk(valid_ds_path)

scorer_fn = partial(
    scorer,
    valid_dataset=valid_ds,
    threshold=0.5,
    focus_seq=config["text_sequence_identifier"]
)
model = NbmeModelMTLNeo.load_from_checkpoint(
    config["student_path"],
    config=config,
    scorer=scorer_fn
)

#------------------------ Finetune Preparation -----------------------------------------#


@dataclass
class Learner:
    model: NbmeModelMTLNeo
    train_dl: DataLoader
    valid_dl: DataLoader
    config: dict
    callbacks: list


def get_callbacks(config):
    model_dir = config["model_dir"]
    filename = config["model_name"]

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        dirpath=model_dir,
        filename=filename,
        verbose=True,
        monitor="estimated_lb",
        mode="max",
        save_last=False,
        save_weights_only=True,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    return [checkpoint_callback, lr_monitor_callback]  # , swa_callback]


def train(learner):
    config = learner.config

    # experiment_group = config["experiment_group"]
    logger = CSVLogger(f"{config['output_dir']}/pl_logs", name="train")

    # if config["debug"]:
    #     experiment_group += "-debug"

    # # wandb setup
    # setup_wandb_run(
    #     experiment_group=experiment_group,
    #     config=config,
    # )
    # wandb.watch(learner.model, log_freq=500)

    # trainer setup
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=learner.config["epochs"],
        precision=16 if learner.config["mixed_precision"] else 32,
        callbacks=learner.callbacks,
        accumulate_grad_batches=learner.config["num_accumulate_grad"],
        val_check_interval=learner.config["val_check_interval"],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=10000.0,
    )

    # training
    trainer.fit(
        learner.model,
        train_dataloaders=learner.train_dl,
        val_dataloaders=learner.valid_dl,
    )
    return learner


#------------------------ Finetuning -----------------------------------------#
callbacks = get_callbacks(config)
learner = Learner(model, train_dl, valid_dl, config, callbacks)
learner = train(learner)
#------------------------ Done -----------------------------------------------#
