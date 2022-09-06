import argparse
import gc
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from itertools import chain

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from datasets import load_from_disk
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import CSVLogger
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModel, DataCollatorForTokenClassification,
                          get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup, get_scheduler)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast
from transformers.trainer_pt_utils import get_parameter_names

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
config["lr"] = 2e-6
config["weight_decay"] = 1e-4
config["epochs"] = 3
config["num_accumulate_grad"] = 1
config["val_check_interval"] = 0.25
config["patience"] = 12
config["encoder_lr"] = 2e-6
config["decoder_lr"] = 2e-6
config["model_dir"] = "../dev-models/delv3/"
config["output_dir"] = "../outputs/finetue_outputs"
config["awp_flag"] = False
config["awp_trigger"] = 1.0
config["freeze_lower_encoders"] = True
config["grad_accumulation"] = 1
config["batch_size"] = 16
config['max_length'] = 480
config["add_markers"] = False

#------------------------ Model --------------------------------------------#


class NbmeModelMTLNeoTorch(nn.Module):
    """The Multi-task NBME model class
    """

    def __init__(self, config, scorer=None):
        super(NbmeModelMTLNeoTorch, self).__init__()
        self.config = config

        # base transformer
        self.base_model = AutoModel.from_pretrained(
            self.config["base_model_path"],
        )

        if config["gradient_checkpointing"]:
            self.base_model.gradient_checkpointing_enable()

        hidden_size = self.base_model.config.hidden_size
        num_layers_in_head = self.config["num_layers_in_head"]

        # token classification head
        self.tok_classifier = nn.Linear(
            in_features=hidden_size * num_layers_in_head,
            out_features=self.config['mtl_tok_num_labels'],
        )

        self.dropout = nn.Dropout(p=self.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        # Loss function [BCEWithLogitsLoss]
        self.tok_loss = nn.BCEWithLogitsLoss(reduction='none')

        if self.config["num_layers_reinit"] > 0:
            self._reint_base_model_layers()

        if self.config.get("freeze_lower_encoders", False):
            n_freeze = config.get("n_freeze", 2)
            print(f"setting requires grad to false for last {n_freeze} layers")

            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:n_freeze].requires_grad_(False)

    def _reint_base_model_layers(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        out = self.base_model(**kwargs, output_hidden_states=True)
        last_hidden_state = out["last_hidden_state"]
        all_hidden_states = out["hidden_states"]

        # token classification logits
        n = self.config["num_layers_in_head"]
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

    def calculate_loss(self, tok_logits, tok_labels):
        # calculate losses

        # token classification loss
        main_task_logits = tok_logits[:, :, 0]
        main_task_labels = tok_labels[:, :, 0]
        main_task_loss = self.tok_loss(
            main_task_logits.view(-1, 1), main_task_labels.view(-1, 1)
        )
        main_task_loss = torch.masked_select(main_task_loss, main_task_labels.view(-1, 1) > -1).mean()

        #  classification loss on auxiliary tasks
        additional_task_logits = tok_logits[:, :, 1:]
        additional_task_labels = tok_labels[:, :, 1:]
        additional_task_loss = self.tok_loss(
            additional_task_logits.reshape(-1, 1), additional_task_labels.reshape(-1, 1)
        )
        additional_task_loss = torch.masked_select(
            additional_task_loss, additional_task_labels.reshape(-1, 1) > -1).mean()

        total_loss = main_task_loss + 2.0*additional_task_loss
        return total_loss

#------------------------ Optimizer & Scheduler ---------------------------------#


def get_optimizer(model, config):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # print(decay_parameters)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # print(decay_parameters)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (config["beta1"], config["beta2"]),
        "eps": config['eps'],
    }

    optimizer_kwargs["lr"] = config["lr"]

    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        lr=config['lr'],
    )

    return adam_bnb_optim


def get_scheduler(optimizer, warmup_steps, total_steps):
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        # num_training_steps=total_steps
    )
    return scheduler

#------------------------ Train Loop Utils ---------------------------------#


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def save_checkpoint(config, state, is_best):
#     os.makedirs(config["model_dir"], exist_ok=True)
#     name = "finetuned_model"
#     filename = f'{config["student_model_dir"]}/{name}_last.pth.tar'
#     torch.save(state, filename, _use_new_zipfile_serialization=False)
#     if is_best:
#         shutil.copyfile(filename, f'{config["model_dir"]}/{name}_best.pth.tar')


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
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


# train_dl, valid_dl = get_train_valid_dataloaders(config)

#------------------------ Train Loop -----------------------------------------#
def execute_finetuning_with_swa(config):

    print("=="*40)
    print("GPU utilization at the very start:")
    print_gpu_utilization()
    print("=="*40)

    #------- initialize model and load weights from checkpoint ----------------#
    model = NbmeModelMTLNeoTorch(config)  # tokenizer will be resized
    checkpoint_path = config["student_path"]

    # load state dict
    model_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print(f"loading student from from previous checkpoints")
    model.load_state_dict(model_checkpoint["state_dict"])

    # print("resizing model embeddings for the marker tokens")
    # config["len_tokenizer"] = 128011  # IMP STEP: RESIZE MODEL EMBEDDINGS
    # model.base_model.resize_token_embeddings(config["len_tokenizer"])

    del model_checkpoint
    gc.collect()

    #------- initialize average model for SWA ---------------------------------#
    swa_model = AveragedModel(model)

    #------- optimizer --------------------------------------------------------#
    ft_optimizer = get_optimizer(model, config)

    #------- get train and valid dataloaders ----------------------------------#
    train_dl, valid_dl = get_train_valid_dataloaders(config)

    #------- prepare accelerator  ---------------------------------------------#

    accelerator = Accelerator(fp16=True)
    model, ft_optimizer, train_dl, valid_dl = accelerator.prepare(
        model, ft_optimizer, train_dl, valid_dl)

    print("=="*40)
    print("GPU utilization after accelerator preparation:")
    print_gpu_utilization()
    print("=="*40)

    #------- setup schedulers  ------------------------------------------------#
    num_epochs = config["epochs"]
    grad_accumulation_steps = 1
    warmup_pct = config["warmup_pct"]

    n_train_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * n_train_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)
    scheduler = get_scheduler(ft_optimizer, num_warmup_steps, num_training_steps)

    #------- SWA LR & CONFIG --------------------------------------------------#

    swa_scheduler = SWALR(ft_optimizer, swa_lr=5e-7, anneal_epochs=20)
    swa_start = 399  # Start SWA from start

    #------- Scorer & Trackers ------------------------------------------------#
    best_score = 0
    valid_ds_path = os.path.join(config["output_dir"], config["valid_dataset_path"])
    valid_ds = load_from_disk(valid_ds_path)
    scorer_fn = partial(
        scorer,
        valid_dataset=valid_ds,
        threshold=0.5,
        focus_seq=config["text_sequence_identifier"]
    )

    #------------- Data Iterators ---------------------------------------------#
    train_iter = iter(train_dl)

    # ------- Training Loop  --------------------------------------------------#
    model.train()

    for step in range(num_training_steps):

        #------ Reset buffers After Validation --------------------------------#
        if step % config["validation_interval"] == 0:
            progress_bar = tqdm(range(min(config["validation_interval"], num_training_steps)))
            loss_meter = AverageMeter()

        model.train()
        ft_optimizer.zero_grad()

        #------ Get Train Batch -----------------------------------------------#
        try:
            train_b = train_iter.next()
        except Exception as e:  # TODO: change to stop iteration error
            train_b = next(train_dl.__iter__())

        #------- Training Steps -----------------------------------------------#
        # get loss of current student on labelled train data
        m_logits_train_b = model.get_logits(train_b)

        # get loss of current student on labelled train data
        train_b_labels = train_b["labels"]
        m_loss_train_b = model.calculate_loss(
            tok_logits=m_logits_train_b,
            tok_labels=train_b_labels,
        )

        # backpropagation of model loss
        accelerator.backward(m_loss_train_b)
        ft_optimizer.step()  # update student params

        if step < swa_start:
            scheduler.step()

        #------ Progress Bar Updates ----------------------------------#
        loss_meter.update(m_loss_train_b.item())

        progress_bar.set_description(
            f"STEP: {step+1:5}/{num_training_steps:5}. "
            f"LR: {get_lr(ft_optimizer):.4f}. "
            f"LOSS: {loss_meter.avg:.4f}. "
        )
        progress_bar.update()

        #------ Evaluation & Checkpointing -----------------------------#
        if (step + 1) % config["validation_interval"] == 0:
            progress_bar.close()

            #----- Model Evaluation  ---------------------------------#
            model.eval()
            model_preds = []

            with torch.no_grad():
                for batch in valid_dl:
                    p = model.get_logits(batch)
                    model_preds.append(p)
            model_preds = [torch.sigmoid(p).detach().cpu().numpy()[:, :, 0] for p in model_preds]
            model_preds = list(chain(*model_preds))
            model_lb = scorer_fn(model_preds)
            print(f"After step {step+1} Model LB: {model_lb}")

            # save model
            accelerator.wait_for_everyone()
            model = accelerator.unwrap_model(model)
            model_state = {
                'step': step + 1,
                'state_dict': model.state_dict(),
                'lb': model_lb
            }
            is_best = False
            if model_lb > best_score:
                best_score = model_lb
                is_best = True
            # save_checkpoint(config, model_state, is_best=is_best)

            # SWA RELATED COMPUTES
            if step >= swa_start:
                print("updating SWA model and taking swalr step")
                model.to('cpu')
                swa_model.update_parameters(model)
                swa_scheduler.step()

                # SAVE SWA MODEL
                os.makedirs(config["model_dir"], exist_ok=True)
                # name = "swa_model_average"
                # filename = f'{config["model_dir"]}/{name}_model.pth.tar'
                filename = f'{config["model_dir"]}/{config["model_name"]}.pth.tar'

                swa_state = {
                    'step': step + 1,
                    'state_dict': swa_model.state_dict(),
                }
                print("saving swa state...")
                torch.save(swa_state, filename, _use_new_zipfile_serialization=False)
                model = accelerator.prepare(model)

            print("=="*40)
            print("GPU utilization after eval:")
            print_gpu_utilization()
            print("clearing the cache")
            torch.cuda.empty_cache()
            print_gpu_utilization()
            print("=="*40)


execute_finetuning_with_swa(config)

# procss checkpoint
if config["process_ckpt"]:
    checkpoint_path = f'{config["model_dir"]}/{config["model_name"]}.pth.tar'
    ckpt = torch.load(checkpoint_path)
    print("removing module from keys...")
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    processed_state = {"state_dict": new_state_dict}
    torch.save(processed_state, checkpoint_path, _use_new_zipfile_serialization=False)
