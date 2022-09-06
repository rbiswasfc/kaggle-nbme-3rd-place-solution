import argparse
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from datasets import load_from_disk
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          get_cosine_schedule_with_warmup, get_scheduler)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast
from transformers.trainer_pt_utils import get_parameter_names

root_path = str(Path(__file__).parents[1])
sys.path.insert(0, root_path)

try:
    from components.data_ingestion import ingest_data
    from components.data_processing import DataProcessor
    from components.dataloader_nbme import DataCollatorForMTLNeo
    from components.dataset_nbme import NbmeMTLDatasetNeo
    from components.eval_nbme import scorer
    from components.optim_utils import get_scheduler
except Exception as e:
    traceback.print_exc()
    print(e)

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


#------------------------ Unlabeled Data ---------------------------------------#
def process_feature_text(text):
    return re.sub('-', ' ', text)


def load_unlabelled_data(config):
    data_dir = config["data_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, config["train_path"]))
    features_df = pd.read_csv(os.path.join(data_dir, config["features_path"]))
    features_df["feature_text"] = features_df["feature_text"].apply(process_feature_text)

    notes_df = pd.read_csv(os.path.join(data_dir, config["notes_path"]))

    train_patients = set(train_df["pn_num"].unique())
    unlabelled_notes_df = notes_df[~notes_df["pn_num"].isin(train_patients)].copy()
    unlabelled_df = pd.merge(unlabelled_notes_df, features_df, on=['case_num'], how='left')
    print(f"# unlabelled examples = {len(unlabelled_df)}\n")
    return unlabelled_df


def get_unlabelled_dataset(config):
    """load and sample unlabelled data and create a dataset
    :param config: cosine config
    :type config: dict
    :return: unlabelled dataset
    :rtype: Dataset
    """
    ########### Load Unlabelled Data #################
    print("loading unlabelled data...")
    df = load_unlabelled_data(config)
    required_examples = config["num_unlabelled"]
    df = df.sample(required_examples)
    keep_cols = ["pn_history", "feature_text", "feature_num"]
    df = df[keep_cols].copy()
    print("unlabelled data loaded and sampled ...")
    print(f"shape of sampled unlabelled data: {df.shape}")

    ########### Create Unlabelled Dataset ##############
    dataset_creator = NbmeMTLDatasetNeo(config)
    pseudo_ds = dataset_creator.get_dataset(df, mode='infer')
    return pseudo_ds


@dataclass
class DataCollatorForMPL(DataCollatorForTokenClassification):
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
        buffer = [feature["sequence_ids"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        # create masks
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]

        # main labels
        masks = []  # (batch, seq_len)

        for seq in buffer:
            padding_length = sequence_length - len(seq)
            mask = [seq_id == 1 for seq_id in seq] + [False]*padding_length
            masks.append(mask)
        batch['label_mask'] = masks
        batch.pop('sequence_ids', None)

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


def get_unlabelled_dataloader(config):
    """get unlabelled dataloader for training MPL

    :param config: mpl config
    :type config: dict
    """
    unlabelled_ds = get_unlabelled_dataset(config)

    ########### Generate Pseudo Labels ##################
    unlabelled_ds.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'sequence_ids']
    )
    print("using auto-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])
    data_collator = DataCollatorForMPL(tokenizer=tokenizer)

    unlabelled_dl = DataLoader(
        unlabelled_ds,
        batch_size=config["batch_size"],
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=True,
    )
    return unlabelled_dl

#------------------------ Labeled Data ---------------------------------------#


def get_train_valid_dataloaders(config):
    """get train and valid dataloader 

    :param config: cosine config
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

#------------------------ MPL Model ---------------------------------------#


class NbmeMPL(nn.Module):
    """The Multi-task NBME model class for Meta Pseudo Labels
    """

    def __init__(self, config):
        super(NbmeMPL, self).__init__()

        self.config = config

        # base transformer
        self.base_model = AutoModel.from_pretrained(
            self.config["base_model_path"],
        )
        self.base_model.gradient_checkpointing_enable()

        n_freeze = config["n_freeze"]
        print(f"setting requires grad to false for last {n_freeze} layers")
        self.base_model.embeddings.requires_grad_(False)
        self.base_model.encoder.layer[:n_freeze].requires_grad_(False)

        hidden_size = self.base_model.config.hidden_size
        num_layers_in_head = self.config["num_layers_in_head"]

        # token classification head
        self.tok_classifier = nn.Linear(
            in_features=hidden_size * num_layers_in_head,
            out_features=self.config['mtl_tok_num_labels'],
        )

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def get_logits(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

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

    def compute_loss(self, logits, labels, masks, is_soft=False):
        loss = F.binary_cross_entropy_with_logits(
            logits, labels,
            reduction='none'
        )
        if is_soft:
            # additional confidence mask
            conf_mask = torch.abs(labels - 0.5) > 0.3
            masks = torch.logical_and(masks, conf_mask)
        loss = torch.masked_select(loss, masks).mean()
        return loss

############################################################################################
# Optimizers & Schedulers
############################################################################################


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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler

############################################################################################
# Training Utilities
############################################################################################


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


def save_checkpoint(config, state, is_teacher, step):
    if is_teacher:
        os.makedirs(config["teacher_model_dir"], exist_ok=True)
        name = config["teacher_save_name"]  # + "_" + str(step)
        filename = f'{config["teacher_model_dir"]}/{name}_last.pth.tar'
    else:
        os.makedirs(config["student_model_dir"], exist_ok=True)
        name = config["student_save_name"]  # + "_" + str(step)
        filename = f'{config["student_model_dir"]}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)

############################################################################################
# Training Loop
############################################################################################


def execute_meta_training(config):
    print("=="*40)
    print("GPU utilization at the very start:")
    print_gpu_utilization()
    print("=="*40)

    #------- load student and teacher from checkpoint ----------------#

    student = NbmeMPL(config)
    teacher = NbmeMPL(config)

    # if config["load_from_stage_1"]:
    #     checkpoint = torch.load(os.path.join(config['stage_1_model_dir'], config["stage_1_model_path"]))
    #     student.load_state_dict(checkpoint["state_dict"])
    #     teacher.load_state_dict(checkpoint["state_dict"])

    #------- get optimizers for student and teacher ------------------#
    s_optimizer = get_optimizer(student, config)
    t_optimizer = get_optimizer(teacher, config)

    #------- get train, valid and unlabelled dataloaders -------------#
    unlabelled_dl = get_unlabelled_dataloader(config)
    train_dl, valid_dl = get_train_valid_dataloaders(config)

    #------- prepare accelerator  ------------------------------------#
    accelerator = Accelerator(fp16=True)
    student, teacher, s_optimizer, t_optimizer, train_dl, valid_dl, unlabelled_dl = accelerator.prepare(
        student, teacher, s_optimizer, t_optimizer, train_dl, valid_dl, unlabelled_dl
    )
    print("=="*40)
    print("GPU utilization after accelerator preparation:")
    print_gpu_utilization()
    print("=="*40)

    #------- setup schedulers  ---------------------------------------#
    num_epochs = config["num_epochs"]
    grad_accumulation_steps = config["grad_accumulation"]
    warmup_pct = config["warmup_pct"]

    n_train_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    n_mpl_steps_per_epoch = len(unlabelled_dl)//grad_accumulation_steps
    n_steps_per_epoch = max(n_train_steps_per_epoch, n_mpl_steps_per_epoch)
    num_steps = num_epochs * n_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_steps)

    s_scheduler = get_scheduler(s_optimizer, num_warmup_steps, num_steps)
    t_scheduler = get_scheduler(t_optimizer, num_warmup_steps, num_steps)

    #------- Scorer & Trackers ----------------------------------------#
    best_teacher_score = 0
    best_student_score = 0

    valid_ds_path = os.path.join(config["output_dir"], config["valid_dataset_path"])
    valid_ds = load_from_disk(valid_ds_path)

    scorer_fn = partial(
        scorer,
        valid_dataset=valid_ds,
        threshold=0.5,
        focus_seq=config["text_sequence_identifier"]
    )

    #------------- Data Iterators -------------------------------------#
    train_iter = iter(train_dl)
    unlabelled_iter = iter(unlabelled_dl)

    # ------- Training Loop  ------------------------------------------#
    for step in range(num_steps):

        #------ Reset buffers After Validation ------------------------#
        if step % config["validation_interval"] == 0:
            progress_bar = tqdm(range(min(config["validation_interval"], num_steps)))
            s_loss_meter = AverageMeter()
            t_loss_meter = AverageMeter()

        teacher.train()
        student.train()

        t_optimizer.zero_grad()
        s_optimizer.zero_grad()

        #------ Get Train & Unlabelled Batch -------------------------#
        try:
            train_b = train_iter.next()
        except Exception as e:  # TODO: change to stop iteration error
            train_b = next(train_dl.__iter__())

        try:
            unlabelled_b = unlabelled_iter.next()
        except:
            unlabelled_b = next(unlabelled_dl.__iter__())
        #######################################################
        # get teacher generated soft pseudo labels for unlabelled data
        unlabelled_b_masks = unlabelled_b["label_mask"].eq(1).unsqueeze(-1)
        t_logits_unlabelled_b = teacher.get_logits(unlabelled_b)
        pseudo_y_unlabelled_b = torch.sigmoid(t_logits_unlabelled_b)  # soft pseudo label

        #------ Train Student: With Pesudo Label Data ------------------#
        s_logits_unlabelled_b = student.get_logits(unlabelled_b)
        s_loss_unlabelled_b = student.compute_loss(
            logits=s_logits_unlabelled_b,
            labels=pseudo_y_unlabelled_b,
            masks=unlabelled_b_masks,
            is_soft=True,
        )

        # backpropagation of student loss on unlabelled data
        accelerator.backward(s_loss_unlabelled_b)
        s_optimizer.step()  # update student params
        s_scheduler.step()

        #------ Train Teacher ------------------------------------------#
        train_b_labels = train_b["labels"]
        train_b_masks = train_b_labels.gt(-0.5)

        s_logits_train_b_new = student.get_logits(train_b)
        s_loss_train_b_new = student.compute_loss(
            logits=s_logits_train_b_new,
            labels=train_b_labels,
            masks=train_b_masks,
        )
        t_loss_mpl = s_loss_train_b_new

        t_logits_train_b = teacher.get_logits(train_b)
        t_loss_train_b = teacher.compute_loss(
            logits=t_logits_train_b,
            labels=train_b_labels,
            masks=train_b_masks
        )
        t_loss = t_loss_train_b + t_loss_mpl

        # backpropagation of teacher's loss
        accelerator.backward(t_loss)
        t_optimizer.step()
        t_scheduler.step()

        #------ Progress Bar Updates ----------------------------------#
        s_loss_meter.update(s_loss_train_b_new.item())
        t_loss_meter.update(t_loss.item())

        progress_bar.set_description(
            f"STEP: {step+1:5}/{num_steps:5}. "
            f"LR: {get_lr(s_optimizer):.4f}. "
            f"TL: {t_loss_meter.avg:.4f}. "
            f"SL: {s_loss_meter.avg:.4f}. "
        )
        progress_bar.update()

        #------ Evaluation & Checkpointing -----------------------------#
        if (step + 1) % config["validation_interval"] == 0:
            progress_bar.close()

            #----- Teacher Evaluation  ---------------------------------#

            # save teacher
            accelerator.wait_for_everyone()
            teacher = accelerator.unwrap_model(teacher)
            teacher_state = {
                'step': step + 1,
                'state_dict': teacher.state_dict(),
                'optimizer': t_optimizer.state_dict(),
            }
            is_best = False
            # if teacher_lb > best_teacher_score:
            #     best_teacher_score = teacher_lb
            #     is_best = True
            save_checkpoint(config, teacher_state, is_teacher=True, step=step)

            accelerator.wait_for_everyone()
            student = accelerator.unwrap_model(student)
            student_state = {
                'step': step + 1,
                'state_dict': student.state_dict(),
                'optimizer': s_optimizer.state_dict(),
            }

            save_checkpoint(config, student_state, is_teacher=False, step=step)

            print("=="*40)
            print("GPU utilization after eval:")
            print_gpu_utilization()
            print("clearing the cache")
            torch.cuda.empty_cache()
            print_gpu_utilization()
            print("=="*40)


#------------------------ Execution ---------------------------------------#
execute_meta_training(config)
#------------------------ Done --------------------------------------------#
