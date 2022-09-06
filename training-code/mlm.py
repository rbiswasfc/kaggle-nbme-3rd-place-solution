import argparse
import json
import math
import os
import random
import shutil
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
from accelerate import Accelerator
from datasets import Dataset
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          default_data_collator, get_scheduler)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import \
    DebertaV2TokenizerFast
from transformers.trainer_pt_utils import get_parameter_names

try:
    import bitsandbytes as bnb
except ImportError:
    print("Warning: bitsandbytes not available")

pd.set_option('display.max_colwidth', None)
#-----------------------------------------------------------------------#


#------------------------------- GPU utils -----------------------------#
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


#------------------------------- MLM config -----------------------------#
ap = argparse.ArgumentParser()
ap.add_argument('--config_path', type=str, required=True)
args = ap.parse_args()

with open(args.config_path, "r") as f:
    config = json.load(f)

#------------------------------- Train Data -----------------------------#
data_path = "../data/train_data/patient_notes.csv"
notes_df = pd.read_csv(data_path)
notes_df = notes_df[["pn_history"]].reset_index(drop=True)


#------------------------------- Dataset Script --------------------------#
def get_mlm_dataset(notes_df, config):
    nbme_dataset = Dataset.from_pandas(notes_df)

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])

    def tokenize_function(examples):
        result = tokenizer(examples[config["text_col"]])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_datasets = nbme_dataset.map(
        tokenize_function, batched=True, remove_columns=[config["text_col"]]
    )

    try:
        tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
    except:
        pass

    chunk_size = config["chunk_size"]

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size

        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    test_pct = config["test_pct"]
    max_train_examples = config["max_train_examples"]
    max_test_examples = int(config["max_train_examples"]*test_pct)

    test_size = int(len(lm_datasets) * test_pct)
    train_size = len(lm_datasets) - test_size

    test_size = min(test_size, max_test_examples)
    train_size = min(train_size, max_train_examples)

    downsampled_dataset = lm_datasets.train_test_split(
        train_size=train_size, test_size=test_size, seed=config["seed"]
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config["mask_probability"]
    )

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    try:
        downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    except Exception as e:
        traceback.print_exc()

    downsampled_dataset["test"] = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )

    try:
        downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
                "masked_token_type_ids": "token_type_ids",
            }
        )
    except:
        downsampled_dataset["test"] = downsampled_dataset["test"].rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )

    return downsampled_dataset


#------------------------------- Model ------------------------------------#
model = AutoModelForMaskedLM.from_pretrained(config["model_checkpoint"])
model.deberta.resize_token_embeddings()

if config["gradient_checkpointing"]:
    print("enabling gradient checkpointing")
    model.gradient_checkpointing_enable()

#------------------------------- Optimizer ------------------------------------#
if config["use_bnb_optim"]:
    print("using bnb optimizer....")
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

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
    optimizer = adam_bnb_optim
else:
    print("using AdamW optimizer....")
    optimizer = AdamW(model.parameters(), lr=config["lr"])


#------------------------------- Datasets ----------------------------------#
sampling_fraction = config["sampling_fraction"]
notes_df_sample = notes_df.sample(int(sampling_fraction*len(notes_df)))
mlm_dataset = get_mlm_dataset(notes_df_sample, config)

eval_dataset = deepcopy(mlm_dataset['test'])

#------------------------------- DataLoaders -------------------------------#
if ('v3' in config["model_checkpoint"].lower()) | ('v2' in config["model_checkpoint"].lower()):
    print("using DebertaV2TokenizerFast tokenizer")
    tokenizer = DebertaV2TokenizerFast.from_pretrained(config["model_checkpoint"])
else:
    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=config["mask_probability"]
)

batch_size = config["batch_size"]

train_dataloader = DataLoader(
    mlm_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

eval_dataloader = DataLoader(
    mlm_dataset["test"],
    batch_size=batch_size,
    collate_fn=default_data_collator,
)

print(len(mlm_dataset["train"]), len(mlm_dataset["test"]))

#------------------------------- Accelerator -----------------------------#
accelerator = Accelerator(fp16=config["fp16"])

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

print_gpu_utilization()

#------------------------------- Train Arguments ---------------------------#
num_train_epochs = config["num_epochs"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
warmup_frac = config["warmup_pct"]

num_update_steps_per_epoch = len(train_dataloader)//gradient_accumulation_steps
num_training_steps = num_train_epochs * num_update_steps_per_epoch
num_warmup_steps = int(warmup_frac*num_training_steps)

print(f"num_training_steps = {num_training_steps}")
print(f"num_warmup_steps = {num_warmup_steps}")

output_dir = config["trained_model_name"]

#------------------------------- LR Scheduler --------------------------------#
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

#------------------------------- Training -------------------------------------#
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()

    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     100.0
            # )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        if step % config["eval_frequency"] == 0:
            model.eval()
            losses = []
            n_correct = 0
            n_total = 0

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                    tok_preds = torch.max(outputs['logits'], dim=-1)[1]
                    tok_labels = batch['labels']
                    curr = torch.masked_select(tok_preds == batch['labels'], batch['labels'] > -100).sum()
                    tot = torch.masked_select(tok_preds == batch['labels'], batch['labels'] > -100).size(0)
                    n_correct += curr
                    n_total += tot

                loss = outputs.loss
                losses.append(accelerator.gather(loss.repeat(batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]

            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")

            accuracy = round((n_correct*100/n_total).item(), 2)
            print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
            print(f">>> Epoch {epoch}: Accuracy: {accuracy}")

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
            torch.cuda.empty_cache()
            model.train()
#------------------------------- Done ----------------------------------------#
