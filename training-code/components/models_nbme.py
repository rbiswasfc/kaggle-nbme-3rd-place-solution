import warnings
from itertools import chain

import torch
import torch.nn as nn
import wandb
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel

from .optim_utils import get_optimizer_params, get_scheduler

warnings.filterwarnings('ignore')

########################################################################
### Model Utils ########################################################
########################################################################


def reinit_roberta(base_model, num_reinit_layers, reinit_pooler=False):
    """re-initialize top n layers of roberta model

    :param base_model: base roberta model
    :type base_model: transformer model
    :param num_reinit_layers: number of top layers to re-initialize
    :type num_reinit_layers: int
    :param reinit_pooler: whether to re-initialize the pooler, defaults to False
    :type reinit_pooler: bool, optional
    """
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    if reinit_pooler:
        print('Re-initializing Pooler Layer ...')
        base_model.pooler.dense.weight.data.normal_(mean=0.0, std=config.initializer_range)
        base_model.pooler.dense.bias.data.zero_()
        for p in base_model.pooler.parameters():
            p.requires_grad = True


def reinit_deberta(base_model, num_reinit_layers):
    """re-initialize top n layers of the base deberta model

    :param base_model: base deberta model
    :type base_model: deberta model
    :param num_reinit_layers: number of top layers to re-initialize
    :type num_reinit_layers: int
    """
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


def reinit_electra(base_model, num_reinit_layers):
    raise NotImplementedError


class ReInit:
    """helper class for re-initializing top layers of transformer models
    """

    def __init__(self, arch):
        self.arch = arch

    def reinit(self, base_model, num_reinit_layers):
        print(f'Re-initializing last {num_reinit_layers} Layers ...')

        if self.arch == 'roberta':
            reinit_roberta(base_model, num_reinit_layers)
        elif self.arch == 'deberta':
            reinit_deberta(base_model, num_reinit_layers)
        elif self.arch == 'electra':
            reinit_electra(base_model, num_reinit_layers)
        else:
            raise
########################################################################
### AWP (Adverserial Weight Perturbation) ##############################
########################################################################


class AWP:
    """Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    currently coupled with NbmeModelMTLNeo model 
    TODO: refactor to have a generic modular version
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.0005):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        tok_logits = self.model.get_logits(batch)
        tok_labels = batch["labels"]
        adv_loss = self.model.calculate_loss(tok_logits, tok_labels)
        adv_loss = adv_loss.mean()
        self.optimizer.zero_grad()
        self._restore()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

########################################################################
### NBME MODEL WITH MULTI-TASK LEARNING ################################
########################################################################


class NbmeModelMTLNeo(LightningModule):
    """The Multi-task NBME model class
    """

    def __init__(self, config, scorer=None):
        super(NbmeModelMTLNeo, self).__init__()
        self.save_hyperparameters(ignore='scorer')

        # base transformer
        self.base_model = AutoModel.from_pretrained(
            self.hparams.config["base_model_path"],
        )

        if config["gradient_checkpointing"]:
            self.base_model.gradient_checkpointing_enable()

        hidden_size = self.base_model.config.hidden_size
        num_layers_in_head = self.hparams.config["num_layers_in_head"]

        # token classification head
        self.tok_classifier = nn.Linear(
            in_features=hidden_size * num_layers_in_head,
            out_features=self.hparams.config['mtl_tok_num_labels'],
        )

        self.dropout = nn.Dropout(p=self.hparams.config["dropout"])

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        # Loss function [BCEWithLogitsLoss]
        self.tok_loss = nn.BCEWithLogitsLoss(reduction='none')

        if self.hparams.config["num_layers_reinit"] > 0:
            self._reint_base_model_layers()

        if self.hparams.config.get("freeze_lower_encoders", False):
            n_freeze = 2
            print(f"setting requires grad to false for last {n_freeze} layers")

            self.base_model.embeddings.requires_grad_(False)
            self.base_model.encoder.layer[:n_freeze].requires_grad_(False)

        self.scorer = scorer
        self.awp_flag = self.hparams.config.get("awp_flag", False)
        self.awp_trigger = self.hparams.config.get("awp_trigger", 0.865)

        self.lb = 0.0  # Eval Metric

    def _reint_base_model_layers(self):
        reinit_layers = self.hparams.config["num_layers_reinit"]
        backbone = getattr(self, 'base_model')
        ReInit(self.hparams.config["arch"]).reinit(backbone, reinit_layers)

    def forward(self, **kwargs):
        out = self.base_model(**kwargs, output_hidden_states=True)
        last_hidden_state = out["last_hidden_state"]
        all_hidden_states = out["hidden_states"]

        # token classification logits
        n = self.hparams.config["num_layers_in_head"]
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

    def training_step(self, batch, batch_idx):
        # print(f"training batch: {batch_idx}")
        tok_logits = self.get_logits(batch)
        tok_labels = batch["labels"]

        total_loss = self.calculate_loss(tok_logits, tok_labels)

        if (self.awp_flag) & (self.lb > self.awp_trigger):  # AWP Attack
            total_loss.backward()
            total_loss = self.awp.attack_backward(batch)

        self.log("train_loss", total_loss, on_step=True)
        # wandb.log({"train_loss": total_loss})
        return {"loss": total_loss}

    def training_epoch_end(self, outputs):
        ave_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_ave_loss", ave_loss, logger=True)

    def validation_step(self, batch, batch_idx):
        tok_logits = self.get_logits(batch)
        tok_labels = batch["labels"]
        total_loss = self.calculate_loss(tok_logits, tok_labels)

        main_logits = tok_logits[:, :, 0]
        main_preds = main_logits.sigmoid()

        return {"loss": total_loss, "preds": main_preds}

    def validation_epoch_end(self, outputs):
        ave_loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = [x['preds'].squeeze(-1).cpu().numpy() for x in outputs]
        preds = list(chain(*preds))

        assert self.scorer is not None, "No scoring function is provided"
        char_f1_lb = 0
        try:
            char_f1_lb = self.scorer(preds)
            self.lb = char_f1_lb
        except ValueError:
            print("Warning! scorer failed! Its okay during validation sanity checks, but not during actual validation")

        self.log("val_loss", ave_loss, logger=True)
        self.log("estimated_lb", char_f1_lb, logger=True)

        print(f"validation loss: {ave_loss}")
        print(f"estimated lb: {char_f1_lb}")
        # wandb.log({"val_loss": ave_loss})
        # wandb.log({"estimated_lb": char_f1_lb})

    def predict_step(self, batch, batch_idx):
        tok_logits = self.get_logits(batch)
        preds = torch.sigmoid(tok_logits)
        return preds

    def configure_optimizers(self):
        optimizer_parameters = get_optimizer_params(self, self.hparams.config)
        lr_list = [group.get('lr', self.hparams.config["lr"]) for group in optimizer_parameters]

        optimizer = AdamW(
            optimizer_parameters,
            lr=self.hparams.config["lr"],
            eps=self.hparams.config["eps"],
            weight_decay=self.hparams.config["weight_decay"],
        )
        if self.awp_flag:
            self.awp = AWP(self, optimizer, adv_lr=1.0, adv_eps=0.001)
            print("=="*30)
            print("AWP is injected!")
            print(f"AWP will be triggered after LB reached {self.awp_trigger}")
            print("=="*30)

        scheduler = get_scheduler(optimizer, self.hparams.config, lr_list)
        if scheduler:
            lr_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            to_return = {"optimizer": optimizer, "lr_scheduler": lr_config}
        else:
            to_return = optimizer
        return to_return


########################################################################
### Sequential Classification Head #####################################
########################################################################

class SequenceClassificationHead(nn.Module):
    """Head for sentence-level classification tasks.
    """

    def __init__(self, hidden_size, dropout_rate, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.clf_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.clf_dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.clf_dropout(x)
        x = self.out_proj(x)
        return x

########################################################################
### Relevance Model ####################################################
########################################################################


class NbmeRelevance(LightningModule):
    """NBME relevance model
    """

    def __init__(self, config):
        super(NbmeRelevance, self).__init__()
        self.save_hyperparameters()

        # base transformer
        self.base_model = AutoModel.from_pretrained(
            self.hparams.config["base_model_path"],
            add_pooling_layer=False,
        )
        if config["gradient_checkpointing"]:
            self.base_model.gradient_checkpointing_enable()

        hidden_size = self.base_model.config.hidden_size
        num_layers_in_head = self.hparams.config["num_layers_in_head"]

        num_labels = self.hparams.config['mtl_sent_num_labels']
        self.sent_classifier = SequenceClassificationHead(
            hidden_size=hidden_size,
            dropout_rate=self.hparams.config["dropout"],
            num_labels=num_labels,
        )

        self.dropout = nn.Dropout(p=self.hparams.config["dropout"])

        # Loss function [BCEWithLogitsLoss]
        self.loss = nn.BCEWithLogitsLoss()

        if self.hparams.config["num_layers_reinit"] > 0:
            self._reint_base_model_layers()

    def _reint_base_model_layers(self):
        reinit_layers = self.hparams.config["num_layers_reinit"]
        backbone = getattr(self, 'base_model')
        ReInit(self.hparams.config["backbone_arch"]).reinit(backbone, reinit_layers)

    def forward(self, **kwargs):
        out = self.base_model(**kwargs)
        last_hidden_state = out["last_hidden_state"]
        # sentence classification logits
        sent_logits = self.sent_classifier(last_hidden_state)
        return sent_logits

    def get_logits(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        sent_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return sent_logits

    def calculate_loss(self, sent_logits, sent_labels):
        # calculate loss
        sent_loss = self.loss(sent_logits, sent_labels)
        return sent_loss

    def training_step(self, batch, batch_idx):
        sent_logits = self.get_logits(batch)
        sent_labels = batch["labels"]
        loss = self.calculate_loss(sent_logits, sent_labels)
        self.log("train_loss", loss, on_step=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        ave_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_ave_loss", ave_loss, logger=True)

    def validation_step(self, batch, batch_idx):
        sent_logits = self.get_logits(batch)
        sent_labels = batch["labels"]
        loss = self.calculate_loss(sent_logits, sent_labels)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        ave_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", ave_loss, logger=True)

    def predict_step(self, batch, batch_idx):
        sent_logits = self.get_logits(batch)
        main_logits = sent_logits  # (batch, seq_len)
        preds = torch.sigmoid(main_logits).squeeze(dim=-1).to("cpu").detach().numpy()
        return preds

    def configure_optimizers(self):
        optimizer_parameters = get_optimizer_params(self, self.hparams.config)
        lr_list = [group.get('lr', self.hparams.config["lr"]) for group in optimizer_parameters]

        optimizer = AdamW(
            optimizer_parameters,
            lr=self.hparams.config["lr"],
            eps=self.hparams.config["eps"],
            weight_decay=self.hparams.config["weight_decay"],
        )
        scheduler = get_scheduler(optimizer, self.hparams.config, lr_list)
        if scheduler:
            lr_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            to_return = {"optimizer": optimizer, "lr_scheduler": lr_config}
        else:
            to_return = optimizer
        return to_return


######
