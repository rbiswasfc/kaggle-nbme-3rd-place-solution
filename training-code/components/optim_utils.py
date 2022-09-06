import warnings

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

warnings.filterwarnings('ignore')


def get_optimizer_params(model, config):

    no_decay = ['bias', "LayerNorm.bias", "LayerNorm.weight"]
    base_params = model.base_model.named_parameters()

    optimizer_parameters = [
        {
            "params": [p for n, p in base_params if not any(nd in n for nd in no_decay)],
            "lr": config["encoder_lr"],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in base_params if any(nd in n for nd in no_decay)],
            "lr": config["encoder_lr"],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "base_model" not in n],
            "lr": config["decoder_lr"],
            "weight_decay": config["weight_decay"]*1e-2,
        },
    ]

    return optimizer_parameters


def get_scheduler(optimizer, scheduler_params, learning_rates=None):
    if scheduler_params["scheduler_name"] == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params["T_0"],
            eta_min=scheduler_params["min_lr"],
            last_epoch=-1,
        )
    elif scheduler_params["scheduler_name"] == "OneCycleLR":
        if isinstance(learning_rates, list):
            max_lr = learning_rates
        else:
            max_lr = scheduler_params["max_lr"]

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=scheduler_params["steps_per_epoch"],
            epochs=scheduler_params["epochs"],
            pct_start=scheduler_params["pct_start"],
            anneal_strategy=scheduler_params["anneal_strategy"],
            div_factor=scheduler_params["div_factor"],
            final_div_factor=scheduler_params["final_div_factor"],
        )
    else:
        return

    return scheduler
