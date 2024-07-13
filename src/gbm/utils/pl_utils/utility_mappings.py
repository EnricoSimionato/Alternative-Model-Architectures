import torch

optimizers_mapping = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "adamax": torch.optim.Adamax
}
