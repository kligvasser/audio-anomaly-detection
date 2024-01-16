import copy
import logging
import torch
from ast import literal_eval

import models

from torchvision import models as torch_models


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_model_entire(model, path):
    torch.save(model, path)


def load_model(args, path=None):
    model = models.__dict__[args.model]

    if args.model_config != "":
        model_config = dict({}, **literal_eval(args.model_config))
    else:
        model_config = {}

    model = model(**model_config)

    if path:
        logging.info("Loading model from {}...".format(path))
        model.load_state_dict(torch.load(path))

    return model


def load_model_entire(path):
    model = torch.load(path)
    model.eval()

    return model


def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])
