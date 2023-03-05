from model import Model
from dataloader import PGOGraphDataset
import torch.nn as nn
import torch
import numpy as np
import random
import subprocess

from torch.utils.data import random_split
from torch.utils.data import Dataset


def do_shell_command_call(cmd):
    command = (str(cmd))
    # print (command)
    subprocess.call(command, shell=True)


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class PGOConfig:
    # Model Configurations
    conv_layer: str = "transformer"
    jkn_mode: str = "max"
    activation: str = "relu"
    node_attention: bool = True

    # node feature length
    # in_channels
    in_features: int = 1433

    hidden_features: int = 32
    out_features: int = 64
    num_heads: int = 2
    num_layers: int = 5
    num_classes: int = 5
    # Dropout probability
    dropout: float = 0.6

    # train version
    train_version: str
    # Dataset
    dataset: PGOGraphDataset
    # dataset location
    train_dataset_path: str = "train_dataset/"
    # seed for all
    seed: int = 42
    # dataloader worker num
    dl_num_workers: int = 8
    # train split percentage
    train_percent: float = 0.8

    # task type: 0->classification 1->regression
    task: int = 0

    # Number of training iterations
    epochs: int = 1_000
    # learning rate
    lr: float = 0.005
    # Loss function
    loss_func_0 = nn.CrossEntropyLoss()
    loss_func_1 = nn.MSELoss()
    # Device to train on
    gpu: int = 0
    device: torch.device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    # Optimizer
    optimizer: str = "Adam"

    log_path: str = "logs/"

    def gen_config(self):
        mark = "tver_"+self.train_version+"_"\
                "opt_" + self.optimizer + "_" \
                "conv_"+self.conv_layer+"_" \
                "nconv_"+str(self.num_layers)+"_"\
                "jkn_"+self.jkn_mode+"_"\
                "act_"+self.activation+"_"\
                "noatt_"+str(self.node_attention)+"_" \
                "drop_" + str(self.dropout)+"_"\
                "task_" + str(self.task) + "_" \
                "epoch_" + str(self.epochs) + "_" \
                "lr_" + str(self.lr) + "_" \
                "tp_" + str(self.train_percent) + "_" \
                "seed_" + str(self.seed) + "_" \
                "if_" + str(self.in_features) + "_" \
                "hf_" + str(self.hidden_features) + "_" \
                "of_" + str(self.out_features) + "_" \
                "nh_" + str(self.num_heads) + "_" \
                "nl_" + str(self.num_layers) + "_" \
                "nc_" + str(self.num_classes)
        return mark
