# -*- encoding: utf-8 -*-
"""
@File    :   opti.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/18 18:48   thgpddl      1.0         None
"""
import torch
from torch import nn
from utils.utils import cross_entropy


def get_loss_fn(config):
    if config['label_smooth']:
        loss_fn = cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def get_opti(config, model):
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'], nesterov=True)

    if config['scheduler'] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    return optimizer, scheduler
