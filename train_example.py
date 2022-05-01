import copy
import torch
import torch.nn as nn

from prune import prune_network, compute_stats, copy_weights
from models.ModelShop import ModelShop
from datamodules.cifar10_datamodule import CIFAR10DataModule
from train import train_epoch, train_iteration
from learning_rate_warmup import WarmupConstantSchedule

# interface
max_iter_epoch = 160  # "Number of iterations" 112480 : iteration, 160 : epoch
train_type = "epoch" # "epoch" or "iteration"

batch_size = 64
prune_iter = 10  # "Number of prune iterations"
prune_ratio = 0.2 # "Percentage of weights to remove"
# Pm = 100% * (1 - prune_ratio) ^ prune_iter
prune_method = 'l1'
prune_type = 'global'
prune_layers_type = (nn.Conv2d)
learning_rate = 0.1
lr_decrease_iters = [56320, 84480] # 80, 120 epochs
momentum = 0.9
weight_decay = 0.0001
val_freq = 1     # 1000 for iteration, 1 for epoch setting
warmup_k = 10000 # for 10000 iterations (even if train_type is "epoch")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# prepare model
model = ModelShop(
    M_model_name = "VGG19",
    I_image_channel = 3,
    I_image_size = [32,32],
    O_num_classes = 10,
)
model = model.to(device)

model_copy = copy.deepcopy(model)

# train_method
if train_type == 'iteration':   train = train_iteration
elif train_type == 'epoch':     train = train_epoch
else:                           raise ValueError

# prepare datamodule
datamodule = CIFAR10DataModule()
datamodule.prepare_data()
datamodule.setup()

# prune model
if prune_ratio > 0:
    for prune_it in range(prune_iter):

        # prepare criterion, optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss().to(device)
        lr_schedular = WarmupConstantSchedule(optimizer, warmup_k, lr_decrease_iters)

        train(model, datamodule.train_dataloader(), criterion, optimizer, lr_schedular, max_iter_epoch, dataloader_val=datamodule.val_dataloader(), val_freq=val_freq, device=device)
        prune_network(model, prune_ratio, prune_method, prune_type, prune_layers_type)
        copy_weights(model_copy, model, prune_layers_type)

        stats = compute_stats(model, prune_layers_type)
        print(f"{prune_it+1}/{prune_iter}\ttotal_params : {stats['total_params']}\ttotal_pruned_params : {stats['total_pruned_params']}\tactual_prune_ratio : {stats['actual_prune_ratio']}")