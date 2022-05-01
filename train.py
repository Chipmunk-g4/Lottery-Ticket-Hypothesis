import torch
import torch.nn as nn
import tqdm
from utils import loop_dataloader

def train_iteration(
    model,
    dataloader_train,
    loss_inst,
    optimizer,
    lr_schedular,
    max_iter=10_000,
    dataloader_val=None,
    val_freq=500,
    device = 'cpu',
):
    iterable = loop_dataloader(dataloader_train)
    iterable = tqdm.tqdm(iterable, total=max_iter)
    acc = torch.tensor([0])
    accs = []

    model.train()

    it = 0
    for X_batch, y_batch in iterable:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if it == max_iter:
            break

        logit_batch = model(X_batch)

        loss = loss_inst(logit_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedular.step()

        if (it % val_freq == 0 and dataloader_val is not None):
            model.eval()

            is_equal = []

            for X_batch_val, y_batch_val in dataloader_val:

                X_batch_val = X_batch_val.to(device)
                y_batch_val = y_batch_val.to(device)

                is_equal.append(
                    model(X_batch_val).argmax(dim=-1) == y_batch_val
                )

            is_equal_t = torch.cat(is_equal)
            acc = is_equal_t.sum() / len(is_equal_t) 
            accs.append(acc.item())

        it += 1

        iterable.set_postfix({'lr': optimizer.param_groups[0]['lr'],'loss': loss.item(), 'acc': acc.item()})
    print(f"max acc : {max(accs)}")

def train_epoch(
    model,
    dataloader_train,
    loss_inst,
    optimizer,
    lr_schedular,
    max_epoch=100,
    dataloader_val=None,
    val_freq=1,
    device = 'cpu',
):
    epoch_loop = tqdm.tqdm(range(max_epoch), leave=True)
    # train_data_loop = tqdm.tqdm(dataloader_train)
    acc = torch.tensor([0])
    accs = []

    for e in epoch_loop:
        
        # train
        train_data_loop = tqdm.tqdm(dataloader_train, leave=False)
        model.train()
        for X_batch, y_batch in train_data_loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logit_batch = model(X_batch)
            loss = loss_inst(logit_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedular.step()

            train_data_loop.set_postfix({'lr': optimizer.param_groups[0]['lr'],'loss': loss.item()})

        # evaluation
        if e % val_freq == 0 and dataloader_val is not None:
            model.eval()

            is_equal = []

            for X_batch_val, y_batch_val in dataloader_val:

                X_batch_val = X_batch_val.to(device)
                y_batch_val = y_batch_val.to(device)

                is_equal.append(
                    model(X_batch_val).argmax(dim=-1) == y_batch_val
                )

            is_equal_t = torch.cat(is_equal)
            acc = is_equal_t.sum() / len(is_equal_t) 
            accs.append(acc.item())

        epoch_loop.set_postfix({'lr': optimizer.param_groups[0]['lr'], 'acc': acc.item(), 'max_acc' : max(accs)})
    print(f"max acc : {max(accs)}")