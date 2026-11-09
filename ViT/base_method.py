import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter,defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset,Subset,DataLoader,random_split
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import pickle

__all__=[
    'plot_loss_lists'
]

def calc_loss_batch(input_batch,target_batch,model,criterion,device):
  #label_smoothing ??
  input_batch,target_batch = input_batch.to(device),target_batch.to(device)
  logits=model(input_batch)
  loss=criterion(logits,target_batch)
  return loss

def calc_loss_loader(data_loader,model,criterion,device,num_batches=None):
  total_loss=0.0
  if len(data_loader) == 0:
    print("Warning: Empty data loader")
    raise ValueError #return float("nan")
  elif num_batches is None:
    num_batches=len(data_loader)
  else:
    num_batches=min(num_batches,len(data_loader))

  for i,(src,trg) in enumerate(data_loader):
    if i >= num_batches:
      break

    loss=calc_loss_batch(src,trg,model,criterion,device)
    total_loss+=loss.item()

  return total_loss/num_batches

def evaluate_model(model,data_loader,criterion,device,eval_iter=None):
  model.eval()
  with torch.no_grad():
    loss=calc_loss_loader(data_loader,model,criterion,device,eval_iter)
  model.train()
  return loss

@torch.no_grad()
def get_accuracy(loader, model, device='cpu'):
    
    was_training = model.training
    model.eval()

    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    model.train(was_training)
    return correct / total


def plot_loss_lists(train_loss_lists, val_loss_lists, names=None, ax=None, title="Loss_list"):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if names is None:
        names = list(val_loss_lists.keys())

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, name in enumerate(names):
        color = colors[i % len(colors)]

        if name in train_loss_lists:
            ax.plot(train_loss_lists[name], color=color, linestyle='--')
        ax.plot(val_loss_lists[name], label=name, color=color, linestyle='-')

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    return ax