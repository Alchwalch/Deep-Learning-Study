
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
import regex as re
import pickle

__all__ =[
    'compute_mean_attention_dist',
    'plot_attention_distance',
    'plot_attention_map',
    'plot_loss_lists'
]

def init_t_xy(end_x: int, end_y: int):
  t = torch.arange(end_x * end_y,dtype=torch.float32)
  t_x = (t % end_x).float()
  t_y = torch.div(t, end_x, rounding_mode="floor").float()

  return t_x, t_y

def compute_distance_matrix(length):
  t_x, t_y = init_t_xy(length, length)
  coords = torch.stack([t_x, t_y], dim=1)

  diff = coords[:, None, :] - coords[None, :, :]
  dist = torch.sqrt((diff ** 2).sum(dim=-1))
  return dist

def compute_mean_attention_dist(scores, patch_size=1):
  # scores: (n_layers, B, heads, 1+N, 1+N), softmax 이후 값
  scores = scores[..., 1:, 1:]                        # (n_layers, B, heads, N, N)
  #scores = scores / scores.sum(dim=-1, keepdim=True)   재정규화는 건너뛰기

  num_patches = scores.shape[-1]
  length = math.isqrt(num_patches)
  assert length ** 2 == num_patches, "Num patches is not perfect square"

  dist = compute_distance_matrix(length).to(scores.device) * patch_size
  dist = dist[None, None, None]                       # (1, 1, 1, N, N)

  mean_dist = (scores * dist).sum(dim=-1)             # (n_layers, B, heads, N)
  return mean_dist.mean(dim=-1).mean(dim=1)           # (n_layers, heads)

def plot_attention_distance(
    dist,
    title="ViT",
    ax=None,
    legend_heads=3
):
    dist = (
        dist.detach().cpu().numpy()
        if hasattr(dist, "detach")
        else np.asarray(dist)
    )

    n_layers, n_heads = dist.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.get_cmap("tab20" if n_heads > 10 else "tab10")
    layers = np.arange(1, n_layers + 1)

    for h in range(n_heads):
        label = f"Head {h + 1}" if h < legend_heads else None

        ax.plot(
            layers,
            dist[:, h],
            marker="o",
            markersize=3,
            linewidth=1,
            color=cmap(h % cmap.N),
            label=label
        )

    ax.set_title(title)
    ax.set_xlabel("Network depth (layer)")
    ax.set_ylabel("Mean attention distance (pixels)")
    ax.set_xlim(0, n_layers + 1)
    ax.set_ylim(bottom=0)

    if legend_heads > 0:
        ax.legend(loc="best")

    return ax

def attention_rollout(scores):
    n_layers, B, H, T, _ = scores.shape
    eye=torch.eye(T,device=scores.device)
    rollout=eye.expand(B,-1,-1)

    for attn in scores:
        attn = attn.mean(dim=1)
        attn = 0.5*attn + 0.5*eye
        rollout = attn @ rollout
    return rollout #(B,T,T)

def plot_attention_map(images, targets, scores, logits,
                       class_names=None, mean=None, std=None,
                       max_images=4, mode="mask"):
  # images : (B, C, H, W)
  # targets: (B,)
  # scores : (n_layers, B, heads, N+1, N+1)
  # logits : (B, n_classes)
  # mode   : "mask"(이미지에 mask 곱하기, 논문 Fig.14 스타일) | "heatmap"(jet 오버레이)

  B, C, H, W = images.shape
  n = min(B, max_images)

  images = images.detach().cpu().float()
  if mean is not None and std is not None:
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    images = images * s + m
  imgs = images.clamp(0, 1).permute(0, 2, 3, 1).numpy()      # (B, H, W, C)
  if C == 1:
    imgs = np.repeat(imgs, 3, axis=-1)

  with torch.no_grad():
    rollout = attention_rollout(scores.detach())             # (B, 1+N, 1+N)
  mask = rollout[:, 0, 1:]                                   # (B, N)
  L = math.isqrt(mask.size(-1))
  assert L * L == mask.size(-1), "Num patches is not perfect square"

  mask = mask.reshape(B, 1, L, L)
  mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)  # (B,1,H,W)

  # 이미지 정규화
  mn = mask.amin(dim=(1, 2, 3), keepdim=True)
  mx = mask.amax(dim=(1, 2, 3), keepdim=True)
  mask = ((mask - mn) / (mx - mn + 1e-8)).squeeze(1).cpu().numpy()   # (B, H, W)

  probs = torch.softmax(logits.detach().cpu().float(), dim=-1)
  conf, preds = probs.max(dim=-1)
  targets = targets.detach().cpu()
  name = (lambda i: class_names[int(i)]) if class_names is not None else (lambda i: str(int(i)))

  fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
  cmap = plt.get_cmap("jet")

  for i in range(n):
    if mode == "mask":
      overlay = imgs[i] * mask[i][..., None]
    else:
      heat = cmap(mask[i])[..., :3]
      overlay = 0.5 * imgs[i] + 0.5 * heat

    axes[i, 0].imshow(imgs[i])
    axes[i, 0].set_title(f"Target: {name(targets[i])}", fontsize=10)

    correct = int(preds[i]) == int(targets[i])
    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f"Pred: {name(preds[i])} ({conf[i]:.0%})",
                         fontsize=10, color="green" if correct else "red")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

  plt.tight_layout()
  return fig

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
