
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

class PatchEmbed(nn.Module):
    def __init__(self,img_size,patch_size,in_channels=3,embed_dim=768):
      super().__init__()

      assert patch_size > 0
      assert img_size % patch_size == 0

      self.img_size=img_size
      self.patch_size=patch_size
      self.n_patches=(img_size//patch_size)**2

      self.proj=nn.Conv2d(
          in_channels,
          embed_dim,
          kernel_size=patch_size,
          stride=patch_size
      )

    def forward(self,x):
      x=self.proj(x)
      x=x.flatten(2) #(B,D,P,P) -> #(B,D,P^2)
      x=x.transpose(1,2)
      return x

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, heads=12, attn_p=0.,proj_p=0.,qkv_bias=True):
    super(MultiHeadAttention,self).__init__()
    assert(d_model % heads == 0)
    self.d_model = d_model
    self.heads = heads
    self.head_dim = d_model // heads
    self.attn_p = attn_p

    self.qkv=nn.Linear(d_model,3*d_model,bias=qkv_bias)
    self.out_fc=nn.Linear(d_model,d_model,bias=qkv_bias)
    self.attn_dropout=nn.Dropout(attn_p)
    self.resid_dropout=nn.Dropout(proj_p)

  def forward(self,x,return_attn=False):
    B, C, D=x.shape
    assert D == self.d_model

    qkv=self.qkv(x).reshape(B,C,3,self.heads,self.head_dim)
    q, k, v=qkv.permute(2,0,3,1,4)

    if return_attn:
      energy=q @ k.transpose(-1,-2)/(self.head_dim**(1/2))
      score=torch.softmax(energy,dim=-1)
      out=self.attn_dropout(score) @ v
    else:
      out= F.scaled_dot_product_attention(
          q,k,v,
          attn_mask=None,
          dropout_p=self.attn_p if self.training else 0.0,
          is_causal=False
      )
      score = None

    out=out.transpose(1,2).reshape(B,C,D)
    out=self.out_fc(out)
    out=self.resid_dropout(out)
    return out,score

class FeedForward(nn.Module):
  def __init__(self,d_model,drop_p=0.,bias=True):
    super(FeedForward,self).__init__()
    self.layers=nn.Sequential(
        nn.Linear(d_model,d_model*4,bias=bias),
        nn.GELU(),
        nn.Dropout(drop_p),
        nn.Linear(d_model*4,d_model,bias=bias),
        nn.Dropout(drop_p)
    )

  def forward(self,x):
    x=self.layers(x)
    return x

class TransformerBlock(nn.Module):
  def __init__(self,cfg):
    super(TransformerBlock,self).__init__()
    self.norm1=nn.LayerNorm(cfg.d_model,bias=cfg.bias)
    self.attn=MultiHeadAttention(
        d_model=cfg.d_model,
        heads=cfg.heads,
        attn_p=cfg.attn_p,
        proj_p=cfg.proj_p,
        qkv_bias=cfg.bias
    )
    self.norm2=nn.LayerNorm(cfg.d_model,bias=cfg.bias)
    self.mlp=FeedForward(
        d_model=cfg.d_model,
        bias=cfg.bias
    )

  def forward(self,x,return_attn=False):
    attn_out,score=self.attn(self.norm1(x),return_attn)
    x=x+attn_out
    x=x+self.mlp(self.norm2(x))
    return x,score

class VisionTransformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.patch_embed = PatchEmbed(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.d_model
    )
    num_patches = (cfg.img_size // cfg.patch_size) ** 2

    self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
    self.pos_embed = nn.Embedding(1 + num_patches, cfg.d_model)
    self.pos_drop = nn.Dropout(cfg.pos_p)

    self.blocks = nn.ModuleList(
        [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
    )
    self.norm = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
    self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    self.apply(self._init_weights)
    nn.init.trunc_normal_(self.cls_token, std=0.02)

  def _init_weights(self, module):
    if isinstance(module, VisionTransformer):
      nn.init.normal_(module.cls_token, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
      nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
      nn.init.ones_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)

  def forward(self, x,return_attn=False):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.size(0), -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x=x+self.pos_embed(torch.arange(x.size(1),device=x.device))
    x=self.pos_drop(x)

    scores=[] if return_attn else None
    for block in self.blocks:
      x,score=block(x,return_attn)
      if return_attn:
        scores.append(score)

    x=self.norm(x)
    cls_token_final=x[:,0]
    logits= self.head(cls_token_final)
    return (logits,scores) if return_attn else logits
