
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
      x=x.flatten(2) #(B,D,P,P)
      x=x.transpose(1,2)

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, heads=12, attn_p=0.,proj_p=0.,qkv_bias=True):
    super(MultiHeadAttention,self).__init__()
    assert(d_model % heads == 0)
    self.d_model = d_model
    self.heads = heads
    self.head_dim = d_model // heads

    self.qkv=nn.Linear(d_model,3*d_model,bias=qkv_bias)
    self.out_fc=nn.Linear(d_model,d_model,bias=qkv_bias)
    self.attn_p=nn.Dropout(attn_p)
    self.resid_dropout=nn.Dropout(proj_p)



  def forward(self,x):
    batch_size, seq_len, dim=x.shape

    assert dim == self.d_model
    qkv=self.qkv(x)

    q=self.query(q).view(batch_size,q_len,self.heads,self.head_dim)
    k=self.key(k).view(batch_size,k_len,self.heads,self.head_dim)
    v=self.value(v).view(batch_size,v_len,self.heads,self.head_dim)

    energy=torch.einsum("nqhd,nkhd->nhqk",[q,k])

    if mask is not None:
      energy=energy.masked_fill(mask==0,-torch.inf)

    attention=torch.softmax(energy/(self.head_dim**(1/2)),dim=-1)
    attention=self.attn_p(attention)

    out=torch.einsum("nhql,nlhd->nqhd",[attention,v])
    out=out.reshape(batch_size,q_len,self.d_model)
    out=self.out_fc(out)
    out=self.resid_dropout(out)
    return out
