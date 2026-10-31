
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
import os

__all__=[
  'train_model',
  'load_ckpt',
  'load_best'
]

META_KEYS = ('best_losses', 'no_improves', 'es', 'train_loss_lists', 'val_loss_lists')

def save_ckpt(
    path,
    models, optimizers, schedulers,
    epoch, model_idx, batch_offset, running_loss,
    base_seed=67,
    best_losses=None, no_improves=None, es=None,
    train_loss_lists=None, val_loss_lists=None,
    device='cpu'
):
  if path is None:
    print(f"{path}에 대한 경로 없음.")
    return

  p = os.path.join(path, 'ckpt.pt')

  ckpt = {
      'epoch': epoch, 'model_idx': model_idx, 'batch_offset': batch_offset,
      'running_loss': running_loss, 'base_seed': base_seed,
      'models': {n: m.state_dict() for n, m in models.items()},
      'optimizers': {n: o.state_dict() for n, o in optimizers.items()},
      'schedulers': {n: s.state_dict() for n, s in schedulers.items()},
  }

  if best_losses is not None:
    ckpt.update(best_losses=best_losses, no_improves=no_improves, es=es)
  if train_loss_lists is not None:
    ckpt.update(train_loss_lists=train_loss_lists, val_loss_lists=val_loss_lists)

  missing = [k for k in META_KEYS if k not in ckpt]
  if missing and os.path.exists(p):
    try:
      old = torch.load(p, map_location='cpu', weights_only=False)
      for k in missing:
        if k in old:
          ckpt[k] = old[k]
      del old
    except Exception as e:
      print(f"[save_ckpt] 기존 ckpt 읽기 실패, 메타 정보 (loss_list, early stopping) 없이 저장: {e}")

  # 원자적 저장: 저장 도중 끊겨도 기존 ckpt.pt가 안 깨짐 (클로드 감사감사)
  torch.save(ckpt, p + '.tmp')
  os.replace(p + '.tmp', p)

def load_ckpt(path,models,optimizers,schedulers,
              base_seed=67, device='cpu'):

  if path is None:
    print("경로 없음. 처음부터 시작")
    return 0, 0, 0, 0.0, base_seed, None, None, None, None, None

  p = os.path.join(path, 'ckpt.pt')

  if not os.path.exists(p):
    print("체크포인트 없음, 처음부터 시작")
    return 0, 0, 0, 0.0, base_seed, None, None, None, None, None

  ckpt=torch.load(p,map_location=device,weights_only=False)

  for name in models.keys():
    models[name].load_state_dict(ckpt['models'][name])
    optimizers[name].load_state_dict(ckpt['optimizers'][name])
    schedulers[name].load_state_dict(ckpt['schedulers'][name])

  resume_epoch=ckpt['epoch']
  resume_model_idx=ckpt['model_idx']
  batch_offset=ckpt['batch_offset']
  running_loss=ckpt['running_loss']
  base_seed=ckpt['base_seed']

  if 'best_losses' in ckpt:
    best_losses=ckpt['best_losses']
    no_improves=ckpt['no_improves']
    es=ckpt['es']

  else:
    best_losses, no_improves, es = None, None, None

  if 'train_loss_lists' in ckpt:
    train_loss_lists=ckpt['train_loss_lists']
    val_loss_lists=ckpt['val_loss_lists']

  else:
    train_loss_lists, val_loss_lists = None, None

  return resume_epoch,resume_model_idx,batch_offset,running_loss,base_seed,best_losses,no_improves,es,train_loss_lists,val_loss_lists

def save_best(path,name,model):

  if path is None:
    print(f"{path}에 대한 경로 없음.")
    return

  p=os.path.join(path,f'{name}_best.pt')

  torch.save({
      'model_state_dict':model.state_dict()
  },p+ '.tmp')
  os.replace(p + '.tmp', p)

def load_best(path,name,model,device):

  if path is None:
    print(f"{path}에 대한 경로 없음.")
    return

  p=os.path.join(path,f'{name}_best.pt')

  if not os.path.exists(p):
    print(f"{p}에 대한 모델 없음.")
    return

  ckpt=torch.load(p,map_location=device)
  model.load_state_dict(ckpt['model_state_dict'])

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

def train_model(
    models, optimizers, schedulers, criterion,
    train_loader, val_loader,
    epochs, eval_interval,
    best_losses=None, no_improves=None, es=None, es_patience=5,
    max_norm=1.0,
    resume_epoch=0, resume_model_idx=0, batch_offset=0, running_loss=0.0,
    train_loss_lists=None, val_loss_lists=None,
    ckpt_path=None, best_path=None,
    base_seed=67, device='cpu'
):

  if es is None:
    es = set()
  elif not isinstance(es, set):
    es = set(es)

  if train_loss_lists is None:
    train_loss_lists = {n: [] for n in models}
  if val_loss_lists is None:
    val_loss_lists = {n: [] for n in models}

  if best_losses is None:
    best_losses = {n: torch.inf for n in models}
  if no_improves is None:
    no_improves = {n: 0 for n in models}

  model_names = list(models.keys())
  steps_per_epoch = len(train_loader)

  for epoch in tqdm(range(resume_epoch,epochs), desc="Epoch", initial=resume_epoch,total=epochs):
    for idx,name in enumerate(model_names):

      if epoch==resume_epoch and idx<resume_model_idx:
        continue

      if name in es:
        continue

      model = models[name]
      optimizer = optimizers[name]
      scheduler = schedulers[name]

      start=batch_offset if (epoch==resume_epoch and idx==resume_model_idx) else 0

      # 모든 모델이 같은 셔플 순서를 보도록 에폭 기준으로 시드 고정
      train_loader.generator.manual_seed(base_seed + epoch)
      it=itertools.islice(train_loader,start,None) if start>0 else train_loader

      train_loss=running_loss if start>0 else 0.0
      model.train()

      for step,(train_batch,target_batch) in enumerate(tqdm(it,desc=f"Train {name}", leave=False, initial=start, total=steps_per_epoch),start=start):
        optimizer.zero_grad()
        loss=calc_loss_batch(train_batch,target_batch,model,criterion,device)
        loss.backward()

        if max_norm> 0.0:
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        train_loss+=loss.item()

        if (step + 1) % eval_interval == 0:
          save_ckpt(ckpt_path, models, optimizers, schedulers,
                    epoch=epoch,
                    model_idx=idx,
                    batch_offset=step+1,
                    running_loss=train_loss,
                    base_seed=base_seed,
                    device=device)

      scheduler.step()
      val_loss = evaluate_model(model, val_loader, criterion, device)
      train_loss_lists[name].append(train_loss / steps_per_epoch)
      val_loss_lists[name].append(val_loss)

      if val_loss<best_losses[name]:
        best_losses[name]=val_loss
        no_improves[name]=0
        save_best(best_path,name,model)

      else:
        no_improves[name]+=1
        if no_improves[name] >= es_patience:
          print(f"Early Stopping {name} at Epoch {epoch+1}")
          es.add(name)

      next_idx, next_epoch = (idx + 1, epoch) if idx + 1 < len(model_names) else (0, epoch + 1)
      save_ckpt(ckpt_path, models, optimizers, schedulers,
                next_epoch, next_idx, 0, 0.0, base_seed,
                best_losses,no_improves,list(es),
                train_loss_lists,val_loss_lists,device=device)


    resume_model_idx,batch_offset=0,0

    if len(es)==len(models):
      break
