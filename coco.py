import os
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import glob
import numpy as np
from pycocotools.coco import COCO

from augmentations import train_aug, val_aug


def train_collate(batch):
    imgs, targets, masks = [], [], []
    valid_batch = [aa for aa in batch if aa[0] is not None]

    lack_len = len(batch) - len(valid_batch)
    if lack_len > 0:
        for i in range(lack_len):
            valid_batch.append(valid_batch[i])

    for sample in valid_batch:
        imgs.append(torch.tensor(sample[0], dtype=torch.float32))
        targets.append(torch.tensor(sample[1], dtype=torch.float32))
        masks.append(torch.tensor(sample[2], dtype=torch.float32))

    return torch.stack(imgs, 0), targets, masks


def val_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    targets = torch.tensor(batch[0][1], dtype=torch.float32)
    masks = torch.tensor(batch[0][2], dtype=torch.float32)
    return imgs, targets, masks, batch[0][3], batch[0][4]


def detect_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2]

def save_latest(net, cfg_name, step):
    weight = glob.glob('drive/My Drive/YOLACT/weights/latest*')
    weight = [aa for aa in weight if cfg_name in aa]
    assert len(weight) <= 1, 'Error, multiple latest weight found.'
    if weight:
        os.remove(weight[0])

    print(f'\nSaving the latest model as \'latest_{cfg_name}_{step}.pth\'.\n')
    torch.save(net.state_dict(), f'drive/My Drive/YOLACT/weights/latest_{cfg_name}_{step}.pth')

