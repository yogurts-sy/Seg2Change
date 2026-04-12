from copy import deepcopy
import math
import re
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # common set
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
    ])

SECOND_CLASS_MAPPING = {
    'background': 0,
    'grass': 1,
    'ground': 2,
    'tree': 3,
    'water': 4,
    'building': 5,
    'playground': 6,
}

SCSCD_CLASS_MAPPING = {
    'bareland': 0,
    'water': 1,
    'building': 2,
    'structure': 3,
    'farmland': 4,
    'vegetation': 5,
    'road': 6,
}

LsSCD_CLASS_MAPPING = {
    'bareland': 0,
    'rangeland': 1,
    'developed_space': 2,
    'road': 3,
    'tree': 4,
    'water': 5,
    'agriculture': 6,
    'building': 7,
}

class OVCDDataset(Dataset):
    def __init__(self, mode, root, dataset, crop_size, labeled_id=None, classname='', pretrain=False, A='A', B='B', label='label'):
        if pretrain:
            self.root = root
        else:
            self.root = os.path.join(root, mode)
        self.mode = mode
        self.pretrain = pretrain
        self.classname = classname
        self.dataset = dataset
        self.size = crop_size
        self.A = A
        self.B = B
        self.label = label

        self.ids = []

        if mode == 'train':
            if labeled_id == None:
                for f in os.listdir(os.path.join(self.root, self.A)):
                    if str(f).endswith('.png') or str(f).endswith('jpg') or str(f).endswith('tif'): 
                        self.ids.append(f)
            else:
                with open(labeled_id, 'r') as f:
                    self.ids = f.read().splitlines()

        else:
            if self.dataset == 'SECOND' or self.dataset == 'SCSCD' or self.dataset == 'LsSCD':
                for f in os.listdir(os.path.join(self.root, 'T1')):
                    if str(f).endswith('.png') or str(f).endswith('jpg'):
                        self.ids.append(f)
            else:
                for f in os.listdir(os.path.join(self.root, 'A')):
                    if str(f).endswith('.png') or str(f).endswith('jpg'):
                        self.ids.append(f)

            if self.dataset == 'LEVIR-CD':
                self.ids = sorted(self.ids, key=lambda x: int(re.search(r'_(\d+)', x).group(1)))
            elif self.dataset == 'DSIFN-CD':
                self.ids = sorted(self.ids, key=lambda x: int(x.rstrip('.jpg')))
            elif self.dataset == 'BANDON':
                self.ids = sorted(self.ids, key=lambda x: int(re.search(r'_(\d+)', x).group(1)))


    def __getitem__(self, item):
        id = self.ids[item]

        if self.mode == 'train':

            if self.dataset == 'SECOND':
                imgA_path = os.path.join(self.root, 'T1', id)
                imgB_path = os.path.join(self.root, 'T2', id)
                mask_path = os.path.join(self.root, 'GT_CD', id)

            else:
                imgA_path = os.path.join(self.root, self.A, id)
                imgB_path = os.path.join(self.root, self.B, id)
                mask_path = os.path.join(self.root, self.label, id)

            imgA = Image.open(imgA_path).convert('RGB')
            imgB = Image.open(imgB_path).convert('RGB')
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)

            if self.dataset == 'CNAM':
                mask[mask > 0] = 255

            mask = mask / 255

            mask = Image.fromarray(mask.astype(np.uint8))

            ow = self.size
            oh = self.size

            imgA = imgA.resize((ow, oh), Image.BILINEAR)
            imgB = imgB.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            A = TRANSFORMS(Image.open(imgA_path).resize((ow, oh), Image.BILINEAR))
            B = TRANSFORMS(Image.open(imgB_path).resize((ow, oh), Image.BILINEAR))

            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, A, B

        else:
            ow = self.size
            oh = self.size

            if self.dataset == 'SECOND':
                imgA_path = os.path.join(self.root, 'T1', id)
                imgB_path = os.path.join(self.root, 'T2', id)
                mask_path = os.path.join(self.root, 'GT_T1', id)
                mask2_path = os.path.join(self.root, 'GT_T2', id)

                imgA = Image.open(imgA_path).convert('RGB')
                imgB = Image.open(imgB_path).convert('RGB')

                imgA = imgA.resize((ow, oh), Image.BILINEAR)
                imgB = imgB.resize((ow, oh), Image.BILINEAR)

                mask1 = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                mask2 = np.array(Image.open(mask2_path).convert('L'), dtype=np.uint8)

                class_id = SECOND_CLASS_MAPPING[self.classname]
                mask = ((mask1 == class_id) | (mask2 == class_id)).astype(np.uint8)

            elif self.dataset == 'SCSCD':
                imgA_path = os.path.join(self.root, 'T1', id)
                imgB_path = os.path.join(self.root, 'T2', id)
                mask_path = os.path.join(self.root, 'GT_T1', id)
                mask2_path = os.path.join(self.root, 'GT_T2', id)

                gt_mask_path = os.path.join(self.root, 'GT_CD', id)

                imgA = Image.open(imgA_path).convert('RGB')
                imgB = Image.open(imgB_path).convert('RGB')

                imgA = imgA.resize((ow, oh), Image.BILINEAR)
                imgB = imgB.resize((ow, oh), Image.BILINEAR)

                mask1 = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                mask2 = np.array(Image.open(mask2_path).convert('L'), dtype=np.uint8)

                gt_mask = np.array(Image.open(gt_mask_path).convert('L'), dtype=np.uint8)

                gt_mask = gt_mask / 255
                gt_mask = gt_mask.astype(np.uint8)

                class_id = SCSCD_CLASS_MAPPING[self.classname]
                mask = ((mask1 == class_id) | (mask2 == class_id)).astype(np.uint8)

                mask[gt_mask != 1] = 0

            elif self.dataset == 'LsSCD':
                imgA_path = os.path.join(self.root, 'T1', id)
                imgB_path = os.path.join(self.root, 'T2', id)
                mask_path = os.path.join(self.root, 'GT_T1', id)
                mask2_path = os.path.join(self.root, 'GT_T2', id)

                gt_mask_path = os.path.join(self.root, 'GT_CD', id)

                imgA = Image.open(imgA_path).convert('RGB')
                imgB = Image.open(imgB_path).convert('RGB')

                imgA = imgA.resize((ow, oh), Image.BILINEAR)
                imgB = imgB.resize((ow, oh), Image.BILINEAR)

                mask1 = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                mask2 = np.array(Image.open(mask2_path).convert('L'), dtype=np.uint8)

                gt_mask = np.array(Image.open(gt_mask_path).convert('L'), dtype=np.uint8)

                gt_mask = gt_mask / 255
                gt_mask = gt_mask.astype(np.uint8)

                class_id = LsSCD_CLASS_MAPPING[self.classname]
                mask = ((mask1 == class_id) | (mask2 == class_id)).astype(np.uint8)

                mask[gt_mask != 1] = 0

            else:
                imgA_path = os.path.join(self.root, self.A, id)
                imgB_path = os.path.join(self.root, self.B, id)
                mask_path = os.path.join(self.root, self.label, id.replace(".jpg", '.tif') if self.dataset == 'DSIFN-CD' else id)

                imgA = Image.open(imgA_path).convert('RGB')
                imgB = Image.open(imgB_path).convert('RGB')

                imgA = imgA.resize((ow, oh), Image.BILINEAR)
                imgB = imgB.resize((ow, oh), Image.BILINEAR)
            
                mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                if self.dataset in ['WHU-CD', 'LEVIR-CD', 'SECOND', 'CLCD', 'BANDON', 'xView2', 'TUE-CD']:
                    mask = mask / 255
                
            mask = Image.fromarray(mask.astype(np.uint8))

            if self.mode == 'test' or self.mode == 'test_ood':
                return TRANSFORMS(imgA), TRANSFORMS(imgB), TRANSFORMS(mask), imgA_path, imgB_path, mask_path

            if self.mode == 'train':
                return TRANSFORMS(imgA), TRANSFORMS(imgB), TRANSFORMS(mask), imgA_path, imgB_path, mask_path

    def __len__(self):
        return len(self.ids)