from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset

from util.transform import resize, flip

from torchvision import transforms


class UCDDataset(Dataset):
    def __init__(self, preprocess, building_dir, tree_dir, sand_dir):
        self.ids = []
        self.building_dir = building_dir
        self.tree_dir = tree_dir
        self.sand_dir = sand_dir
        self.building_ids = []
        self.tree_ids = []
        self.sand_ids = []
        self.preprocess = transforms.Compose([
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        self.name_dict = dict()

        for f in os.listdir(building_dir):
            if str(f).endswith('.png'):
                self.building_ids.append(f)
                self.name_dict[f] = 'b'
        
        for f in os.listdir(tree_dir):
            if str(f).endswith('.png'):
                self.tree_ids.append(f)
                self.name_dict[f] = 't'
        
        for f in os.listdir(sand_dir):
            if str(f).endswith('.png'):
                self.sand_ids.append(f)
                self.name_dict[f] = 's'

        self.ids = self.building_ids + self.tree_ids + self.sand_ids

    def __getitem__(self, item):
        id_path = self.ids[item]
        name = self.name_dict[id_path]
        
        if name == 'b':
            target = 0
            real_id_path = os.path.join(self.building_dir, id_path)
        elif name == 't':
            target = 1
            real_id_path = os.path.join(self.tree_dir, id_path)
        else:
            target = 2
            real_id_path = os.path.join(self.sand_dir, id_path)

        real_img = Image.open(real_id_path)

        # real_img.save('/root/autodl-tmp/output/before.png')

        # 弱增强
        img_auged = resize(real_img, (0.8, 1.2))
        img_auged = flip(img_auged)

        # img_auged.save('/root/autodl-tmp/output/after.png')

        img_auged = self.preprocess(img_auged)

        return img_auged, target

    def __len__(self):
        return len(self.ids)

class RSSCDataset(Dataset):
    def __init__(self, preprocess, building_dir, tree_dir, sand_dir):
        self.ids = []
        self.building_dir = building_dir
        self.tree_dir = tree_dir
        self.sand_dir = sand_dir
        self.building_ids = []
        self.tree_ids = []
        self.sand_ids = []
        self.preprocess = preprocess

        self.name_dict = dict()

        for f in os.listdir(building_dir):
            if str(f).endswith('.png'):
                self.building_ids.append(f)
                self.name_dict[f] = 'b'
        
        for f in os.listdir(tree_dir):
            if str(f).endswith('.png'):
                self.tree_ids.append(f)
                self.name_dict[f] = 't'
        
        for f in os.listdir(sand_dir):
            if str(f).endswith('.png'):
                self.sand_ids.append(f)
                self.name_dict[f] = 's'

        self.ids = self.building_ids + self.tree_ids + self.sand_ids

    def __getitem__(self, item):
        id_path = self.ids[item]
        name = self.name_dict[id_path]

        if name == 'b':
            target = 0
            real_id_path = os.path.join(self.building_dir, id_path)
        elif name == 't':
            target = 1
            real_id_path = os.path.join(self.tree_dir, id_path)
        else:
            target = 2
            real_id_path = os.path.join(self.sand_dir, id_path)

        real_img = Image.open(real_id_path)

        # real_img.save('/root/autodl-tmp/output/before.png')

        # 弱增强
        img_auged = resize(real_img, (0.8, 1.2))
        img_auged = flip(img_auged)

        # img_auged.save('/root/autodl-tmp/output/after.png')

        img_auged = self.preprocess(img_auged)

        return img_auged, target

    def __len__(self):
        return len(self.ids)


class ImageFolderWithClass(Dataset):
    def __init__(self, root_dir, class_list, preprocess):
        self.samples = []  # [(image_path, class_idx)]
        self.preprocess = preprocess
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_list)}

        for cls_name in class_list:
            if " " in cls_name:
                cls_name_modified = cls_name.replace(" ", "_")
            else:
                cls_name_modified = cls_name.lower()
            cls_dir = os.path.join(root_dir, cls_name_modified)
            if not os.path.isdir(cls_dir):
                print(f"[Warning] 类别目录不存在：{cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image)
        except Exception as e:
            print(f"读取失败：{path}, {e}")
            image = torch.zeros(3, 224, 224)
        return image, label, path