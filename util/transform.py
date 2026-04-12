import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def resize(img, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    return img

def crop(img, size):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))

    return img

def flip(img):
    flip_type = random.choice(['none', 'horizontal', 'vertical'])

    if flip_type == 'horizontal':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_type == 'vertical':
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return img