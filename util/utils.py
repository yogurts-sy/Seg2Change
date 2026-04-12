import numpy as np
import math
from scipy import stats
import torch
from torchvision import utils

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
    
    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp )
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        OA = (tp+tn) / (tp + fp + tn + fn)
        return OA

    def Kappa(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        PRE = ((tp + fp) * (tp + fn) + (tn + fn) * (fp + tn)) / ((tp + fp + tn + fn) * (tp + fp + tn + fn))
        OA = (tp + tn) / (tp + fp + tn + fn)
        Kappa = (OA - PRE) / (1 - PRE)
        return Kappa

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]                          # 交集部分
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))      # bins指定统计的区间个数
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection                    # 总和部分 = output部分 + mask部分 - 交集部分
    return area_intersection, area_union, area_target

def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

def visualize_pred(pred):
    pred = torch.argmax(pred, dim=1, keepdim=True)
    pred_vis = pred * 255
    return pred_vis

def visualize_gt(gt):
    gt_vis = gt * 255
    gt_vis = gt_vis.unsqueeze(1)
    return gt_vis

def cal_kappa_from_hist(hist: np.ndarray) -> float:
    n = hist.sum()
    if n == 0:
        return 0.0
    po = np.trace(hist) / n
    pe = (hist.sum(1) @ hist.sum(0)) / (n * n + 1e-12)
    if abs(1 - pe) < 1e-12:
        return 0.0
    return (po - pe) / (1 - pe)

def sek_from_confusion_matrix(cm: np.ndarray) -> float:
    """
    cm: 2x2 confusion matrix, layout from your _generate_matrix:
        rows = gt, cols = pred
        cm = [[TN, FP],
              [FN, TP]]
    """
    assert cm.shape == (2, 2)

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    IoU_fg = TP / (TP + FP + FN)

    cm_n0 = cm.astype(np.float64).copy()
    cm_n0[0, 0] = 0.0
    kappa_n0 = cal_kappa_from_hist(cm_n0)

    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    return Sek

def oa_from_confusion_matrix(cm: np.ndarray) -> float:
    # cm = [[TN, FP],
    #       [FN, TP]]
    total = cm.sum()
    if total == 0:
        return 0.0
    return (cm[0, 0] + cm[1, 1]) / total