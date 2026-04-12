import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from tqdm import tqdm

from model.ovcd.change_head_fdr_dino import CHead_FPN_FDR as CHead
from seg_model_sam3 import SegEarthOV3Segmentation
from segearthov1_segmentor import SegEarthSegmentation
import torch
import pprint
import numpy as np
import cv2
import logging
import argparse
import torch.nn as nn
from util.utils import Evaluator, intersectionAndUnion, sek_from_confusion_matrix, oa_from_confusion_matrix
from PIL import Image
from torchvision import transforms
from dataset.ovcd import OVCDDataset
from torch.utils.data import DataLoader
from mmseg.structures import SegDataSample

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from model.backbone.dinov2 import DINOv2

import torch.nn.functional as F

GT_COLORMAP = np.array([[[0, 0, 0], [0, 0, 185]], [[0, 178, 0], [255, 255, 255]]])

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
    ])

VOC_COLORMAP = [[0, 0, 0],
                [255, 255, 255],
                [128, 128, 0],
                [255, 195, 128],
                [34, 97, 38],
                [128, 0, 0],
                [0, 255, 36],
                [0, 69, 255],
                [178, 102, 178],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

intermediate_layer_idx = {
    'small': [2, 5, 8, 11],
    'base': [2, 5, 8, 11], 
    'large': [4, 11, 17, 23], 
    'giant': [9, 19, 29, 39]
}

def show_mask(mask, ax, random_color=False, default_color =  [30/255, 144/255, 255/255]):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.concatenate([default_color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color


def show_results(save_path, image, seg_pred, seg_logit, name_list, add_bg=False, vis_thresh = 0.01):
    if type(image) == str:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cls_pred = F.one_hot(seg_pred.squeeze(0).long(), num_classes=len(name_list) + int(add_bg)).permute(2, 0, 1).float()  # [C, H, W]
    patches = []  # To store legend patches
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    for i in range(len(name_list)):
        bool_mask = cls_pred[i].bool() & (seg_logit[i] > vis_thresh)
        cls_color = [c / 255 for c in VOC_COLORMAP[1:][i]] # Skip the first color (black)
        color = show_mask(bool_mask.cpu().numpy(), plt.gca(), default_color=cls_color)
        patch = Patch(color=color, label=str(i) + ' ' + name_list[i])
        patches.append(patch)
    plt.axis('off')
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_img_feature(model, img_tensor, img_path):
    torch.cuda.empty_cache()

    data_sample = SegDataSample()
    data_sample.set_metainfo({'img_path': img_path, 'ori_shape': img_tensor.shape[1:]})

    sam_enc_feats = model.predict(img_tensor, data_samples=[data_sample], get_feats=True)

    return sam_enc_feats

def get_seg_model_predict(args, model, img_tensor, img_path, target_classes):
    torch.cuda.empty_cache()

    seg_pred, pred_mask, seg_logits, sam_enc_feats = None, None, None, None

    if args.ovss_model == 'SegEarth-OV3' or args.ovss_model == 'SAM3':

        data_sample = SegDataSample()
        data_sample.set_metainfo({'img_path': img_path, 'ori_shape': img_tensor.shape[1:]})

        seg_logits, seg_pred, sam_enc_feats, _ = model.predict(img_tensor, data_samples=[data_sample])

        pred_mask = torch.zeros_like(seg_pred, dtype=torch.bool).cuda()

        for target_class in target_classes:
            pred_mask[seg_pred == target_class] = True

    elif args.ovss_model == 'SegEarth-OV1':
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = F.interpolate(img_tensor, (448, 448), mode='bilinear', align_corners=False)
        seg_pred = model.predict(img_tensor, data_samples=None)
        seg_pred = seg_pred.to(dtype=torch.float32)
        seg_pred = F.interpolate(seg_pred.unsqueeze(0), (512, 512), mode='bilinear', align_corners=False)
        seg_pred = seg_pred.squeeze(0).squeeze(0)
        seg_pred = seg_pred.to(dtype=torch.float32)

        pred_mask = torch.zeros_like(seg_pred, dtype=torch.bool).cuda()

        for target_class in target_classes:
            pred_mask[seg_pred == target_class] = True

    return seg_pred, pred_mask.unsqueeze(0), seg_logits, sam_enc_feats


def get_img_feature_dinov2(backbone, encoder_size, img_t):
    sam_enc_feats = backbone.get_intermediate_layers(img_t, intermediate_layer_idx[encoder_size])

    return sam_enc_feats


def build_dino_set_finetune(dino, dino_ft):
    for name, params in dino.named_parameters():
        if dino_ft == "attention":
            if "attn.qkv.weight" in name:
                params.requires_grad = True
            elif "pos_embed" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif dino_ft == "full":
            params.requires_grad = True
        else:
            params.requires_grad = False

    return dino


def tensor_to_heatmap_overlay(
    image_tensorA,
    image_tensorB,
    score_tensor,
    save_heatmap_path=None,
    save_overlayA_path=None,
    save_overlayB_path=None,
    alpha=0.5,
    colormap=cv2.COLORMAP_TURBO# cv2.COLORMAP_JET
):
    """
    image_tensor: [1,3,H,W] or [3,H,W], normalized tensor
    score_tensor: [1,H,W] or [H,W] or [1,1,H,W]
    """

    # ---- image tensor -> RGB uint8 ----
    if image_tensorA.dim() == 4:
        image_tensorA = image_tensorA[0]
    imageA = image_tensorA.detach().cpu().float().permute(1, 2, 0).numpy()

    imageA = np.clip(imageA, 0, 1)
    imageA = (imageA * 255).astype(np.uint8)

    if image_tensorB.dim() == 4:
        image_tensorB = image_tensorB[0]
    imageB = image_tensorB.detach().cpu().float().permute(1, 2, 0).numpy()

    imageB = np.clip(imageB, 0, 1)
    imageB = (imageB * 255).astype(np.uint8)

    if score_tensor.dim() == 4:
        score_tensor = score_tensor[0, 0]
    elif score_tensor.dim() == 3:
        score_tensor = score_tensor[0]

    score = score_tensor.detach().cpu().float().numpy()
    score = score - score.min()
    score = score / (score.max() + 1e-8)

    score = cv2.resize(score, (imageA.shape[1], imageA.shape[0]))
    heatmap = np.uint8(score * 255)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlayA = np.uint8((1 - alpha) * imageA + alpha * heatmap)
    overlayB = np.uint8((1 - alpha) * imageB + alpha * heatmap)

    if save_heatmap_path is not None:
        cv2.imwrite(save_heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    if save_overlayA_path is not None:
        cv2.imwrite(save_overlayA_path, cv2.cvtColor(overlayA, cv2.COLOR_RGB2BGR))
    if save_overlayB_path is not None:
        cv2.imwrite(save_overlayB_path, cv2.cvtColor(overlayB, cv2.COLOR_RGB2BGR))

    return heatmap, overlayA, overlayB


def build_guided_weight_map(pred_mask_A, pred_mask_B, true_scale=1.8, false_scale=0.4, mode='union'):

    if mode == 'union':
        guide_mask = pred_mask_A | pred_mask_B
    elif mode == 'intersect':
        guide_mask = pred_mask_A & pred_mask_B
    elif mode == 'xor':
        guide_mask = pred_mask_A ^ pred_mask_B
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    weight_map = torch.where(
        guide_mask,
        torch.full_like(pred_mask_A.float(), true_scale),
        torch.full_like(pred_mask_A.float(), false_scale)
    )
    return weight_map

def save_guided_output_heatmap(
    img_tensor,
    output_tensor,
    pred_mask_A,
    pred_mask_B,
    save_heatmap_path,
    save_overlay_path=None,
    mode='union',
    true_scale=1.8,
    false_scale=0.4,
    alpha=0.5,
    blur_ksize=0,
    colormap=cv2.COLORMAP_TURBO
):

    if output_tensor.shape[1] > 1:
        score = torch.softmax(output_tensor, dim=1)[:, 1:2] 
    else:
        score = torch.sigmoid(output_tensor)

    weight_map = build_guided_weight_map(
        pred_mask_A, pred_mask_B,
        true_scale=true_scale,
        false_scale=false_scale,
        mode=mode
    )

    guided_score = score * weight_map

    img = img_tensor[0].detach().cpu().float().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    guided_map = guided_score[0, 0].detach().cpu().float().numpy()

    guided_map = guided_map - guided_map.min()
    guided_map = guided_map / (guided_map.max() + 1e-8)

    guided_map = cv2.resize(guided_map, (img.shape[1], img.shape[0]))

    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        guided_map = cv2.GaussianBlur(guided_map, (blur_ksize, blur_ksize), 0)

    heatmap = np.uint8(guided_map * 255)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8((1 - alpha) * img + alpha * heatmap)

    cv2.imwrite(save_heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    if save_overlay_path is not None:
        cv2.imwrite(save_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return guided_score, weight_map


def save_binary_mask(mask_tensor, save_path):

    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor[0, 0]
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor[0]

    mask = mask_tensor.detach().cpu().numpy()
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(save_path, mask)


def evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader, dataset_name, SAVE_FIG=True, classname='building'):
    evaluator_test = Evaluator(num_class=2)

    change_head.eval()
    os.makedirs(os.path.join(args.feat_path, dataset_name, 'test'), exist_ok=True)

    AB_diff_sup = False

    for imgA, imgB, mask, imgA_path, imgB_path, _ in tqdm(valloader, desc="Processing", unit="iteration"):
        imgA, imgB, mask = imgA.cuda(), imgB.cuda(), mask.cuda()

        base_path = os.path.basename(imgA_path[0])

        # if '0000' in base_path or '0001' in base_path:
        #     continue

        feat_path = os.path.join(args.feat_path, dataset_name, 'test', base_path + '.pt')
        if os.path.exists(feat_path):
            feats = torch.load(feat_path, weights_only=False)
            pred_mask_A, pred_mask_B = feats['pred_mask_A'], feats['pred_mask_B'] if 'pred_mask_B' in feats else None

        else:

            _, pred_mask_A, _, _ = get_seg_model_predict(args, seg_model, imgA[0], imgA_path[0], target_classes)
            _, pred_mask_B, _, _ = get_seg_model_predict(args, seg_model, imgB[0], imgB_path[0], target_classes)
            feats = {'pred_mask_A': pred_mask_A, 'pred_mask_B': pred_mask_B}
            torch.save(feats, feat_path)

        sam_enc_feats_A = get_img_feature_dinov2(backbone_dino, args.encoder_size, imgA)
        sam_enc_feats_B = get_img_feature_dinov2(backbone_dino, args.encoder_size, imgB)

        output, _, _, _, _, _ = change_head(sam_enc_feats_A, sam_enc_feats_B)
        output = F.interpolate(output, size=(512, 512), mode="bilinear", align_corners=True)

        pred_mask_A_mask = torch.where(pred_mask_A.squeeze(0).squeeze(0), 255, 0).cpu().numpy().astype(np.uint8)
        pred_mask_B_mask = torch.where(pred_mask_B.squeeze(0).squeeze(0), 255, 0).cpu().numpy().astype(np.uint8)

        pred = torch.argmax(output, dim=1, keepdim=True)
        pred_mask = np.where(pred.squeeze(0).squeeze(0).cpu().numpy() > 0, 255, 0).astype(np.uint8)

        if SAVE_FIG:
            cv2.imwrite(os.path.join(args.save_path, 'A', 'pred_mask_A_' + base_path), pred_mask_A_mask)
            cv2.imwrite(os.path.join(args.save_path, 'B', 'pred_mask_B_' + base_path), pred_mask_B_mask)
            cv2.imwrite(os.path.join(args.save_path, 'CM', 'pred_mask_' + base_path), pred_mask)

        if AB_diff_sup:
            pred_mask_diff = pred_mask_A ^ pred_mask_B
            pred[pred_mask_diff] = 1

            pred_mask = np.where(pred.squeeze(0).squeeze(0).cpu().numpy() > 0, 255, 0).astype(np.uint8)
            if SAVE_FIG:
                cv2.imwrite(os.path.join(args.save_path, 'CM', 'sup_pred_mask_' + base_path), pred_mask)

        pred_A = pred * pred_mask_A
        pred_B = pred * pred_mask_B
        pred = ((pred_A + pred_B) > 0).to(pred.dtype)
        pred = pred.squeeze(0).squeeze(0)
        df_int8 = pred.cpu().numpy()

        mask = mask * 255
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()

        mask = mask[np.newaxis, :]
        df_int8 = df_int8[np.newaxis, :]

        evaluator_test.add_batch(mask.astype('int'), df_int8.astype('int'))

        intersection, union, target = \
                intersectionAndUnion(mask.astype('int'), df_int8.astype('int'), 2, 255)

        iou = intersection / (union + 1e-10)

        if SAVE_FIG:

            pred_rgb_out = np.array([GT_COLORMAP[l][p] for l, p in zip(mask.astype('int').reshape(-1), df_int8.astype('int').reshape(-1))])
            pred_rgb_out = pred_rgb_out.reshape((mask.shape[1], mask.shape[2], 3)).astype(np.uint8)
            cv2.imwrite(os.path.join(args.save_path, 'CM', f'{dataset_name}_color_mask_' + base_path), pred_rgb_out)

            if df_int8.ndim == 3 and ('SCSCD' in dataset_name or 'SECOND' in dataset_name):
                df_int8 = df_int8[0]
                h, w = df_int8.shape
                vis_color = np.zeros((h, w, 3), dtype=np.uint8)
                if 'SECOND' in dataset_name:
                    if 'Building' in dataset_name:
                        color = np.array([255, 171, 124], dtype=np.uint8)
                    elif 'Tree' in dataset_name:
                        color = np.array([84, 129, 94], dtype=np.uint8)
                    elif 'Water' in dataset_name:
                        color = np.array([135, 206, 235], dtype=np.uint8)
                    elif 'Grass' in dataset_name:
                        color = np.array([199, 229, 189], dtype=np.uint8)
                    elif 'Ground' in dataset_name:
                        color = np.array([215, 220, 225], dtype=np.uint8)
                    elif 'Playground' in dataset_name:
                        color = np.array([255, 181, 192], dtype=np.uint8)
                elif 'SCSCD' in dataset_name:
                    if 'Building' in dataset_name:
                        color = np.array([255, 171, 124], dtype=np.uint8)
                    elif 'Bareland' in dataset_name:
                        color = np.array([215, 220, 225], dtype=np.uint8)
                    elif 'Water' in dataset_name:
                        color = np.array([135, 206, 235], dtype=np.uint8)
                    elif 'Structure' in dataset_name:
                        color = np.array([255, 255, 0], dtype=np.uint8)
                    elif 'Farmland' in dataset_name:
                        color = np.array([0, 255, 0], dtype=np.uint8)
                    elif 'Vegetation' in dataset_name:
                        color = np.array([199, 229, 189], dtype=np.uint8)
                    elif 'Road' in dataset_name:
                        color = np.array([128, 128, 128], dtype=np.uint8)
                
                vis_color[df_int8 == 1] = color
                Image.fromarray(vis_color).save(os.path.join(args.save_path, 'CM', f'{dataset_name}_color_class_mask_' + base_path))

        with open(os.path.join(args.save_path, 'CM', f'{dataset_name}_IoU.txt'), 'a') as txt_file:
            txt_file.write('{:} IoU: {:1f}'.format(base_path, iou[1]) + '\n')

    IoU = evaluator_test.Intersection_over_Union()[1]
    ACC = evaluator_test.OA()[1]
    Pre = evaluator_test.Precision()[1]
    Recall = evaluator_test.Recall()[1]
    F1 = evaluator_test.F1()[1]
    Kappa = evaluator_test.Kappa()[1]

    print('***** Evaluation %s ***** >>>> IoU: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, OA: %.2f, Kappa: %.4f' % (dataset_name, IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa))

    with open(os.path.join(args.save_path, 'test.txt'), 'a') as txt_file:
        txt_file.write('***** Evaluation {:} ***** >>>> IoU: {:.2f}, F1: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, OA: {:.2f}, Kappa: {:.4f}'.format(dataset_name, IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa) + '\n')

    return IoU, F1, ACC, Kappa


def slide_infer(change_head, pred_mask_A, pred_mask_B, sam_enc_feats_A, sam_enc_feats_B, mode='cd', crop_size=512, stride=512):
    _, _, h, w = sam_enc_feats_A.shape
    final_output = torch.zeros((1, 2, h, w)).cuda()
    count_mat = torch.zeros((1, 1, h, w)).cuda()

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            y2 = min(y + crop_size, h)
            x1 = x
            x2 = min(x + crop_size, w)
            y1 = max(0, y2 - crop_size)
            x1 = max(0, x2 - crop_size)

            pred_mask_A_crop = pred_mask_A[:, :, y1:y2, x1:x2]
            pred_mask_B_crop = pred_mask_B[:, :, y1:y2, x1:x2]
            sam_enc_feats_A_crop = sam_enc_feats_A[:, :, y1:y2, x1:x2]
            sam_enc_feats_B_crop = sam_enc_feats_B[:, :, y1:y2, x1:x2]

            output_crop, _, _ = change_head(pred_mask_A_crop, pred_mask_B_crop, sam_enc_feats_A_crop, sam_enc_feats_B_crop, mode=mode)

            final_output[:, :, y1:y2, x1:x2] += output_crop
            count_mat[:, :, y1:y2, x1:x2] += 1

    final_output = final_output / count_mat

    return final_output

def main():
    parser = argparse.ArgumentParser(description='Seg2Change')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--feat_path', type=str, default='/data2/suyou/Codes/Seg2Change/features/features_ck03', help='test feature cache path')
    parser.add_argument('--checkpoint_path', type=str, default='/data2/suyou/Codes/Seg2Change/weights/cach/best.pth', help='load change head checkpoint path')
    parser.add_argument('--save_path', type=str, default='/data2/suyou/Codes/Seg2Change/exp/train_cach_dino_base_CACD/test01/Infer_SCSCD-ALL_ck03', help='path to save evaluation results')
    parser.add_argument('--encoder_size', type=str, default='base', help='DINOv2 encoder size: small, base, large')
    parser.add_argument('--dino_ft', type=str, default='frozen', help='DINOv2 fine-tuning strategy: full, attention')
    parser.add_argument('--crop_size', type=int, default=504, help='training dataset crop size')
    parser.add_argument('--ovss_model', type=str, default='SegEarth-OV3', help='OVSS Model')
    parser.add_argument('--test_dataset', type=str, default='SCSCD-ALL', help='evaluation dataset: WHU-CD, LEVIR-CD, DSIFN-CD, CLCD, SECOND-ALL, SCSCD-ALL')
    parser.add_argument('--dataset_root_path', type=str, default='/data2/suyou/Codes/datasets/OVCD_Benchmark/', help='evaluation dataset root path')
    parser.add_argument('--nclass', type=int, default=2)

    args = parser.parse_args()

    # WHU-CD, LEVIR-CD
    if args.test_dataset == 'WHU-CD' or args.test_dataset == 'LEVIR-CD':

        # name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
        #         'tree,forest', 'water,river', 'cropland', 'building,roof,house']

        name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
             'tree,forest', 'water,river', 'cropland', 'building']

        target_classes = [8]

    # DSIFN-CD
    elif args.test_dataset == 'DSIFN-CD':

        name_list = ['background', 'building,roof,house,garden,playground,construction,apartment,residential,materials', 'tree,forest', 'water,river']
        target_classes = [1]

    # CLCD
    elif args.test_dataset == 'CLCD':

        name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
                 'tree,forest', 'water,river', 'cropland', 'building,roof,house']
        target_classes = [1, 2, 3, 5, 6, 7, 8]

    # BANDON\BANDON-ood
    elif args.test_dataset == 'BANDON' or args.test_dataset == 'BANDON-OOD':

        name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
                 'tree,forest', 'water,river', 'cropland', 'building,roof,house']
        target_classes = [8]

    # SECOND
    elif 'SECOND' in args.test_dataset:

        name_list = ['background', 'bareland,barren,ground', 'grass', 'sports field', 'car',
                 'tree,forest', 'water,river', 'cropland', 'building,roof,house']

        if args.test_dataset == 'SECOND-Building':
            target_classes = [8]
        elif args.test_dataset == 'SECOND-Tree':
            target_classes = [5]
        elif args.test_dataset == 'SECOND-Water':
            target_classes = [6]
        elif args.test_dataset == 'SECOND-Grass':
            target_classes = [2, 7]
        elif args.test_dataset == 'SECOND-Ground':
            target_classes = [1]
        elif args.test_dataset == 'SECOND-Playground':
            target_classes = [3]

    elif 'SCSCD' in args.test_dataset:

        name_list = ['bareland,barren,ground,floor,soil', 'water,river,pond', 'building,roof,house', 
                     'structure,construction,greenhouse', 'farmland,cropland,terrace', 
                     'grass,vegetation,tree,forest', 'road']

        if args.test_dataset == 'SCSCD-Bareland':
            target_classes = [0]
        elif args.test_dataset == 'SCSCD-Water':
            target_classes = [1]
        if args.test_dataset == 'SCSCD-Building':
            target_classes = [2]
        elif args.test_dataset == 'SCSCD-Structure':
            target_classes = [3]
        elif args.test_dataset == 'SCSCD-Farmland':
            target_classes = [4]
        elif args.test_dataset == 'SCSCD-Vegetation':
            target_classes = [5]
        elif args.test_dataset == 'SCSCD-Road':
            target_classes = [6]

    if args.encoder_size == 'small':
        state_dict = torch.load('./weights/dinov2/dinov2_vits14_pretrain.pth', weights_only=False)
        dino_dim = 384
    elif args.encoder_size == 'base':
        state_dict = torch.load('./weights/dinov2/dinov2_vitb14_pretrain.pth', weights_only=False)
        dino_dim = 768
    
    backbone_dino = DINOv2(model_name=args.encoder_size).cuda()
    backbone_dino.load_state_dict(state_dict)
    backbone_dino = build_dino_set_finetune(backbone_dino, args.dino_ft)
    
    with open('./configs/my_name.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list)-1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
    writers.close()

    args.save_path = args.save_path + "/" + args.test_dataset

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'CM'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'A'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'B'), exist_ok=True)

    if args.ovss_model == 'SegEarth-OV3':

        seg_model = SegEarthOV3Segmentation(type='SegEarthOV3Segmentation', model_type='SAM3', classname_path='./configs/my_name.txt', 
            prob_thd=0.1, confidence_threshold=0.1, slide_stride=512, slide_crop=512, use_vfm=False, version='segearth-ov3')

    elif args.ovss_model == 'SegEarth-OV1':

        seg_model = SegEarthSegmentation(
            clip_type='CLIP', 
            vit_type='ViT-B/16',
            model_type='SegEarth',
            ignore_residual=True,
            feature_up=True,
            feature_up_cfg=dict(
                model_name='jbu_one',
                model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
            cls_token_lambda=-0.3,
            name_path='./configs/my_name.txt',
            prob_thd=0.1,
        ).cuda()

    elif args.ovss_model == 'SAM3':

        seg_model = SegEarthOV3Segmentation(type='SegEarthOV3Segmentation', model_type='SAM3', classname_path='./configs/my_name.txt', 
            prob_thd=0.1, confidence_threshold=0.1, slide_stride=512, slide_crop=512, use_vfm=False, version='sam3')

    change_head = CHead(in_channels=dino_dim, hidden_dim=64, nclass=args.nclass).cuda()

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        change_head.load_state_dict(checkpoint)

    if args.test_dataset == 'WHU-CD':

        valset_whu = OVCDDataset('test', args.dataset_root_path + 'WHU-CD-512/', "WHU-CD", args.crop_size)
        valloader_whu = DataLoader(valset_whu, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_whu, 'WHU-CD-512')

    elif args.test_dataset == 'LEVIR-CD':

        valset_levir = OVCDDataset('test', args.dataset_root_path + 'LEVIR-CD-512/', "LEVIR-CD", args.crop_size)
        valloader_levir = DataLoader(valset_levir, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_levir, 'LEVIR-CD-512')

    elif args.test_dataset == 'DSIFN-CD':

        valset_dsifn = OVCDDataset('test', args.dataset_root_path + 'DSIFN-512/', "DSIFN-CD", args.crop_size)    
        valloader_dsifn = DataLoader(valset_dsifn, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_dsifn, 'DSIFN-CD-512')
    
    elif args.test_dataset == 'CLCD':

        valset_clcd = OVCDDataset('test', args.dataset_root_path + 'CLCD-512/', "CLCD", args.crop_size)
        valloader_clcd = DataLoader(valset_clcd, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_clcd, 'CLCD-512')
        
    elif args.test_dataset == 'BANDON':

        valset_bandon = OVCDDataset('test', args.dataset_root_path + 'BANDON-512/', "BANDON", args.crop_size)
        valset_bandon_ood = OVCDDataset('test_ood', args.dataset_root_path + 'BANDON-512-OOD/', "BANDON", args.crop_size)
        valloader_bandon = DataLoader(valset_bandon, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_bandon_ood = DataLoader(valset_bandon_ood, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_bandon, 'BANDON-512')
        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_bandon_ood, 'BANDON-512-OOD')
        
    elif args.test_dataset == 'SECOND-Building':

        valset_second_building = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='building')
        valloader_second_building = DataLoader(valset_second_building, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_building, 'SECOND-Building')

    elif args.test_dataset == 'SECOND-Tree':

        valset_second_tree = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='tree')
        valloader_second_tree = DataLoader(valset_second_tree, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_tree, 'SECOND-Tree')

    elif args.test_dataset == 'SECOND-Water':

        valset_second_water = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='water')
        valloader_second_water = DataLoader(valset_second_water, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_water, 'SECOND-Water')

    elif args.test_dataset == 'SECOND-Grass':

        valset_second_grass = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='grass')
        valloader_second_grass = DataLoader(valset_second_grass, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_grass, 'SECOND-Grass')

    elif args.test_dataset == 'SECOND-Ground':

        valset_second_ground = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='ground')
        valloader_second_ground = DataLoader(valset_second_ground, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_ground, 'SECOND-Ground')

    elif args.test_dataset == 'SECOND-Playground':

        valset_second_playground = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='playground')
        valloader_second_playground = DataLoader(valset_second_playground, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
    
        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_playground, 'SECOND-Playground')

    elif args.test_dataset == 'SECOND-ALL':

        valset_second_building = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='building')
        valset_second_tree = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='tree')
        valset_second_water = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='water')
        valset_second_grass = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='grass')
        valset_second_ground = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='ground')
        valset_second_playground = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='playground')

        valloader_second_building = DataLoader(valset_second_building, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_tree = DataLoader(valset_second_tree, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_water = DataLoader(valset_second_water, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_grass = DataLoader(valset_second_grass, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_ground = DataLoader(valset_second_ground, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_playground = DataLoader(valset_second_playground, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        iou_building, f1_building, oa_building, kappa_building = evaluate(args, seg_model, backbone_dino, [8], change_head, valloader_second_building, 'SECOND-Building', SAVE_FIG=True)
        iou_tree, f1_tree, oa_tree, kappa_tree = evaluate(args, seg_model, backbone_dino, [5], change_head, valloader_second_tree, 'SECOND-Tree', SAVE_FIG=True)
        iou_water, f1_water, oa_water, kappa_water = evaluate(args, seg_model, backbone_dino, [6], change_head, valloader_second_water, 'SECOND-Water', SAVE_FIG=True)
        iou_grass, f1_grass, oa_grass, kappa_grass = evaluate(args, seg_model, backbone_dino, [2, 7], change_head, valloader_second_grass, 'SECOND-Grass', SAVE_FIG=True)
        iou_ground, f1_ground, oa_ground, kappa_ground = evaluate(args, seg_model, backbone_dino, [1], change_head, valloader_second_ground, 'SECOND-Ground', SAVE_FIG=True)
        iou_playground, f1_playground, oa_playground, kappa_playground = evaluate(args, seg_model, backbone_dino, [3], change_head, valloader_second_playground, 'SECOND-Playground', SAVE_FIG=True)

        iou_avg = (iou_building + iou_tree + iou_water + iou_grass + iou_ground + iou_playground) / 6.0
        f1_avg = (f1_building + f1_tree + f1_water + f1_grass + f1_ground + f1_playground) / 6.0
        oa_avg = (oa_building + oa_tree + oa_water + oa_grass + oa_ground + oa_playground) / 6.0
        kappa_avg = (kappa_building + kappa_tree + kappa_water + kappa_grass + kappa_ground + kappa_playground) / 6.0

        print('***** Evaluation {:} ***** >>>> IoU_avg: {:.2f}, F1_avg: {:.2f}, OA_avg: {:.2f}, Kappa_avg: {:.2f}'.format(args.test_dataset, iou_avg * 100, f1_avg * 100, oa_avg * 100, kappa_avg * 100))

        with open(os.path.join(args.save_path, 'test.txt'), 'a') as txt_file:
            txt_file.write('***** Evaluation {:} ***** >>>> IoU_avg: {:.2f}, F1_avg: {:.2f}, OA_avg: {:.2f}, Kappa_avg: {:.2f}'.format(args.test_dataset, iou_avg * 100, f1_avg * 100, oa_avg * 100, kappa_avg * 100) + '\n')

    elif args.test_dataset == 'SCSCD-Bareland':

        valset_scscd_bareland = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='bareland')
        valloader_scscd_bareland= DataLoader(valset_scscd_bareland, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_bareland, 'SCSCD-Bareland')

    elif args.test_dataset == 'SCSCD-Water':

        valset_scscd_water = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='water')
        valloader_scscd_water = DataLoader(valset_scscd_water, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_water, 'SCSCD-Water')

    elif args.test_dataset == 'SCSCD-Building':

        valset_scscd_building = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='building')
        valloader_scscd_building = DataLoader(valset_scscd_building, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_building, 'SCSCD-Building')

    elif args.test_dataset == 'SCSCD-Structure':

        valset_scscd_structure = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='structure')
        valloader_scscd_structure = DataLoader(valset_scscd_structure, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_structure, 'SCSCD-Structure')

    elif args.test_dataset == 'SCSCD-Farmland':

        valset_scscd_farmland = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='farmland')
        valloader_scscd_farmland = DataLoader(valset_scscd_farmland, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_farmland, 'SCSCD-Farmland')

    elif args.test_dataset == 'SCSCD-Vegetation':

        valset_scscd_vegetation = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='vegetation')
        valloader_scscd_vegetation = DataLoader(valset_scscd_vegetation, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_vegetation, 'SCSCD-Vegetation')

    elif args.test_dataset == 'SCSCD-Road':

        valset_scscd_road = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='road')
        valloader_scscd_road = DataLoader(valset_scscd_road, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_scscd_road, 'SCSCD-Road')

    elif args.test_dataset == 'SCSCD-ALL':

        valset_scscd_bareland = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='bareland')
        valset_scscd_water = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='water')
        valset_scscd_building = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='building')
        valset_scscd_structure = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='structure')
        valset_scscd_farmland = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='farmland')
        valset_scscd_vegetation = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='vegetation')
        valset_scscd_road = OVCDDataset('test', args.dataset_root_path + 'SCSCD/', "SCSCD", args.crop_size, classname='road')

        valloader_scscd_bareland= DataLoader(valset_scscd_bareland, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_scscd_water = DataLoader(valset_scscd_water, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_scscd_building = DataLoader(valset_scscd_building, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_scscd_structure = DataLoader(valset_scscd_structure, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_scscd_farmland = DataLoader(valset_scscd_farmland, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_scscd_vegetation = DataLoader(valset_scscd_vegetation, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_scscd_road = DataLoader(valset_scscd_road, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

        iou_bareland, f1_bareland, oa_bareland, kappa_bareland = evaluate(args, seg_model, backbone_dino, [0], change_head, valloader_scscd_bareland, 'SCSCD-Bareland')
        iou_water, f1_water, oa_water, kappa_water = evaluate(args, seg_model, backbone_dino, [1], change_head, valloader_scscd_water, 'SCSCD-Water')
        iou_building, f1_building, oa_building, kappa_building = evaluate(args, seg_model, backbone_dino, [2], change_head, valloader_scscd_building, 'SCSCD-Building')
        iou_structure, f1_structure, oa_structure, kappa_structure = evaluate(args, seg_model, backbone_dino, [3], change_head, valloader_scscd_structure, 'SCSCD-Structure')
        iou_farmland, f1_farmland, oa_farmland, kappa_farmland = evaluate(args, seg_model, backbone_dino, [4], change_head, valloader_scscd_farmland, 'SCSCD-Farmland')
        iou_vegetation, f1_vegetation, oa_vegetation, kappa_vegetation = evaluate(args, seg_model, backbone_dino, [5], change_head, valloader_scscd_vegetation, 'SCSCD-Vegetation')
        iou_road, f1_road, oa_road, kappa_road = evaluate(args, seg_model, backbone_dino, [6], change_head, valloader_scscd_road, 'SCSCD-Road')

        iou_avg = (iou_bareland + iou_water + iou_building + iou_structure + iou_farmland + iou_vegetation + iou_road) / 7.0
        f1_avg = (f1_bareland + f1_water + f1_building + f1_structure + f1_farmland + f1_vegetation + f1_road) / 7.0
        oa_avg = (oa_bareland + oa_water + oa_building + oa_structure + oa_farmland + oa_vegetation + oa_road) / 7.0
        kappa_avg = (kappa_bareland + kappa_water + kappa_building + kappa_structure + kappa_farmland + kappa_vegetation + kappa_road) / 7.0

        print('***** Evaluation {:} ***** >>>> IoU_avg: {:.2f}, F1_avg: {:.2f}, OA_avg: {:.2f}, Kappa_avg: {:.2f}'.format(args.test_dataset, iou_avg * 100, f1_avg * 100, oa_avg * 100, kappa_avg * 100))

        with open(os.path.join(args.save_path, 'test.txt'), 'a') as txt_file:
            txt_file.write('***** Evaluation {:} ***** >>>> IoU_avg: {:.2f}, F1_avg: {:.2f}, OA_avg: {:.2f}, Kappa_avg: {:.2f}'.format(args.test_dataset, iou_avg * 100, f1_avg * 100, oa_avg * 100, kappa_avg * 100) + '\n')

main()