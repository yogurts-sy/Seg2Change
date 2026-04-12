import os

from model.ovcd.change_head_fdr_dino import CHead_FPN_FDR as CHead
from seg_model_sam3 import SegEarthOV3Segmentation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pprint
import numpy as np
import cv2
import argparse
import torch.nn as nn
from util.utils import Evaluator, intersectionAndUnion
from torchvision import transforms
from dataset.ovcd import OVCDDataset
from torch.utils.data import DataLoader
from mmseg.structures import SegDataSample

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch.nn.functional as F

GT_COLORMAP = np.array([[[0, 0, 0], [0, 0, 185]], [[0, 178, 0], [255, 255, 255]]])

from model.backbone.dinov2 import DINOv2

from util.utils import make_numpy_grid, visualize_pred, visualize_gt
from util.losses import UnchangedSimilarity

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
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def get_img_feature_dinov2(backbone, encoder_size, img_t):
    sam_enc_feats = backbone.get_intermediate_layers(img_t, intermediate_layer_idx[encoder_size])

    return sam_enc_feats

def get_seg_model_predict(model, img_tensor, img_path, target_classes):
    torch.cuda.empty_cache()

    data_sample = SegDataSample()
    data_sample.set_metainfo({'img_path': img_path, 'ori_shape': img_tensor.shape[1:]})

    seg_logits, seg_pred, sam_enc_feats, _ = model.predict(img_tensor, data_samples=[data_sample])

    pred_mask = torch.zeros_like(seg_pred, dtype=torch.bool).cuda()

    for target_class in target_classes:
        pred_mask[seg_pred == target_class] = True

    return seg_pred, pred_mask.unsqueeze(0), seg_logits, sam_enc_feats


def vis_predict(imgA, imgB, pred, gt, save_dir, epoch_id, batch_id):
    vis_input = make_numpy_grid(imgA)
    vis_input2 = make_numpy_grid(imgB)

    vis_pred = make_numpy_grid(visualize_pred(pred))

    vis_gt = make_numpy_grid(visualize_gt(gt))   
    vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    file_name = os.path.join(
        save_dir, 'train_' + str(epoch_id) + '_' + str(batch_id) + '.png')
    plt.imsave(file_name, vis)


def evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader, dataset_name):
    evaluator_test = Evaluator(num_class=2)

    change_head.eval()
    
    for i, (imgA, imgB, mask, imgA_path, imgB_path, mask_path) in enumerate(valloader):
        imgA, imgB, mask = imgA.cuda(), imgB.cuda(), mask.cuda()

        base_path = os.path.basename(imgA_path[0])
        feat_path = os.path.join(args.feat_path, dataset_name, 'test', base_path + '.pt')
        if os.path.exists(feat_path):
            feats = torch.load(feat_path, weights_only=False)
            pred_mask_A, pred_mask_B = feats['pred_mask_A'], feats['pred_mask_B'] if 'pred_mask_B' in feats else None
            
        else:
            _, pred_mask_A, _, _ = get_seg_model_predict(seg_model, imgA[0], imgA_path[0], target_classes)
            _, pred_mask_B, _, _ = get_seg_model_predict(seg_model, imgB[0], imgB_path[0], target_classes)
            feats = {'pred_mask_A': pred_mask_A, 'pred_mask_B': pred_mask_B}
            os.makedirs(os.path.join(args.feat_path, dataset_name, 'test'), exist_ok=True)
            torch.save(feats, feat_path)

        sam_enc_feats_A = get_img_feature_dinov2(backbone_dino, args.encoder_size, imgA)
        sam_enc_feats_B = get_img_feature_dinov2(backbone_dino, args.encoder_size, imgB)

        output, _, _, _, _, _ = change_head(sam_enc_feats_A, sam_enc_feats_B)
        output = F.interpolate(output, size=(512, 512), mode="bilinear", align_corners=True)

        pred = torch.argmax(output, dim=1, keepdim=True)
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

        with open(os.path.join(args.save_path, 'CM', f'{dataset_name}_IoU.txt'), 'a') as txt_file:
            txt_file.write('{:} IoU: {:1f}'.format(base_path, iou[1]) + '\n')

    IoU = evaluator_test.Intersection_over_Union()[1]
    ACC = evaluator_test.OA()[1]
    Pre = evaluator_test.Precision()[1]
    Recall = evaluator_test.Recall()[1]
    F1 = evaluator_test.F1()[1]
    Kappa = evaluator_test.Kappa()[1]

    print('***** Evaluation %s ***** >>>> IoU: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, OA: %.2f, Kappa: %.4f' % (dataset_name, IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa))

    with open(os.path.join(args.save_path, f'{dataset_name}_IoU.txt'), 'a') as txt_file:
        txt_file.write('\n***** Evaluation {:} ***** >>>> IoU: {:.2f}, F1: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, OA: {:.2f}, Kappa: {:.4f}\n'.format(dataset_name, IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa) + '\n')

    with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
        txt_file.write('***** Evaluation {:} ***** >>>> IoU: {:.2f}, F1: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, OA: {:.2f}, Kappa: {:.4f}'.format(dataset_name, IoU * 100, F1 * 100, Pre * 100, Recall * 100, ACC * 100, Kappa) + '\n')

    return IoU

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

def main():
    parser = argparse.ArgumentParser(description='Seg2Change')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--ft_dataset', type=str, default='CACDD', help='training dataset name')
    parser.add_argument('--dino_ft', type=str, default='frozen', help='DINOv2 fine-tuning strategy: full, attention')
    parser.add_argument('--crop_size', type=int, default=504, help='training dataset crop size')
    parser.add_argument('--encoder_size', type=str, default='base', help='DINOv2 encoder size: small, base, large')
    parser.add_argument('--val_dataset', type=str, default='whu_levir', help='val dataset, whu_levir, second')
    parser.add_argument('--loss_multi', type=bool, default=True, help='whether to use multi-scale loss')
    parser.add_argument('--loss_sim', type=bool, default=True, help='whether to use unchanged similarity loss')
    parser.add_argument('--feat_path', type=str, default='/data2/suyou/Codes/Seg2Change/features/features_ck03', help='training feature cache path')
    parser.add_argument('--save_path', type=str, default='/data2/suyou/Codes/Seg2Change/exp/train_cach_dino_base_CACD/test02', help='path to save training logs and models')
    parser.add_argument('--dataset_root_path', type=str, default='/data2/suyou/Codes/datasets/OVCD_Benchmark/', help='training dataset root path')
    parser.add_argument('--nclass', type=int, default=2)

    args = parser.parse_args()

    pretrain = False

    if args.ft_dataset == 'SECOND':
        args.dataset_root = args.dataset_root_path + 'SECOND/'
        args.labeled_id = args.dataset_root_path + 'SECOND/train.txt'

        A_name = 'T1'
        B_name = 'T2'
        label_name = 'GT_CD'

    elif args.ft_dataset == 'JL1-CD':
        args.dataset_root = args.dataset_root_path + 'JL1-CD-512/'
        args.labeled_id = None

        A_name = 'A'
        B_name = 'B'
        label_name = 'label'

    elif args.ft_dataset == 'CNAM':
        args.dataset_root = args.dataset_root_path + 'CNAM-CD/'
        args.labeled_id = None

        A_name = 'A'
        B_name = 'B'
        label_name = 'label'

        pretrain = True

    elif args.ft_dataset == 'CACDD':
        args.dataset_root = args.dataset_root_path + 'CACDD/'
        args.labeled_id = None

        A_name = 'A'
        B_name = 'B'
        label_name = 'label'

        pretrain = True

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'CM'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'A'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'B'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'CK'), exist_ok=True)

    all_args = {**vars(args)}
    with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
        txt_file.write('{}\n'.format(pprint.pformat(all_args)))

    if args.encoder_size == 'small':
        state_dict = torch.load('./weights/dinov2/dinov2_vits14_pretrain.pth', weights_only=False)
        dino_dim = 384
    elif args.encoder_size == 'base':
        state_dict = torch.load('./weights/dinov2/dinov2_vitb14_pretrain.pth', weights_only=False)
        dino_dim = 768
    backbone_dino = DINOv2(model_name=args.encoder_size).cuda()
    backbone_dino.load_state_dict(state_dict)
    backbone_dino = build_dino_set_finetune(backbone_dino, args.dino_ft)

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()

    criterion_sim = UnchangedSimilarity(T=3.0).cuda()

    trainset = OVCDDataset('train', args.dataset_root, args.ft_dataset, args.crop_size, args.labeled_id, pretrain=pretrain, A=A_name, B=B_name, label=label_name)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=False, shuffle=True)

    if args.val_dataset == 'whu_levir':
        valset_whu = OVCDDataset('test', args.dataset_root_path + 'WHU-CD-512/', "WHU-CD", args.crop_size)
        valset_levir = OVCDDataset('test', args.dataset_root_path + 'LEVIR-CD-512/', "LEVIR-CD", args.crop_size)
        valloader_whu = DataLoader(valset_whu, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_levir = DataLoader(valset_levir, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
    else:  # val_dataset == 'second':
        valset_second_building = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='building')
        valset_second_tree = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='tree')
        valset_second_water = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='water')
        valset_second_grass = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='grass')
        valset_second_ground = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='ground')
        valset_second_playground = OVCDDataset('test', args.dataset_root_path + 'SECOND/', "SECOND", args.crop_size, classname='playground')
        valloader_second_building = DataLoader(valset_second_building, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_tree = DataLoader(valset_second_tree, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_water = DataLoader(valset_second_water, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_grass = DataLoader(valset_second_grass, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_ground = DataLoader(valset_second_ground, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)
        valloader_second_playground = DataLoader(valset_second_playground, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False)

    name_list = ['background', 'bareland,barren', 'grass', 'road', 'car',
             'tree,forest', 'water,river', 'cropland', 'building']

    with open('./configs/my_name.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list)-1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
    writers.close()

    # Open-Vocabulary Semantic Segmentation Model
    seg_model = SegEarthOV3Segmentation(type='SegEarthOV3Segmentation', model_type='SAM3', classname_path='./configs/my_name.txt', 
        prob_thd=0.1, confidence_threshold=0.1, slide_stride=512, slide_crop=512, use_vfm=False)

    # Category-Agnostic Change Head
    change_head = CHead(in_channels=dino_dim, hidden_dim=64, nclass=args.nclass, multi=args.loss_multi).cuda()

    optimizer = torch.optim.AdamW(change_head.parameters(), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(trainloader))
    
    target_classes = [8] # Foreground class index in SAM3 segmentation output, corresponding to 'building' in our setting
    epoch = -1
    previous_best_iou_avg = 0.0

    print('Change Head Total params: {:.1f}M\n'.format(count_params(change_head)))

    for epoch in range(epoch + 1, args.epochs):
        loss_list = []
        torch.cuda.empty_cache()
        change_head.train()

        print('===========> Epoch: {:}, Previous best AVG {:} Changed IoU: {:.2f}'.format(epoch, args.val_dataset, previous_best_iou_avg * 100))
        with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
            txt_file.write('\n===========> Epoch: {:}, Previous best AVG {:} Changed IoU: {:.2f}\n'.format(epoch, args.val_dataset, previous_best_iou_avg * 100))

        for i, (imgA, imgB, mask, A, B) in enumerate(trainloader):

            imgA, imgB, mask = imgA.cuda(), imgB.cuda(), mask.cuda()

            sam_enc_feats_A = get_img_feature_dinov2(backbone_dino, args.encoder_size, imgA)
            sam_enc_feats_B = get_img_feature_dinov2(backbone_dino, args.encoder_size, imgB)

            output, output1, output2, output3, outputA, outputB = change_head(sam_enc_feats_A, sam_enc_feats_B)
            loss_cd = criterion(output, mask)
            loss_sim = criterion_sim(outputA, outputB, mask)

            loss_multi = criterion(output1, mask) + criterion(output2, mask) + criterion(output3, mask)
            loss = loss_cd * 0.8 + loss_multi * 0.1 + loss_sim * 0.1

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            if i % 128 == 0:
                vis_predict(A, B, output, mask, os.path.join(args.save_path, 'CM'), epoch, i)
                print(f"Epoch [{epoch}/{args.epochs}] Iter [{i}/{len(trainloader)}] Loss: {sum(loss_list)/len(loss_list):.4f}")

                with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
                    txt_file.write(f"Epoch [{epoch}/{args.epochs}] Iter [{i}/{len(trainloader)}] Loss: {sum(loss_list)/len(loss_list):.4f}\n")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {sum(loss_list)/len(loss_list):.4f} LR: {current_lr:.6f}\n")
        with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
            txt_file.write(f"Epoch [{epoch}/{args.epochs}] Loss: {sum(loss_list)/len(loss_list):.4f} LR: {current_lr:.6f}\n\n")

        print()
        print('Evaluation: ')

        if args.val_dataset == 'whu_levir':
            iou_whu = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_whu, 'WHU-CD-512')
            iou_levir = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_levir, 'LEVIR-CD-512')

            iou_avg = (iou_whu + iou_levir) / 2.0

        else:
            iou_building = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_building, 'SECOND-Building')
            iou_tree = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_tree, 'SECOND-Tree')
            iou_water = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_water, 'SECOND-Water')
            iou_grass = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_grass, 'SECOND-Grass')
            iou_ground = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_ground, 'SECOND-Ground')
            iou_playground = evaluate(args, seg_model, backbone_dino, target_classes, change_head, valloader_second_playground, 'SECOND-Playground')

            iou_avg = (iou_building + iou_tree + iou_water + iou_grass + iou_ground + iou_playground) / 6.0

        is_best = iou_avg > previous_best_iou_avg
        if is_best:
            previous_best_iou_avg = iou_avg
            torch.save(change_head.state_dict(), os.path.join(args.save_path, 'CK', 'best.pth'))
            print('Best model saved with Avg IoU: {:.2f}\n'.format(previous_best_iou_avg * 100))
            with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
                txt_file.write('Best model saved with Avg IoU: {:.2f}\n\n'.format(previous_best_iou_avg * 100))
        else:
            with open(os.path.join(args.save_path, 'train.txt'), 'a') as txt_file:
                txt_file.write('\n\n')

        print()     

main()