import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from myutils import UnNormalize
from torchvision import transforms


@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 use_vfm=True,
                 version='segearth-ov3',
                 **kwargs):
        super().__init__()
        
        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path=f"./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        if version == 'segearth-ov3':
            self.use_sem_seg = use_sem_seg
            self.use_presence_score = use_presence_score
            self.use_transformer_decoder = use_transformer_decoder
        else:
            self.use_sem_seg = False
            self.use_presence_score = False
            self.use_transformer_decoder = False

        self.use_vfm = use_vfm        

        if use_vfm:
            self.vfm_model = 'dino'
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = self.vfm.half()
            for p in self.vfm.parameters():
                p.requires_grad = False
            self.vfm.eval().to(device)

            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            if self.vfm_model == 'dino':
                self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                    hook_fn_forward_qkv)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.sam_enc_feats = None
        self.dst_v = None

    def get_dino_features(self, imgs_norm):
        patch_size = self.vfm.patch_embed.patch_size                     # 16
        if type(patch_size) is tuple: patch_size = patch_size[0]
        feat = self.vfm.get_intermediate_layers(imgs_norm)[0]            # [N, 442, 768]    patches features
        nb_im = feat.shape[0]  # Batch size                              # N
        vfm_h, vfm_w = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size       # 21, 21
        vfm_feats = feat[:, 1:, :].reshape(nb_im, vfm_h, vfm_w, -1).permute(0, 3, 1, 2)                 # batch, c, h, w  -->  [N, 768, 21, 21]

        return vfm_feats

    def _inference_single_view(self, image, get_feat=False):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)
        instance_mask = []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            # sam_enc_feats = inference_state["backbone_out"]["vision_features"]  # [1, 256, 72, 72]

            if get_feat:
                return inference_state["backbone_out"]["backbone_fpn"]

            for query_idx, query_word in enumerate(self.query_words):  # text prompt loop
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)
                if query_word == 'building':
                    instance_mask = inference_state['masks'].squeeze(1)

                if self.use_transformer_decoder:  # Instance Decoder
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            # instance_mask = inference_state['masks'][inst_id].squeeze()

                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape),
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                
                else:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape),
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze()
                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits)


                if self.use_sem_seg:      # Semantic Decoder
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits,
                                size=(h, w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()

                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)

                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]
                
        return seg_logits, inference_state["backbone_out"]["backbone_fpn"], instance_mask
    
    def get_slide_windows(self, image, h_img, w_img):
        h_crop, w_crop = (512, 512)
        h_stride, w_stride = (512, 512)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        
        crop_imgs = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_imgs.append(crop_img)
        return crop_imgs, h_grids, w_grids

    def get_dst_v(self, image_path):
        h_crop, w_crop = (32, 32)
        h_stride, w_stride = (32, 32)
        image = Image.open(image_path).convert('RGB')
        w_img, h_img = image.size
        img_batch, h_grids, w_grids = self.get_slide_windows(image, h_img, w_img)
        imgs_norm = [self.transform(img_batch[i]) for i in range(len(img_batch))]  # replace norm here
        imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = imgs_norm.cuda().half()
        dst_v = torch.zeros((768, 64, 64), device=self.device)
        count_mat = torch.zeros((1, 64, 64), device=self.device)

        patch_size = self.vfm.patch_embed.patch_size
        dino_feats = self.vfm.get_intermediate_layers(imgs_norm)[0]
        nb_im = dino_feats.shape[0]
        vfm_h, vfm_w = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size       # 32, 32
        v_imgs = dino_feats[:, 1:, :].reshape(nb_im, vfm_h, vfm_w, -1).permute(0, 3, 1, 2)              # [N, 768, 32, 32]

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, 72)
                x2 = min(x1 + w_crop, 72)

                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                idx = h_idx * w_grids + w_idx

                v_img = v_imgs[idx]
                dst_v[:, y1:y2, x1:x2] += v_img
                count_mat[:, y1:y2, x1:x2] += 1

        dst_v = dst_v / count_mat

        return dst_v


    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        h_stride_0, w_stride_0 = (144, 144)
        h_stride_1, w_stride_1 = (72, 72)
        h_stride_2, w_stride_2 = (36, 36)

        h_crop_0, w_crop_0 = (144, 144)
        h_img_0, w_img_0 = (288, 288)

        h_crop_1, w_crop_1 = (72, 72)
        h_img_1, w_img_1 = (144, 144)

        h_crop_2, w_crop_2 = (36, 36)
        h_img_2, w_img_2 = (72, 72)


        # Initialize accumulators
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        sam_enc_feats0 = torch.zeros((256, 288, 288), device=self.device)
        sam_enc_feats1 = torch.zeros((256, 144, 144), device=self.device)
        sam_enc_feats2 = torch.zeros((256, 72, 72), device=self.device)

        sam_enc_feats = []
        instance_masks = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # feature slide windows
                y1_0 = h_idx * h_stride_0
                x1_0 = w_idx * w_stride_0
                y2_0 = min(y1_0 + h_crop_0, h_img_0)
                x2_0 = min(x1_0 + w_crop_0, w_img_0)
                y1_0 = max(y2_0 - h_crop_0, 0)
                x1_0 = max(x2_0 - w_crop_0, 0)

                y1_1 = h_idx * h_stride_1
                x1_1 = w_idx * w_stride_1
                y2_1 = min(y1_1 + h_crop_1, h_img_1)
                x2_1 = min(x1_1 + w_crop_1, w_img_1)
                y1_1 = max(y2_1 - h_crop_1, 0)
                x1_1 = max(x2_1 - w_crop_1, 0)

                y1_2 = h_idx * h_stride_2
                x1_2 = w_idx * w_stride_2
                y2_2 = min(y1_2 + h_crop_2, h_img_2)
                x2_2 = min(x1_2 + w_crop_2, w_img_2)
                y1_2 = max(y2_2 - h_crop_2, 0)
                x1_2 = max(x2_2 - w_crop_2, 0)

                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))

                # Inference on crop
                crop_seg_logit, backbone_fpn, instance_mask = self._inference_single_view(crop_img)
                instance_masks.append(instance_mask)

                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

                backbone_fpn[0] = F.interpolate(backbone_fpn[0], size=(144, 144), mode='bilinear', align_corners=False)
                backbone_fpn[1] = F.interpolate(backbone_fpn[1], size=(72, 72), mode='bilinear', align_corners=False)
                backbone_fpn[2] = F.interpolate(backbone_fpn[2], size=(36, 36), mode='bilinear', align_corners=False)

                sam_enc_feats0[:, y1_0:y2_0, x1_0:x2_0] += backbone_fpn[0].clone().squeeze(0)
                sam_enc_feats1[:, y1_1:y2_1, x1_1:x2_1] += backbone_fpn[1].clone().squeeze(0)
                sam_enc_feats2[:, y1_2:y2_2, x1_2:x2_2] += backbone_fpn[2].clone().squeeze(0)

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        
        preds = preds / count_mat

        sam_enc_feats = [sam_enc_feats0.unsqueeze(0), sam_enc_feats1.unsqueeze(0), sam_enc_feats2.unsqueeze(0)]

        return preds, sam_enc_feats, instance_masks
    
    def slide_get_features(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        h_stride_0, w_stride_0 = (144, 144)
        h_stride_1, w_stride_1 = (72, 72)
        h_stride_2, w_stride_2 = (36, 36)

        h_crop_0, w_crop_0 = (144, 144)
        h_img_0, w_img_0 = (288, 288)

        h_crop_1, w_crop_1 = (72, 72)
        h_img_1, w_img_1 = (144, 144)

        h_crop_2, w_crop_2 = (36, 36)
        h_img_2, w_img_2 = (72, 72)

        # Initialize accumulators
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        sam_enc_feats0 = torch.zeros((256, 288, 288), device=self.device)
        sam_enc_feats1 = torch.zeros((256, 144, 144), device=self.device)
        sam_enc_feats2 = torch.zeros((256, 72, 72), device=self.device)

        sam_enc_feats = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # feature slide windows
                y1_0 = h_idx * h_stride_0
                x1_0 = w_idx * w_stride_0
                y2_0 = min(y1_0 + h_crop_0, h_img_0)
                x2_0 = min(x1_0 + w_crop_0, w_img_0)
                y1_0 = max(y2_0 - h_crop_0, 0)
                x1_0 = max(x2_0 - w_crop_0, 0)

                y1_1 = h_idx * h_stride_1
                x1_1 = w_idx * w_stride_1
                y2_1 = min(y1_1 + h_crop_1, h_img_1)
                x2_1 = min(x1_1 + w_crop_1, w_img_1)
                y1_1 = max(y2_1 - h_crop_1, 0)
                x1_1 = max(x2_1 - w_crop_1, 0)

                y1_2 = h_idx * h_stride_2
                x1_2 = w_idx * w_stride_2
                y2_2 = min(y1_2 + h_crop_2, h_img_2)
                x2_2 = min(x1_2 + w_crop_2, w_img_2)
                y1_2 = max(y2_2 - h_crop_2, 0)
                x1_2 = max(x2_2 - w_crop_2, 0)

                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))

                # Inference on crop
                backbone_fpn = self._inference_single_view(crop_img, get_feat=True)

                backbone_fpn[0] = F.interpolate(backbone_fpn[0], size=(144, 144), mode='bilinear', align_corners=False)
                backbone_fpn[1] = F.interpolate(backbone_fpn[1], size=(72, 72), mode='bilinear', align_corners=False)
                backbone_fpn[2] = F.interpolate(backbone_fpn[2], size=(36, 36), mode='bilinear', align_corners=False)

                sam_enc_feats0[:, y1_0:y2_0, x1_0:x2_0] += backbone_fpn[0].clone().squeeze(0)
                sam_enc_feats1[:, y1_1:y2_1, x1_1:x2_1] += backbone_fpn[1].clone().squeeze(0)
                sam_enc_feats2[:, y1_2:y2_2, x1_2:x2_2] += backbone_fpn[2].clone().squeeze(0)


        sam_enc_feats = [sam_enc_feats0.unsqueeze(0), sam_enc_feats1.unsqueeze(0), sam_enc_feats2.unsqueeze(0)]
        return sam_enc_feats

    def predict(self, inputs, data_samples, get_feats=False):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        if get_feats:
            for i, meta in enumerate(batch_img_metas):
                # Load original image to preserve details for SAM3
                image_path = meta.get('img_path')
                image = Image.open(image_path).convert('RGB')
                ori_shape = meta['ori_shape']

                # Determine inference mode
                if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                    sam_enc_feats = self.slide_get_features(image, self.slide_stride, self.slide_crop)
                else:
                    sam_enc_feats = self._inference_single_view(image, get_feat=True)

                return sam_enc_feats

        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits, sam_enc_feats, instance_masks = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits, sam_enc_feats, instance_masks = self._inference_single_view(image)

            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            seg_pred = torch.argmax(seg_logits, dim=0)

            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx


        return seg_logits, seg_pred.unsqueeze(0), sam_enc_feats, instance_masks

    def _forward(data_samples):
            """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices