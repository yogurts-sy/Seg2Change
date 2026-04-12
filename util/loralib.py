import os
import argparse
import torch
import torch.nn as nn

from typing import Dict

from tqdm import tqdm

from .layers import AdaMultiheadAttentionLoRA, LoRALayer, PlainMultiheadAttentionLoRA

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]

    return acc

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            ...
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def apply_LSAda(args, clip_model):
    list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = AdaMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = AdaMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers

def save_lora(args, save_path, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }


    save_path = f'{save_path}/best_lora_10.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def save_adalora(args, save_path, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'lora_A': layer.q_proj.lora_A.data,
                'lora_E': layer.q_proj.lora_E.data,
                'lora_B': layer.q_proj.lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'lora_A': layer.k_proj.lora_A.data,
                'lora_E': layer.k_proj.lora_E.data,
                'lora_B': layer.k_proj.lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'lora_A': layer.v_proj.lora_A.data,
                'lora_E': layer.v_proj.lora_E.data,
                'lora_B': layer.v_proj.lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'lora_A': layer.proj.lora_A.data,
                'lora_E': layer.proj.lora_E.data,
                'lora_B': layer.proj.lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }


    save_path = f'{save_path}/best_adaLoRA.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')

def save_adalora(args, save_path, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'lora_A': layer.q_proj.lora_A.data,
                'lora_E': layer.q_proj.lora_E.data,
                'lora_B': layer.q_proj.lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'lora_A': layer.k_proj.lora_A.data,
                'lora_E': layer.k_proj.lora_E.data,
                'lora_B': layer.k_proj.lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'lora_A': layer.v_proj.lora_A.data,
                'lora_E': layer.v_proj.lora_E.data,
                'lora_B': layer.v_proj.lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'lora_A': layer.proj.lora_A.data,
                'lora_E': layer.proj.lora_E.data,
                'lora_B': layer.proj.lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }


    save_path = f'{save_path}/best_adaLoRA.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')

def save_LSAda(args, save_path, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'lora_A': layer.q_proj.lora_A.data,
                'lora_E': layer.q_proj.lora_E.data,
                'lora_B': layer.q_proj.lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'lora_A': layer.k_proj.lora_A.data,
                'lora_E': layer.k_proj.lora_E.data,
                'lora_B': layer.k_proj.lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'lora_A': layer.v_proj.lora_A.data,
                'lora_E': layer.v_proj.lora_E.data,
                'lora_B': layer.v_proj.lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'lora_A': layer.proj.lora_A.data,
                'lora_E': layer.proj.lora_E.data,
                'lora_B': layer.proj.lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    save_path = f'{save_path}/best_LSAda.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')

def load_lora(args, save_path, list_lora_layers):
    # to manage names like ViT-B/16
    load_path = f'{save_path}/best_lora_10.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}\n')


def load_adalora(args, save_path, list_lora_layers):
    # to manage names like ViT-B/16
    load_path = f'{save_path}/best_adaLoRA.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.lora_A.data.copy_(
                layer_weights['q_proj']['lora_A'])
            layer.q_proj.lora_E.data.copy_(
                layer_weights['q_proj']['lora_E'])
            layer.q_proj.lora_B.data.copy_(
                layer_weights['q_proj']['lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.lora_A.data.copy_(
                layer_weights['k_proj']['lora_A'])
            layer.k_proj.lora_E.data.copy_(
                layer_weights['k_proj']['lora_E'])
            layer.k_proj.lora_B.data.copy_(
                layer_weights['k_proj']['lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.lora_A.data.copy_(
                layer_weights['v_proj']['lora_A'])
            layer.v_proj.lora_E.data.copy_(
                layer_weights['v_proj']['lora_E'])
            layer.v_proj.lora_B.data.copy_(
                layer_weights['v_proj']['lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.lora_A.data.copy_(layer_weights['proj']['lora_A'])
            layer.proj.lora_E.data.copy_(layer_weights['proj']['lora_E'])
            layer.proj.lora_B.data.copy_(layer_weights['proj']['lora_B'])

    print(f'LoRA weights loaded from {load_path}\n')

def load_LSAda(args, save_path, clip_model):
    # to manage names like ViT-B/16
    # load_path = f'{save_path}/best_LSAda.pt'
    load_path = save_path

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")
    
    list_lora_layers = apply_LSAda(args, clip_model)
    clip_model = clip_model.cuda()

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.lora_A.data.copy_(
                layer_weights['q_proj']['lora_A'])
            layer.q_proj.lora_E.data.copy_(
                layer_weights['q_proj']['lora_E'])
            layer.q_proj.lora_B.data.copy_(
                layer_weights['q_proj']['lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.lora_A.data.copy_(
                layer_weights['k_proj']['lora_A'])
            layer.k_proj.lora_E.data.copy_(
                layer_weights['k_proj']['lora_E'])
            layer.k_proj.lora_B.data.copy_(
                layer_weights['k_proj']['lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.lora_A.data.copy_(
                layer_weights['v_proj']['lora_A'])
            layer.v_proj.lora_E.data.copy_(
                layer_weights['v_proj']['lora_E'])
            layer.v_proj.lora_B.data.copy_(
                layer_weights['v_proj']['lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.lora_A.data.copy_(layer_weights['proj']['lora_A'])
            layer.proj.lora_E.data.copy_(layer_weights['proj']['lora_E'])
            layer.proj.lora_B.data.copy_(layer_weights['proj']['lora_B'])

    print(f'LSAda weights loaded from {load_path}\n')
    
    return clip_model


def evaluate_lora(clip_model, loader, clip_weights):
    clip_model.eval()
    
    acc = 0.
    correct_samples, all_samples = 0, 0

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            
            image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            clip_probs = 100. * image_features @ clip_weights
            acc = cls_acc(clip_probs, target)
            correct_samples += acc / 100 * len(clip_probs)
            all_samples += len(clip_probs)

    acc = correct_samples / all_samples

    print("**** UCD-LSAda's Test Acc: {:.2f} ({:}/{:}) ****\n".format(acc, correct_samples, all_samples))
    return acc


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # parser.add_argument('--backbone', default='ViT-B/32', type=str)
    # Training arguments
    parser.add_argument('--lr', default=1e-3, type=float) # 2e-4, 1e-3
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v', 'o'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=6, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=4, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args