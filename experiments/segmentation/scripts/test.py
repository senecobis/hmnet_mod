# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of {{ project }} nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('data_list', type=str, help='Path for directory containing file lists for testing')
    parser.add_argument('data_root', type=str, help='Path for dataset root directory')
    parser.add_argument('--mode', type=str, default='single_process', choices=('single_process', 'multi_process', 'cuda_stream'), help='')
    parser.add_argument('--speed_test', action='store_true', help='Measure inference time')
    parser.add_argument('--name', type=str, default=None, help='Name of the model. (default value is set by this script name)')
    parser.add_argument('--gpuid', type=str, default=0, help='GPU ID')
    parser.add_argument('--cpu', action='store_true', help='Run in CPU mode')
    parser.add_argument('--test_chunks', type=str, default='1/1', help='"{CHUNK_ID}/{NUM_CHUNKS}": Split test data into NUM_CHUNKS and run inference on a specified CHUNK_ID.')
    parser.add_argument('--pretrained', type=str, help='Path for the pretrained weight (checkpoint file in workspace will be loaded by default)')
    parser.add_argument('--random_init', action='store_true', help='Run without pretrained weights')
    parser.add_argument('--devices', type=int, nargs='*', help='')
    parser.add_argument('--fast', action='store_true', help='Convert to fast model')
    parser.add_argument('--fp16', action='store_true', help='Run in FP16 mode')
    parser.add_argument('--fuse_right', action='store_true', help='Use right images for fusion models')
    args = parser.parse_args()

# ======= for debug ===========
DEBUG = False
PREFIX = './debug/preds'
#PREFIX = './debug/preds_img'
# =============================

import numpy as np
import re 
import sys
import copy
from tqdm import tqdm
from importlib import machinery
from PIL import Image
from functools import partial
from numpy.lib import recfunctions as rfn

import torch
from torch.cuda.amp import autocast
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.utils.common import fix_seed, get_list, get_chunk, mkdir, makedirs, Timer

# cudnn benchmark mode
torch.backends.cudnn.benchmark = True

timer = Timer()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
@torch.no_grad()
def main(config):
    # set device
    if config.cpu:
        config.device = torch.device('cpu')
    else:
        config.device = torch.device('cuda:%d' % config.gpuid)

    # set seed
    fix_seed(42)

    # get model
    model = config.get_model(config.devices, config.mode)

    if hasattr(model, 'to_cuda'):
        model.to_cuda()
    else:
        model = model.to(config.device)
    model.eval()

    # load pretrained weights
    if config.pretrained is None:
        fpath_checkpoint = config.dpath_work + '/' + config.checkpoint
    else:
        fpath_checkpoint = config.pretrained

    if not config.random_init:
        print("=> loading checkpoint '{}'".format(fpath_checkpoint))
        state_dict = get_state_dict(fpath_checkpoint, config.device)
        model.load_state_dict(state_dict)

    # convert to fast model
    if config.fast:
        model.to_fast_model()

    # get list
    list_fpath_evt = get_list(config.fpath_evt_lst, ext=None)
    list_fpath_rgb = get_list(config.fpath_rgb_lst, ext=None)
    list_fpath_lbl = get_list(config.fpath_lbl_lst, ext=None)

    # split targets into chunks
    list_fpath_evt = get_chunk(list_fpath_evt, chunk_str=config.test_chunks)
    list_fpath_rgb = get_chunk(list_fpath_rgb, chunk_str=config.test_chunks)
    list_fpath_lbl = get_chunk(list_fpath_lbl, chunk_str=config.test_chunks)

    for fpath_evt, fpath_rgb, fpath_lbl in zip(list_fpath_evt, list_fpath_rgb, list_fpath_lbl):
        # get dataset
        dataset = config.get_dataset(fpath_evt, fpath_rgb, fpath_lbl, config.fpath_meta, config.fpath_video_duration, config.data_root, fast_mode=config.fast, debug=DEBUG)
        loader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,  # MUST be 0 because dataset.event_transform is needed for backward transformation
                                             batch_size=config.batch_size,
                                             collate_fn=collate_keep_dict)

        results = {}
        for i, data in tqdm(enumerate(loader)):
            events, images, image_metas = parse_event_data(data)

            if getattr(config, 'to_device_in_model', False) == False:
                events = to_device(events, config.device)
                images = to_device(images, config.device)

            with autocast(enabled=config.fp16):
                preds, out_image_metas = model.inference(events, images, image_metas, speed_test=config.speed_test)
            
            if loader.dataset.event_transform is not None:
                preds, out_image_metas = backward_transform(preds, out_image_metas, loader.dataset.event_transform)

            for pred, meta in zip(preds, out_image_metas):
                gtidx = meta['label_path']
                if gtidx is not None:
                    results[gtidx] = np.argmax(pred.cpu().numpy(), axis=0)

            # for debug
            if DEBUG:
                #torch.save(preds, f'{PREFIX}_{i}.pth')
                preds_ref = torch.load(f'{PREFIX}_{i}.pth', map_location='cpu')
                max_idx = preds_ref.argmax(dim=1, keepdim=True)
                a = preds_ref.gather(dim=1, index=max_idx)
                b = preds[-1:].gather(dim=1, index=max_idx)
                print('%.2e' % ((a - b).abs().mean().item()))

        print(f'\nwriting results')

        key_max = max(results.keys())
        Ngt = key_max + 1
        H, W = results[key_max].shape
        dtype = results[key_max].dtype

        output = np.zeros([Ngt,1,H,W], dtype=dtype) - 1
        for k, v in results.items():
            output[k] = v

        outfile = fpath_evt.split('/')[-1].replace('.hdf5', '.npy')
        fpath_out = f'{config.dpath_out}/{outfile}'
        np.save(fpath_out, output)

def backward_transform(preds, img_metas, transform):
    out_preds = []
    out_img_metas = []
    for pred, img_meta in zip(preds, img_metas):
        pred, img_meta = transform.backward(pred, img_meta, types=['image', 'meta'])
        out_preds.append(pred)
        out_img_metas.append(img_meta)

    return torch.stack(out_preds), out_img_metas

def parse_event_data(data):
    def _nested_shape(lst, shape=[]):
        if isinstance(lst, (list, tuple)):
            shape += [len(lst)]
            return _nested_shape(lst[0], shape)
        else:
            return shape

    datas, targets, metas = data
    shape = _nested_shape(metas)

    if len(shape) == 2:
        list_events, list_images, list_image_metas = [], [], []
        for d, t, m in zip(datas, targets, metas):
            events, images, image_metas = parse_event_data([d, t, m])
            list_events.append(events)
            list_images.append(images)
            list_image_metas.append(image_metas)
        return list_events, list_images, list_image_metas
    else:
        events = [ d['events'] for d in datas ]
        images = [ d['images'] for d in datas ]
        image_metas = [ m['image_meta'] for m in metas ]

        if events[0].ndim == 3:
            events = torch.stack(events)

        return events, images, image_metas

def to_device(data, device, non_blocking=True):
    if data is None:
        return data
    elif isinstance(data, (list, tuple)):
        return [ to_device(d, device, non_blocking) for d in data ]
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    else:
        return data

def get_state_dict(fpath, device):
    state_dict = torch.load(fpath, map_location=device)
    if 'state_dict' in state_dict:
        return state_dict['state_dict']
    else:
        return state_dict

def get_config(args):
    config_module = machinery.SourceFileLoader('config', args.config).load_module()
    config = config_module.TestSettings()

    config.fpath_evt_lst = f'{args.data_list}/events.txt'
    if args.fuse_right:
        config.fpath_rgb_lst = f'{args.data_list}/images_right.txt'
    else:
        config.fpath_rgb_lst = f'{args.data_list}/images.txt'
    config.fpath_lbl_lst = f'{args.data_list}/labels.txt'
    config.fpath_meta    = f'{args.data_list}/meta.pkl'
    config.fpath_video_duration = f'{args.data_list}/video_duration.csv'
    config.data_root = args.data_root

    config.cpu = args.cpu
    config.gpuid = args.gpuid
    config.test_chunks = args.test_chunks
    config.pretrained = args.pretrained
    config.random_init = args.random_init
    config.fast = args.fast
    config.fp16 = args.fp16
    config.devices = args.devices
    config.speed_test = args.speed_test
    config.mode = args.mode

    name = args.config.split('/')[-1].replace('.py', '')
    dirname = get_dirname(args.data_list)
    config.dpath_work = f'./workspace/{name}'

    # ---- NEW: build results dir name from checkpoint ----
    # Default tag if we can't detect a checkpoint number.
    ckpt_tag = "default"
    if config.random_init:
        ckpt_tag = "random_init"
    else:
        # Prefer explicit --pretrained, otherwise fall back to config.checkpoint if present.
        ckpt_source = args.pretrained
        if ckpt_source is None and hasattr(config, "checkpoint"):
            ckpt_source = config.checkpoint

        if ckpt_source is not None:
            base = os.path.basename(str(ckpt_source))
            # Try to extract a number from names like "checkpoint_8.pth.tar"
            m = re.search(r'checkpoint[_\-]?(\d+)', base)
            if m:
                ckpt_tag = m.group(1)
            else:
                # Use the stem (without extensions) if no number is found, e.g., "checkpoint_best"
                ckpt_tag = os.path.splitext(os.path.splitext(base)[0])[0]

    # Final path: ./workspace/<name>/results_checkpoint_<tag>/pred_<splitname>
    config.dpath_out = f'./workspace/{name}/results_checkpoint_{ckpt_tag}/pred_{dirname}'
    # -----------------------------------------------------

    return config

def get_dirname(path):
    if path.endswith('/'):
        path = path[:-1]
    return path.split('/')[-1].split('.')[0]

if __name__ == '__main__':
    __spec__ = None
    config = get_config(args)
    makedirs(config.dpath_out)
    main(config)

