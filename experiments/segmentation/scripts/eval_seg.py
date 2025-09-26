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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dpath_pred', type=str, help='Directory path for prediction results')
    parser.add_argument('dpath_gt', type=str, help='Directory path for ground truth images')
    parser.add_argument('dpath_out', type=str, help='')
    parser.add_argument('num_classes', type=int, help='')
    parser.add_argument('--ignore_index', nargs='*', default=[], type=int, help='')
    parser.add_argument('--margin', type=int, default=0, help='Ignore boundary area of ground truth polygons by specified margin')
    parser.add_argument('--pred_type', type=str, default='dir', help='')
    parser.add_argument('--gt_type'  , type=str, default='dir', help='')
    parser.add_argument('--pred_ext' , type=str, default='png', help='')
    parser.add_argument('--gt_ext'   , type=str, default='png', help='')
    parser.add_argument('--pred_hdf5_path', type=str, help='')
    parser.add_argument('--gt_hdf5_path'  , type=str, help='')
    parser.add_argument('--plot', action='store_true', help='Plot prediction and GT side by side')
    args = parser.parse_args()

import numpy as np
import sys
import os
import glob
import time
import matplotlib.pyplot as plt

from hmnet.utils.common import get_list, makedirs, ImageFiles

def get_loader(dpath, type, ext, hdf5_path):
    if type == 'dir':
        return ImageFiles.open_dir(dpath, ext)
    elif type == 'npy':
        return ImageFiles.open_npy(dpath)
    elif type == 'hdf5':
        return ImageFiles.open_hdf5(dpath, hdf5_path)
    elif type == 'npy_files':
        return ImageFiles.open_npy_files(dpath)
    elif type == 'hdf5_files':
        return ImageFiles.open_hdf5_files(dpath, hdf5_path)
    else:
        raise RuntimeError
    
def plot_pred_gt(pred, gt, dpath_out, i):
    plt.clf()
    if not os.path.exists(dpath_out):
        os.makedirs(dpath_out)
    # --- Plot and Save ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred)
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(gt)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Format filename with progressive number, fixed number of zeros (e.g., 0001, 0002, ...)
    fname = f"{i:04d}.png"   # change 4 → 5 or 6 if you expect >9999 images
    save_path = os.path.join(dpath_out, fname)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def eval_seg(dpath_pred, dpath_gt, evalid_to_gtid, ignore_gtid, margin, dpath_out=None,
                          pred_type='dir', pred_ext='png', pred_hdf5_path='',
                          gt_type='dir', gt_ext='png', gt_hdf5_path='', plot=False):

    dpath_out = dpath_out or dpath_pred + '/logs/'
    makedirs(dpath_out)

    pred_list = get_loader(dpath_pred, pred_type, pred_ext, pred_hdf5_path)
    gt_list = get_loader(dpath_gt, gt_type, gt_ext, gt_hdf5_path)

    all_intersection = { k:0 for k in evalid_to_gtid.keys() }
    all_union = { k:0 for k in evalid_to_gtid.keys() }
    all_pixels = 0
    all_tp = 0
    all_scene = {}

    log_scene = ''
    st = time.time()
    for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
        fpath_pred = pred_list.filename(i)
        fpath_gt = gt_list.filename(i)

        if pred.ndim == 3:
            assert pred.shape[0] == 1
            pred = pred[0]
        if gt.ndim == 3:
            assert gt.shape[0] == 1
            gt = gt[0]

        scene_iou = []
        scene_tp = 0
        scene_pixels = 0

        hp, wp = pred.shape
        hg, wg = gt.shape
        if hp < hg or wp < wg:
            print('Size mismatch!! Executing center crop for GT: GT(%d,%d) PRED(%d,%d)' % (hg,wg,hp,wp))
            st_x = int(round((wg - wp) / 2))
            st_y = int(round((hg - hp) / 2))
            gt = gt[st_y:st_y+hp, st_x:st_x+wp]
            hg, wg = gt.shape
        if hg < hp or wg < wp:
            print('Size mismatch!! Executing center crop for PRED: GT(%d,%d) PRED(%d,%d)' % (hg,wg,hp,wp))
            st_x = int(round((wp - wg) / 2))
            st_y = int(round((hp - hg) / 2))
            pred = pred[st_y:st_y+hg, st_x:st_x+wg]
            hp, wp = pred.shape

        ignore_mask = np.zeros_like(gt)
        for ign in ignore_gtid:
            ignore_mask = np.logical_or(ignore_mask, gt == ign)

        # Plot prediction and GT side to side
        if plot:
            seq_name = os.path.splitext(os.path.basename(fpath_pred))[0]
            plot_pred_gt(pred, gt, dpath_out + '/figs/' + f'{seq_name}', i)

        pred[ignore_mask] = -1
        gt[ignore_mask] = -1

        for k,v in evalid_to_gtid.items():
            if v in ignore_gtid:
                continue

            t = gt == v
            p = pred == k
            union = np.logical_or(t,p)
            intersection = np.logical_and(t, p)

            u_sum = union.sum()
            i_sum = intersection.sum()
            t_sum = t.sum()

            all_union[k] += u_sum
            all_intersection[k] += i_sum
            all_tp += i_sum
            all_pixels += t_sum

            scene_tp += i_sum
            scene_pixels += t_sum

            if t_sum > 0:
                iou = float(i_sum) / u_sum
                scene_iou.append(iou)

        scene_miou = np.array(scene_iou).mean()
        scene_acc = float(scene_tp / scene_pixels)

        log = '%d / %d (%.2f sec) mIoU=%.2f acc=%.2f %s %s' % (i, len(pred_list), time.time()-st, scene_miou*100, scene_acc*100, fpath_pred, fpath_gt)
        print(log)
        log_scene += log + '\n'
        st = time.time()

    iou = {}
    for k in evalid_to_gtid.keys():
        if all_union[k] == 0:
            continue
        iou[k] = float(all_intersection[k]) / all_union[k]
    miou = np.array(list(iou.values())).mean()
    acc = float(all_tp) / all_pixels

    with open(dpath_out + '/result_eval_pixel_multiclass_margin%d_scene.txt' % margin, 'w') as fp:
        fp.write(log_scene)

    with open(dpath_out + '/result_eval_pixel_multiclass_margin%d.txt' % margin, 'w') as fp:
        str = '===========================\n'
        str += 'setting: margin pixels = %d\n' % margin
        str += '---------------------------\n'
        str += 'class,iou\n'
        for c,v in iou.items():
            str += '%s, %.2f%%\n' % (c,v*100)
        str += '---------------------------\n'
        str += 'mean IoU = %.2f%%\n' % (miou * 100)
        str += 'overall accuracy = %.2f%%\n' % (acc * 100)
        str += '===========================\n'
        fp.write(str)
        print(str)

if __name__ == '__main__':
    evalid_to_gtid = { c:c for c in range(args.num_classes)}

    eval_seg(args.dpath_pred, args.dpath_gt, evalid_to_gtid, args.ignore_index, args.margin, dpath_out=args.dpath_out,
                          pred_type=args.pred_type, gt_type=args.gt_type, pred_ext=args.pred_ext, gt_ext=args.gt_ext,
                          pred_hdf5_path=args.pred_hdf5_path, gt_hdf5_path=args.gt_hdf5_path, plot=args.plot)




