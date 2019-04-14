# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Evaluation script for object grounding over GT sentences

import json
import argparse
import torch
import itertools
import numpy as np
from collections import defaultdict
from utils import bbox_overlaps_batch, get_frm_mask

def main(args):

    with open(args.reference) as f:
        ref = json.load(f)['annotations']
    with open(args.split_file) as f:
        split_file = json.load(f)
    split = {}
    for s in args.split:
        split.update({i:i for i in split_file[s]})
    ref = {k:v for k,v in ref.items() if k in split}

    with open(args.submission) as f:
        pred = json.load(f)['results']

    print('Number of videos in the reference: {}, number of videos in the submission: {}'.format(len(ref), len(pred)))

    results = defaultdict(list)
    for vid, anns in ref.items():
        for seg, ann in anns['segments'].items():
            if len(ann['frame_ind']) == 0:
                continue # annotation not available

            ref_bbox_all = torch.cat((torch.Tensor(ann['process_bnd_box']), \
                torch.Tensor(ann['frame_ind']).unsqueeze(-1)), dim=1) # 5-D coordinates
            sent_idx = set(itertools.chain.from_iterable(ann['process_idx'])) # index of word in sentence to evaluate
            for idx in sent_idx:
                sel_idx = [ind for ind, i in enumerate(ann['process_idx']) if idx in i]
                ref_bbox = ref_bbox_all[sel_idx] # select matched boxes
                # Note that despite discouraged, a single word could be annotated across multiple boxes/frames
                assert(ref_bbox.size(0) > 0)

                class_name = ann['process_clss'][sel_idx[0]][ann['process_idx'][sel_idx[0]].index(idx)]
                if vid not in pred:
                    results[class_name].append(0) # video not grounded
                elif seg not in pred[vid]:
                    results[class_name].append(0) # segment not grounded
                elif idx not in pred[vid][seg]['idx_in_sent']:
                    results[class_name].append(0) # object not grounded
                else:
                    pred_ind = pred[vid][seg]['idx_in_sent'].index(idx)
                    pred_bbox = torch.cat((torch.Tensor(pred[vid][seg]['bbox_for_all_frames'][pred_ind])[:,:4], \
                        torch.Tensor(range(10)).unsqueeze(-1)), dim=1)

                    frm_mask = torch.from_numpy(get_frm_mask(pred_bbox[:, 4].numpy(), \
                        ref_bbox[:, 4].numpy()).astype('uint8'))
                    overlap = bbox_overlaps_batch(pred_bbox[:, :5].unsqueeze(0), \
                        ref_bbox[:, :5].unsqueeze(0), frm_mask.unsqueeze(0))
                    results[class_name].append(1 if torch.max(overlap) > args.iou else 0)

    print('Number of groundable objects in this split: {}'.format(len(results)))
    grd_accu = np.mean([sum(hm)*1./len(hm) for i,hm in results.items()])

    print('-' * 80)
    print('The overall grounding accuracy is {:.4f}'.format(grd_accu))
    print('-' * 80)
    if args.verbose:
        print('Object frequency and grounding accuracy per class (descending by object frequency):')
        accu_per_clss = {(i, sum(hm)*1./len(hm)):len(hm) for i,hm in results.items()}
        accu_per_clss = sorted(accu_per_clss.items(), key=lambda x:x[1], reverse=True)
        for accu in accu_per_clss:
            print('{} ({}): {:.4f}'.format(accu[0][0], accu[1], accu[0][1]))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ActivityNet-Entities object grounding evaluation script.')
    parser.add_argument('-s', '--submission', type=str, default='', help='submission grounding result file')
    parser.add_argument('-r', '--reference', type=str, default='data/anet_entities_cleaned_class_thresh50_trainval.json', help='reference file')
    parser.add_argument('--split_file', type=str, default='data/split_ids_anet_entities.json', help='path to the split file')
    parser.add_argument('--split', type=str, nargs='+', default=['validation'], help='which split(s) to evaluate')
    parser.add_argument('-iou', type=float, default=0.5, help='the iou threshold for grounding correctness')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
