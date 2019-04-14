# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Evaluation script for object grounding over generated sentences

import json
import argparse
import torch
import itertools
import numpy as np
from collections import defaultdict
from utils import bbox_overlaps_batch, get_frm_mask

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

def main(args):

    nlp = StanfordCoreNLP('tools/stanford-corenlp-full-2018-02-27')
    props={'annotators': 'lemma','pipelineLanguage':'en', 'outputFormat':'json'}

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

    vocab_in_split = set()

    # precision
    prec = defaultdict(list)
    for vid, anns in tqdm(ref.items()):
        for seg, ann in anns['segments'].items():
            if len(ann['frame_ind']) == 0 or vid not in pred or seg not in pred[vid]:
                continue # do not penalize if sentence not annotated

            ref_bbox_all = torch.cat((torch.Tensor(ann['process_bnd_box']), \
                torch.Tensor(ann['frame_ind']).unsqueeze(-1)), dim=1) # 5-D coordinates

            idx_in_sent = {}
            for box_idx, cls_lst in enumerate(ann['process_clss']):
                vocab_in_split.update(set(cls_lst))
                for cls_idx, cls in enumerate(cls_lst):
                    idx_in_sent[cls] = idx_in_sent.get(cls, []) + [ann['process_idx'][box_idx][cls_idx]]

            sent_idx = set(itertools.chain.from_iterable(ann['process_idx'])) # index of gt object words
            exclude_obj = {json.loads(nlp.annotate(token.encode('utf-8'), properties=props) \
                )['sentences'][0]['tokens'][0]['lemma']:1 for token_idx, token in enumerate(ann['tokens'] \
                ) if (token_idx not in sent_idx and token != '')}

            for pred_idx, class_name in enumerate(pred[vid][seg]['clss']):
                if class_name in idx_in_sent:
                    gt_idx = min(idx_in_sent[class_name]) # always consider the first match...
                    sel_idx = [idx for idx, i in enumerate(ann['process_idx']) if gt_idx in i]
                    ref_bbox = ref_bbox_all[sel_idx] # select matched boxes
                    assert(ref_bbox.size(0) > 0)

                    pred_bbox = torch.cat((torch.Tensor(pred[vid][seg]['bbox_for_all_frames'][pred_idx])[:,:4], \
                        torch.Tensor(range(10)).unsqueeze(-1)), dim=1)

                    frm_mask = torch.from_numpy(get_frm_mask(pred_bbox[:, 4].numpy(), \
                        ref_bbox[:, 4].numpy()).astype('uint8'))
                    overlap = bbox_overlaps_batch(pred_bbox[:, :5].unsqueeze(0), \
                        ref_bbox[:, :5].unsqueeze(0), frm_mask.unsqueeze(0))
                    prec[class_name].append(1 if torch.max(overlap) > args.iou else 0)
                elif json.loads(nlp.annotate(class_name.encode('utf-8'), properties=props))['sentences'][0]['tokens'][0]['lemma'] in exclude_obj:
                    pass # do not penalize if gt object word not annotated (missed)
                else:
                    if args.mode == 'all':
                        prec[class_name].append(0) # hallucinated object

    # recall
    recall = defaultdict(list)
    for vid, anns in ref.items():
        for seg, ann in anns['segments'].items():
            if len(ann['frame_ind']) == 0:
                # print('no annotation available')
                continue

            ref_bbox_all = torch.cat((torch.Tensor(ann['process_bnd_box']), \
                torch.Tensor(ann['frame_ind']).unsqueeze(-1)), dim=1) # 5-D coordinates
            sent_idx = set(itertools.chain.from_iterable(ann['process_idx'])) # index of gt object words

            for gt_idx in sent_idx:
                sel_idx = [idx for idx, i in enumerate(ann['process_idx']) if gt_idx in i]
                ref_bbox = ref_bbox_all[sel_idx] # select matched boxes
                # Note that despite discouraged, a single word could be annotated across multiple boxes/frames
                assert(ref_bbox.size(0) > 0)

                class_name = ann['process_clss'][sel_idx[0]][ann['process_idx'][sel_idx[0]].index(gt_idx)]
                if vid not in pred:
                    recall[class_name].append(0) # video not grounded
                elif seg not in pred[vid]:
                    recall[class_name].append(0) # segment not grounded
                elif class_name in pred[vid][seg]['clss']:
                    pred_idx = pred[vid][seg]['clss'].index(class_name) # always consider the first match...
                    pred_bbox = torch.cat((torch.Tensor(pred[vid][seg]['bbox_for_all_frames'][pred_idx])[:,:4], \
                        torch.Tensor(range(10)).unsqueeze(-1)), dim=1)

                    frm_mask = torch.from_numpy(get_frm_mask(pred_bbox[:, 4].numpy(), \
                        ref_bbox[:, 4].numpy()).astype('uint8'))
                    overlap = bbox_overlaps_batch(pred_bbox[:, :5].unsqueeze(0), \
                        ref_bbox[:, :5].unsqueeze(0), frm_mask.unsqueeze(0))
                    recall[class_name].append(1 if torch.max(overlap) > args.iou else 0)
                else:
                    if args.mode == 'all':
                        recall[class_name].append(0) # object not grounded

    num_vocab = len(vocab_in_split)
    print('Number of groundable objects in this split: {}'.format(num_vocab))
    print('Number of objects in prec and recall: {}, {}'.format(len(prec), len(recall)))
    prec_accu = np.sum([sum(hm)*1./len(hm) for i,hm in prec.items()])*1./num_vocab
    recall_accu = np.sum([sum(hm)*1./len(hm) for i,hm in recall.items()])*1./num_vocab
    f1 = 2. * prec_accu * recall_accu / (prec_accu + recall_accu)

    print('-' * 80)
    print('The overall precision / recall / F1 are {:.4f} / {:.4f} / {:.4f}'.format(prec_accu, recall_accu, f1))
    print('-' * 80)
    if args.verbose:
        print('Object frequency and grounding accuracy per class (descending by object frequency):')
        accu_per_clss = {}
        for i in vocab_in_split:
            prec_clss = sum(prec[i])*1./len(prec[i]) if i in prec else 0
            recall_clss = sum(recall[i])*1./len(recall[i]) if i in recall else 0
            accu_per_clss[(i, prec_clss, recall_clss)] = (len(prec[i]), len(recall[i]))
        accu_per_clss = sorted(accu_per_clss.items(), key=lambda x:x[1][1], reverse=True)
        for accu in accu_per_clss:
            print('{} ({} / {}): {:.4f} / {:.4f}'.format(accu[0][0], accu[1][0], accu[1][1], accu[0][1], accu[0][2]))

    nlp.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ActivityNet-Entities object grounding evaluation script.')
    parser.add_argument('-s', '--submission', type=str, default='', help='submission grounding result file')
    parser.add_argument('-r', '--reference', type=str, default='data/anet_entities_cleaned_class_thresh50_trainval.json', help='reference file')
    parser.add_argument('--split_file', type=str, default='data/split_ids_anet_entities.json', help='path to the split file')
    parser.add_argument('--split', type=str, nargs='+', default=['validation'], help='which split(s) to evaluate')
    parser.add_argument('--mode', type=str, default='all', help='all | loc, as whether consider lang error')
    parser.add_argument('-iou', type=float, default=0.5, help='the iou threshold for grounding correctness')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
