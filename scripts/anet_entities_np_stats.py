# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script to print stats on the NP annotation file

import numpy as np
import json
import csv
import sys

src_file = sys.argv[1] # 'anet_entities.json'
dataset_file = sys.argv[2] # 'anet_captions_all_splits.json'
split_file = sys.argv[3] # 'split_ids_anet_entities.json'


if __name__ == '__main__':
    with open(src_file) as f:
        data = json.load(f)['database']
    
    with open(dataset_file) as f:
        raw_data = json.load(f)
    
    split_dict = {}
    with open(split_file) as f:
        split = json.load(f)
        for s,ids in split.items():
            split_dict.update({i:s for i in ids})
    
    num_seg = np.sum([len(dat['segments']) for vid, dat in data.items()])
    
    total_box = {}
    total_dur = []
    seg_splits = {}
    for vid, dat in data.items():
        for seg, ann in dat['segments'].items():
            total_box[split_dict[vid]] = total_box.get(split_dict[vid], 0)+len(ann['objects'])
            total_dur.append(float(raw_data[vid]['timestamps'][int(seg)][1]-raw_data[vid]['timestamps'][int(seg)][0]))
            seg_splits[split_dict[vid]] = seg_splits.get(split_dict[vid], 0)+1
    
    print('number of annotated video: {}'.format(len(data)))
    print('number of annotated video segments: {}'.format(num_seg))
    print('number of segments in each split: {}'.format(seg_splits))
    print('total duration in hr: {}'.format(np.sum(total_dur)/3600))
    print('total number of noun phrase boxes: {}'.format(total_box))
