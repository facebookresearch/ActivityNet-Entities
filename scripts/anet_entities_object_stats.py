# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script to print stats on the object annotation file

import numpy as np
import json
import csv
# import visdom
import sys
from collections import Counter

src_file = sys.argv[1] # 'anet_entities_cleaned_class_thresh50_trainval.json'
dataset_file = sys.argv[2] # 'anet_captions_all_splits.json'
split_file = sys.argv[3] # 'split_ids_anet_entities.json'


if __name__=='__main__':
    with open(src_file) as f:
        data = json.load(f)['annotations']
    
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
    
    box_per_seg = []
    obj_per_box = []
    count_obj = []
    
    for vid, dat in data.items():
        for seg, ann in dat['segments'].items():
            total_box[split_dict[vid]] = total_box.get(split_dict[vid], 0)+len(ann['process_bnd_box'])
            total_dur.append(float(raw_data[vid]['timestamps'][int(seg)][1]-raw_data[vid]['timestamps'][int(seg)][0]))
            seg_splits[split_dict[vid]] = seg_splits.get(split_dict[vid], 0)+1
            box_per_seg.append(len(ann['process_bnd_box']))
            for c in ann['process_clss']:
                obj_per_box.append(len(c))
                count_obj.extend(c)
    
    print('number of annotated video: {}'.format(len(data)))
    print('number of annotated video segments: {}'.format(num_seg))
    print('number of segments in each split: {}'.format(seg_splits))
    print('total duration in hr: {}'.format(np.sum(total_dur)/3600))
    print('total number of phrase (not object) boxes: {}'.format(total_box))
    
    print('box per segment, mean {}, std {}, count {}'.format(np.mean(box_per_seg), np.std(box_per_seg), Counter(box_per_seg)))
    print('object per box, mean {}, std {}, count {}'.format(np.mean(obj_per_box), np.std(obj_per_box), Counter(obj_per_box)))
    
    print('Top 10 object labels: {}'.format(Counter(count_obj).most_common(10)))
    
    """
    # visualization
    vis = visdom.Visdom()
    vis.histogram(X=[i for i in box_per_seg if i < 20],
                  opts={'numbins': 20, 'xtickmax':20, 'xtickmin':0, 'xmax':20, 'xmin':0, 'title':'Distribution of number of boxes per segment', 'xtickfont':{'size':14}, \
                        'ytickfont':{'size':14}, 'xlabel':'Number of boxes', 'ylabel': 'Counts'})
    
    vis.histogram(X=[i for i in obj_per_box if i < 100],
                  opts={'numbins': 100, 'xtickmax':100, 'xtickmin':0, 'xmax':100, 'xmin':0, 'title':'Distribution of number of object labels per box', 'xtickfont':{'size':14}, \
                        'ytickfont':{'size':14}, 'xlabel':'Number of object labels', 'ylabel': 'Counts'})
    """
