# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script to pre-process the raw annotation output to NP/object annotation files

import os
import sys
import json
import numpy as np
from collections import Counter, defaultdict
from stanfordcorenlp import StanfordCoreNLP

data_prefix = sys.argv[1] # the dataset directory, e.g., '/private/home/luoweizhou/subsystem/BottomUpAttnVid/data/anet/'
freq_thresh = int(sys.argv[2]) # 50

src_file = data_prefix+'anet_bb.json' # the raw annotation output from the annotation tool

target_np_file = data_prefix+'anet_entities.json'
target_file = data_prefix+'anet_entities_cleaned_class_thresh'+str(freq_thresh)+'.json'

attr_to_video_file = data_prefix+'attr_to_video.txt'
split_file = data_prefix+'split_ids_anet_entities.json'
train_cap_file = '/private/home/luoweizhou/subsystem/BottomUpAttnVid/data/anet/raw_annotation_file/train.json'
val_cap_file = '/private/home/luoweizhou/subsystem/BottomUpAttnVid/data/anet/raw_annotation_file/val_1.json'

np.random.seed(123) # reproducible


def define_split(database):

    with open(train_cap_file) as f:
        train_ids = json.load(f).keys()

    with open(val_cap_file) as f:
        valtest_ids = json.load(f).keys()

    val_split = np.random.rand(len(valtest_ids))>=0.5 # split a half as the test split
    val_ids = [valtest_ids[i] for i,j in enumerate(val_split) if j]
    test_ids = [valtest_ids[i] for i,j in enumerate(val_split) if ~j]

    vid_ids = set(database.keys())
    train_ann_ids = vid_ids.intersection(set(train_ids))
    val_ann_ids = vid_ids.intersection(set(val_ids))
    test_ann_ids = vid_ids.intersection(set(test_ids))

    print('All data - total: {}, train split: {}, val split: {}, test split: {}'.format(len(train_ids+val_ids+test_ids), len(train_ids), len(val_ids), len(test_ids)))
    print('Annotated data - total: {}, train split: {}, val split: {}, and test split: {}'.format(
        len(vid_ids), len(train_ann_ids), len(val_ann_ids), len(test_ann_ids)))

    return [train_ids, val_ids, test_ids]


def extract_attr(database, splits):

    split_dict = {}
    for split in splits:
        split_dict.update({s:s for s in split})
    print('Object classes defined on {} videos, freq threshold is {}'.format(len(split_dict), freq_thresh))

    attr_all = [] # all the attributes

    for vid_id, vid in database.items():
        if split_dict.get(vid_id, -1) != -1:
            for seg_id, seg in vid['segments'].items():
                for obj in seg['objects']:
                    assert(len(obj['frame_ind']) == 1)
                    for box_id, box in obj['frame_ind'].items():
                        tmp = []
                        attr_lst = []
                        sorted_attr = sorted(box['attributes'], key=lambda x:x[0]) # the attributes are unordered
    
                        for ind, attr in enumerate(sorted_attr):
                            assert(attr[0] >= 0)
                            if len(tmp) == 0:
                                tmp.append(attr[1].lower()) # convert to lowercase
                            else:
                                if attr[0] == (sorted_attr[ind-1][0]+1):
                                    tmp.append(attr[1].lower())
                                else:
                                    attr_lst.append(tmp)
                                    tmp = [attr[1].lower()]
                        if len(tmp) > 0: # the last one
                            attr_lst.append(tmp)
    
                        # exclude empty box (no attribute)
                        # crowd boxes are ok for now
                        if len(attr_lst) == 0: # or box['crowds'] == 1
                            pass
                            # print('empty attribute at video {}, segment {}, box {}'.format(vid_id, seg_id, box_id))
                        else:
                            attr_all.extend([' '.join(i) for i in attr_lst])
    return attr_all


def prep_all(database, database_cap, obj_cls_lst, w2l, nlp):

    w2d = {}
    for ind, obj in enumerate(obj_cls_lst):
        w2d[obj] = ind

    avg_box = [] # number of boxes per segment
    avg_attr = [] # number of attributes per box
    attr_all = [] # all the attributes
    crowd_all = [] # all the crowd labels

    attr_dict = defaultdict(list)
    with open(attr_to_video_file) as f:
        for line in f.readlines():
            line_split = line.split(',')
            attr_id = line_split[0]
            vid_name = line_split[-1]
            attr = ','.join(line_split[1:-1])
            vid_id, seg_id = vid_name.strip().split('_segment_')
            attr_dict[(vid_id, str(int(seg_id)))].append([int(attr_id), attr])

    print('Number of segments with attributes: {}'.format(len(attr_dict)))

    vid_seg_dict = {}
    for vid_id, vid in database.items():
        for seg_id, _ in vid['segments'].items():
            vid_seg_dict[(vid_id, seg_id)] = vid_seg_dict.get((vid_id, seg_id), 0) + 1

    new_database = {}
    new_database_np = {}
    seg_counter = 0
    for vid_id, cap in database_cap.items():
        new_database_np[vid_id] = {'segments':{}}
        new_seg = {}
        for cap_id in range(len(cap['sentences'])):
            new_obj_lst = defaultdict(list)
            seg_id = str(cap_id)
            new_database_np[vid_id]['segments'][seg_id] = {'objects':[]}
            if vid_seg_dict.get((vid_id, seg_id), 0) == 0:
                new_obj_lst['tokens'] = nlp.word_tokenize(cap['sentences'][cap_id].encode('utf-8')) # sentences not in ANet-BB
            else:
                vid = database[vid_id]
                seg = vid['segments'][seg_id]

                # preprocess attributes
                attr_sent = sorted(attr_dict[(vid_id, seg_id)], key=lambda x:x[0])
                start_ind = attr_sent[0][0]

                # legacy token issues from our annotation tool
                for ind, tup in enumerate(attr_sent):
                    if attr_sent[ind][1] == '\\,':
                        attr_sent[ind][1] = ','
                    # elif attr_sent[ind][1] == '':
                    #     attr_sent[ind][1] = 'dummy'
                    # elif attr_sent[ind][1] in  ('.', '!', '?'):
                    #     attr_sent[ind][1] = ','

                new_obj_lst['tokens'] = [i[1] for i in attr_sent] # all the word tokens

                for obj in seg['objects']:
                    assert(len(obj['frame_ind']) == 1)

                    np_ann = {}

                    box_id = obj['frame_ind'].keys()[0]
                    box = obj['frame_ind'].values()[0]

                    np_ann['frame_ind'] = int(box_id)
                    np_ann.update(box)

                    if len(box['attributes']) > 0: # just in case the attribute is empty, though it should not be...
                        tmp = []
                        tmp_ind = []
                        tmp_obj = []
                        attr_lst = []
                        attr_ind_lst = []
                        tmp_np_ind = []
                        np_lst = []
                        sorted_attr = sorted(box['attributes'], key=lambda x:x[0]) # the attributes are unordered
                        sorted_attr = [(x[0]-start_ind, x[1]) for x in sorted_attr] # index relative to the sent
    
                        for ind, attr in enumerate(sorted_attr):
                            assert(attr[0] >= 0)
                            attr_w = attr[1].lower()
                            if len(tmp) == 0:
                                tmp.append(attr_w) # convert to lowercase
                                tmp_np_ind.append(attr[0])
                                if w2l.get(attr_w, -1) != -1:
                                    attr_l = w2l[attr_w]
                                    if w2d.get(attr_l, -1) != -1:
                                        tmp_obj.append(attr_l)
                                        tmp_ind.append(attr[0])
                            else:
                                if attr[0] == (sorted_attr[ind-1][0]+1):
                                    tmp.append(attr_w)
                                    tmp_np_ind.append(attr[0])
                                    if w2l.get(attr_w, -1) != -1:
                                        attr_l = w2l[attr_w]
                                        if w2d.get(attr_l, -1) != -1:
                                            tmp_obj.append(attr_l)
                                            tmp_ind.append(attr[0])
                                else:
                                    np_lst.append([' '.join(tmp), tmp_np_ind])
                                    if len(tmp_obj) >= 1:
                                        attr_lst.append(tmp_obj[-1]) # the last noun is usually the head noun
                                        attr_ind_lst.append(tmp_ind[-1])

                                    tmp = [attr_w]
                                    tmp_np_ind = [attr[0]]
                                    if w2l.get(attr_w, -1) != -1:
                                        attr_l = w2l[attr_w]
                                        if w2d.get(attr_l, -1) != -1:
                                            tmp_obj = [attr_l]
                                            tmp_ind = [attr[0]]
                                        else:
                                            tmp_obj = []
                                            tmp_ind = []
                                    else:
                                        tmp_obj = []
                                        tmp_ind = []

                        if len(tmp) > 0: # the last one
                            np_lst.append([' '.join(tmp), tmp_np_ind])
                            if len(tmp_obj) >= 1:
                                attr_lst.append(tmp_obj[-1]) # the last noun is usually the head noun
                                attr_ind_lst.append(tmp_ind[-1])

                        # if len(np_lst) == 0:
                        #     print('no nps! {} - {}'.format(vid_id, seg_id))
                        assert(len(np_lst) > 0)

                        np_ann['noun_phrases'] = np_lst
                        np_ann.pop('attributes', None)
                        new_database_np[vid_id]['segments'][seg_id]['objects'].append(np_ann)
    
                        # exclude empty box (no attribute)
                        # crowd boxes are ok for now
                        if len(attr_lst) == 0: # or box['crowds'] == 1
                            pass
                            # print('empty attribute at video {}, segment {}, box {}'.format(vid_id, seg_id, box_id))
                        else:
                            new_obj_lst['process_bnd_box'].append([box['xtl'], box['ytl'], box['xbr'], box['ybr']])
                            new_obj_lst['frame_ind'].append(int(box_id))
                            new_obj_lst['crowds'].append(box['crowds'])
                            new_obj_lst['process_clss'].append(attr_lst)
                            new_obj_lst['process_idx'].append(attr_ind_lst)
                            avg_attr.append(len(attr_lst))
                            attr_all.extend([' '.join(i) for i in attr_lst])
                            crowd_all.append(box['crowds'])
    
            avg_box.append(len(new_obj_lst['frame_ind'])) # cound be 0
            if len(new_obj_lst['frame_ind']) == 0:
                new_obj_lst['process_bnd_box'] = []
                new_obj_lst['frame_ind'] = [] # all empty
                new_obj_lst['crowds'] = []
                new_obj_lst['process_clss'] = []
                new_obj_lst['process_idx'] = []
            seg_counter += 1
            new_seg[seg_id] = new_obj_lst

        # new_database[vid_id] = {'rwidth':vid['rwidth'], 'rheight':vid['rheight'], 'segments':new_seg}
        new_database[vid_id] = {'segments':new_seg}

    # quick stats
    print('Number of videos: {} (including empty ones)'.format(len(new_database)))
    print('Number of segments: {}'.format(seg_counter))
    print('Average number of valid segments per video: {}'.format(np.mean([len(vid['segments']) for vid_id, vid in new_database.items()])))
    print('Average number of box per segment: {} and frequency: {}'.format(np.mean(avg_box), Counter(avg_box)))

    print('Average number of attributes per box: {} and frequency: {} (for valid box only)'.format(np.mean(avg_attr), Counter(avg_attr)))
    crowd_freq = Counter(crowd_all)
    print('Percentage of crowds: {} (for valid box only)'.format(crowd_freq[1]*1./(crowd_freq[1]+crowd_freq[0])))

    return new_database, new_database_np


def freq_obj_list(attr_all, nlp, props):
    # generate a list of object classes
    num_nn_per_attr = []
    anet_obj_cls = []
    nn_wo_noun = [] # noun phrases that contain no nouns
    w2lemma = defaultdict(list)

    for i, v in enumerate(attr_all):
        if i%10000 == 0:
            print(i)
        out = json.loads(nlp.annotate(v.encode('utf-8'), properties=props))
        assert(out['sentences'] > 0)
        counter = 0 
        for token in out['sentences'][0]['tokens']:
            if ('NN' in token['pos']) or ('PRP' in token['pos']):
                lemma_w = token['lemma']
                anet_obj_cls.append(lemma_w)
                w2lemma[token['word']].append(lemma_w)
                counter += 1
        num_nn_per_attr.append(counter)
        if counter == 0:
            nn_wo_noun.append(v)
    
    top_nn_wo_noun = Counter(nn_wo_noun)
    print('Frequency of NPs w/o nouns:')
    print(top_nn_wo_noun.most_common(10))
    
    print('Frequency of number of nouns per attribute:')
    print(Counter(num_nn_per_attr))
    
    top_obj_cls = Counter(anet_obj_cls)
    print('Top 10 objects:', top_obj_cls.most_common(20))
    
    obj_cls_lst = []
    for w,freq in top_obj_cls.items():
        if freq >= freq_thresh:
            obj_cls_lst.append(w.encode('ascii'))

    w2l = {}
    for w, l in w2lemma.items():
        # manually correct some machine mistakes
        spec_w2l = {'outfits':'outfit', 'mariachi':'mariachi', 'barrios':'barrio', 'mans':'man', 'bags':'bag', 'aerobics':'aerobic', 'motobikes':'motobike', 'graffiti':'graffiti', 'semi':'semi', 'los':'los', 'tutus':'tutu'}
        if spec_w2l.get(w, -1) != -1: # one special case...
            w2l[w] = spec_w2l[w]
            print('Ambiguous lemma for: {}'.format(w))
        else:
            assert(len(set(l)) == 1)
            w2l[w] = list(set(l))[0]
    print('Number of words derived from lemma visual words {}'.format(len(w2l)))

    return obj_cls_lst, w2l


if __name__ == "__main__":
    nlp = StanfordCoreNLP('/private/home/luoweizhou/subsystem/BottomUpAttn/tools/stanford-corenlp-full-2018-02-27')
    props={'annotators': 'ssplit, tokenize, lemma','pipelineLanguage':'en', 'outputFormat':'json'}

    # load anet captions
    with open(train_cap_file) as f:
        database_cap = json.load(f)
    with open(val_cap_file) as f:
        database_cap.update(json.load(f))
    print('Number of videos in ActivityNet Captions (train+val): {}'.format(len(database_cap)))

    # load raw annotation output anet bb
    with open(src_file) as f:
        database = json.load(f)['database']
    print('Number of videos in ActivityNet-BB (train+val): {}'.format(len(database)))

    if os.path.isfile(split_file):
        with open(split_file) as f:
            all_splits = json.load(f)
            splits = [all_splits['training'], all_splits['validation'], all_splits['testing']]
    else:
        raise 'Cannot find the split file! Uncomment this if you want to create a new split.'
        splits = define_split(database)
        all_splits = {'training':splits[0], 'validation':splits[1], 'testing':splits[2]}
        with open(split_file, 'w') as f:
            json.dump(all_splits, f)

    attr_all = extract_attr(database, splits[:2]) # define object classes on train/val data

    obj_cls_lst, w2l = freq_obj_list(attr_all, nlp, props)

    new_database, new_database_np = prep_all(database, database_cap, obj_cls_lst, w2l, nlp)

    # write raw annotation file
    new_database_np = {'database':new_database_np}
    with open(target_np_file, 'w') as f:
        json.dump(new_database_np, f)

    # write pre-processed annotation file
    new_database = {'vocab':obj_cls_lst, 'annotations':new_database}
    with open(target_file, 'w') as f:
        json.dump(new_database, f)

    # with open(class_file, 'w') as f:
    #     f.write('\n'.join(obj_cls_lst))

    nlp.close() # Do not forget to close the server!
