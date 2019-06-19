# ActivityNet Entities dataset
This repo hosts the dataset and evaluation scripts used in our paper [Grounded Video Description](https://arxiv.org/abs/1812.06587) (GVD). **We also released the source code of GVD in this [repo](https://github.com/facebookresearch/grounded-video-description).**

ActivityNet-Entities, is based on the video description dataset [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) and augments it with 158k bounding box annotations, each grounding a noun phrase (NP). Here we release the complete set of NP-based annotations as well as the pre-processed object-based annotations.

<img src='demo/dataset_teaser.png' alt="dataset teaser" width="80%"/>

### Data
We have the following dataset files under the `data` directory:

- `anet_entities_trainval.json`: The raw dataset file with noun phrase and bounding box annotations. We only release the training and the validation splits for now.
- `anet_entities_cleaned_class_thresh50_trainval.json`: Pre-processed dataset file with object class and bounding box annotations. For training and validation splits only.
- `anet_entities_cleaned_class_thresh50_test_skeleton.json`: Object class annotation for the testing split. This file is for evaluation server purpose and the bounding box annotation is not given. See below for more details.
- `anet_entities_skeleton.txt`: Specify the expected structure of the JSON annotation files.
- `split_ids_anet_entities.json`: Video IDs included in the training/validation/testing splits.

Note: Both the raw dataset file and the pre-processed dataset file contain all the 12469 videos in our training and validation split (training + one half of the validation split as in ActivityNet Captions, which is based on [ActivityNet 1.3](http://activity-net.org/download.html)). This includes 626 videos without box annotations.

### Evaluation
Under the `scripts` directory, we include:

- `attr_prep_tag_NP.py`: The preprocessing scripts to obtain the NP/object annotation files.
- `anet_entities_np_stats.py`, `anet_entities_object_stats.py`: The scripts that print the dataset stats.
- `eval_grd_anet_entities.py`: The evaluation script for object grounding on GT/generated captions. [PyTorch](https://pytorch.org/get-started/locally/), [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) and the [Python wrapper](https://github.com/Lynten/stanford-corenlp) are required.

To evaluate attention/grounding output based upon GT sentences (metrics in paper: Attn., Grd.), run:
```
python scripts/eval_grd_anet_entities.py -s YOUR_SUBMISSION_FILE.JSON --eval_mode GT
```

To evaluate attention (same for grounding) output based upon generated sentences (metrics in paper: F1<sub>all</sub>, F1<sub>loc</sub>), similarly run:
```
python scripts/eval_grd_anet_entities.py -s YOUR_SUBMISSION_FILE.JSON --eval_mode gen --loc_mode $loc_mode
```
where setting `loc_mode=all` to perform evaluation on all object words while setting `loc_mode=loc` to perform evaluation only on correctly-predicted object words.

We provide a Codalab evaluation [server](https://competitions.codalab.org/competitions/22835) for the test set. Please follow the example in `data/anet_entities_skeleton.txt` to format your submission file.


### Others
Please contact <luozhou@umich.edu> if you have any trouble running the code. Please cite the following paper if you use the dataset.
```
@inproceedings{zhou2019grounded,
  title={Grounded Video Description},
  author={Zhou, Luowei and Kalantidis, Yannis and Chen, Xinlei and Corso, Jason J and Rohrbach, Marcus},
  booktitle={CVPR},
  year={2019}
}
```
### License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

The noun phrases in these annotations are based on [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/), which are linked to videos in [ActivityNet 1.3](http://activity-net.org/download.html) 
