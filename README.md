# ActivityNet Entities Dataset and Challenge 2020

### [ActivityNet Entities Object Localization (Grounding) Challenge](http://activity-net.org/challenges/2020/tasks/guest_anet_eol.html) joins the official [ActivityNet Challenge](http://activity-net.org/challenges/2020/challenge.html) as a guest task this year! See [below](#aeol) for details.

This repo hosts the dataset and evaluation scripts used in our paper [Grounded Video Description](https://arxiv.org/abs/1812.06587) (GVD). **We also released the source code of GVD in this [repo](https://github.com/facebookresearch/grounded-video-description).**

ActivityNet-Entities, is based on the video description dataset [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) and augments it with 158k bounding box annotations, each grounding a noun phrase (NP). Here we release the complete set of NP-based annotations as well as the pre-processed object-based annotations.

<img src='demo/dataset_teaser.png' alt="dataset teaser" width="80%"/>


## <a name="aeol"></a>ActivityNet Entities Object Localization Challenge 2020
**The winners will be announced at ActivityNet Challenge at CVPR 2020 (June 14th). Prizes are to be decided.**

For general challenge descriptions, refer to the official ANet-Entities Challenge [page](http://activity-net.org/challenges/2020/tasks/guest_anet_eol.html).

The annotations on the training and validation set is in `data/anet_entities_cleaned_class_thresh50_trainval.json`. `data/anet_entities_skeleton.txt` specifies the JSON file structure on both reference files and submission files.

For Sub-task I, we will use the **public** test set. The skeleton of the file is in `data/anet_entities_cleaned_class_thresh50_test_skeleton.json`, where we intentionally leave out the bounding box annotation for official evaluation purposes.

For Sub-task II, we will use the **hidden** test set (available around April 6th). No video descripitions nor bounding boxes for this split will be provided.

The evaluation script used in our evaluation server will be identical to `scripts/eval_grd_anet_entities.py`. Read the following section for more details.


## General Dataset Info
### Data
We have the following dataset files under the `data` directory:

- `anet_entities_skeleton.txt`: Specify the expected structure of the JSON annotation files.
- `anet_entities_cleaned_class_thresh50_trainval.json`: Pre-processed dataset file with object class and bounding box annotations. For training and validation splits only.
- `anet_entities_cleaned_class_thresh50_test_skeleton.json`: Object class annotation for the **public** testing split. This file is for evaluation server purpose and the bounding box annotation is not given. See below for more details.
- `anet_entities_trainval.json`: The raw dataset file with noun phrase and bounding box annotations. We only release the training and the validation splits for now.
- `split_ids_anet_entities.json`: Video IDs included in the training/validation/**public** testing splits.

Note: Both the raw dataset file and the pre-processed dataset file contain all the 12469 videos in our training and validation split (training + one half of the validation split as in ActivityNet Captions, which is based on [ActivityNet 1.3](http://activity-net.org/download.html)). This includes 626 videos without box annotations.

### Evaluation
Under the `scripts` directory, we include:

- `eval_grd_anet_entities.py`: The evaluation script for object grounding on GT/generated captions. [PyTorch (tested on 1.1 and 1.3)](https://pytorch.org/get-started/locally/), [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) and the [Python wrapper](https://github.com/Lynten/stanford-corenlp) are required.
- `attr_prep_tag_NP.py`: The preprocessing scripts to obtain the NP/object annotation files.
- `anet_entities_np_stats.py`, `anet_entities_object_stats.py`: The scripts that print the dataset stats.

To evaluate attention/grounding output based upon GT sentences (metrics in paper: Attn., Grd.), run:
```
python scripts/eval_grd_anet_entities.py -s YOUR_SUBMISSION_FILE.JSON --eval_mode GT
```

To evaluate attention (same for grounding) output based upon generated sentences (metrics in paper: F1<sub>all</sub>, F1<sub>loc</sub>), similarly run:
```
python scripts/eval_grd_anet_entities.py -s YOUR_SUBMISSION_FILE.JSON --eval_mode gen --loc_mode $loc_mode
```
where setting `loc_mode=all` to perform evaluation on all object words while setting `loc_mode=loc` to perform evaluation only on correctly-predicted object words.

We provide a Codalab evaluation [server](https://competitions.codalab.org/competitions/20537) for the validation set. The evaluation on the **public** test set is available now until ECCV deadline and will be temporarily closed afterwards. The server will reopen on both test sets around April 13th. Please follow the example in `data/anet_entities_skeleton.txt` to format your submission file.

### FAQs
1. How are the 10 frames sampled from each video clip (event)?
⋅⋅⋅We divide each clip evenly into 10 segments and sample the middle frame of each segment. We have clarified this in the skeleton [file](https://github.com/facebookresearch/ActivityNet-Entities/blob/master/data/anet_entities_skeleton.txt#L13).

2. How can I sample the frames by myself and extract feature?
⋅⋅⋅First, you may want to check if the region feature and RGB/motion frame-wise feature we [provided](https://github.com/facebookresearch/grounded-video-description#data-preparation) meet your requirement.⋅⋅
⋅⋅⋅If not, you can first download the ActivityNet videos using this [web crawler](https://github.com/activitynet/ActivityNet/blob/master/Crawler/fetch_activitynet_videos.sh) or contact the dataset [owners](http://activity-net.org/people.html) for help. An incorrect video encoding format would result in a wrong frame resolution and aspect ratio, and therefore a mismatch in the annotation. Hence, make sure you download the videos in the [best mp4 format](https://github.com/activitynet/ActivityNet/blob/master/Crawler/run_crosscheck.py#L32).⋅⋅
⋅⋅⋅Once you have the videos, you can use `ffmpeg` to extract the frames. We provide an example command [here](https://github.com/facebookresearch/ActivityNet-Entities/issues/1#issuecomment-529065386).

### Reference
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
