# Overview

This repository contains a code to train a facial keypoint detection using pretrained resnet models in **PyTorch**.


# Requirements
Once the repo cloned, please run inside the repo:
```
import os
os.mkdir('data')
os.mkdir('checkpoints')
```
Then move the training csv file to __data__ folder.

## Training

To train the model, please run:
```
python -m src.train 
    --batch_size [BATCH SIZE value] 
    --lr [LEARNING RATE value] 
    --wd [WEIGHT DECAY Value] 
    --train_device [choose cpu or gpu]
    --test_device [choose cpu or gpu]
    --modelname [MODEL: 'resnet18' or 'resnet50']
```
All other parameters can be chosen in ```src/training_parser.py```.
