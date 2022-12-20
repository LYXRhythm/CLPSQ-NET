# CLPSQ-NET
This code is for the paper "CLPSQ-NET"

## Requirements
* pytorch 1.7.1+cu101
* torchvision 0.8.2+cu101
* tqdm 4.50.2
* numpy 1.18.5

## Data Preparation
We provide two public datasets for evaluating algorithm performance [PSQ-8](https://drive.google.com/file/d/1e1ZBQKxu9Jfug6RL60h_VO6a95GHe6jR/view?usp=share_link) and PSQ-1000.

Follow the link above to download the dataset and extract it to the root directory. The following command converts the 'csv' format data in the dataset to the 'npy' format used by the program.
```
python data_processing.py
```

## Train
The training process is divided into two stages. The first stage is self-supervised comparative learning, and the second stage is learning tuning again on the basis of the first training stage.

#### Stage one
```
python train_stage1.py
```

#### Stage two
```
python train_stage2.py
```

## Evaluate 
We also provide the [pretrained](https://drive.google.com/file/d/1KYLeIswRZgVTzffcXCtnrzoT2th1enfF/view?usp=share_link) model for simply test, 
```
python evaluate.py
```

**If this repository helps you，please star it and quote from our paper. Thanks.**