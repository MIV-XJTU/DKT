# DKT

The official PyTorch implementation of our CVPR 2023 poster paper:

_DKT: Diverse Knowledge Transfer Transformer for Class Incremental Learning_

GitHub maintainer: Xinyuan Gao 

## Requirement

We use the \
python == 3.9 \
torch == 1.11.0 \
torchvision == 0.12.0 \
timm == 0.5.4 \
continuum == 1.2.3 \

## Accuracy
We provide the accuracy of every phase in different settings in the following table. You can also get them in the logs(We run the official code again, it may be slightly different from the paper). \

|             Variant             |       ARTrack-256       |       ARTrack-384       |      ARTrack-L-384      |
|:-------------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
|          Model Config           | ViT-B, 256^2 resolution | ViT-B, 384^2 resolution | ViT-L, 384^2 resolution |
| GOT-10k (AO / SR 0.5 / SR 0.75) |   73.5 / 82.2 / 70.9    |   75.5 / 84.3 / 74.3    |   78.5 / 87.4 / 77.8    |
|    LaSOT (AUC / Norm P / P)     |   70.4 / 79.5 / 76.6    |   72.6 / 81.7 / 79.1    |   73.1 / 82.2 / 80.3    |
| TrackingNet (AUC / Norm P / P)  |   84.2 / 88.7 / 83.5    |   85.1 / 89.1 / 84.8    |   85.6 / 89.6 / 84.8    |
|  LaSOT_ext (AUC / Norm P / P)   |   46.4 / 56.5 / 52.3    |   51.9 / 62.0 / 58.5    |   52.8 / 62.9 / 59.7    |
|          TNL-2K (AUC)           |          57.5           |          59.8           |          60.3           |
|           NfS30 (AUC)           |          64.3           |          66.8           |          67.9           |
|          UAV123 (AUC)           |          67.7           |          70.5           |          71.2           |


|             CIFAR100 10—10            |      1      |       2      |      3      |    4      |       5      |      6      |    7      |       8      |      9      |       10      |      AVG      |
|:-------------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
|                     | 94.2 | 86.95 | 83.0 | 77.53 | 74.12 | 74.05 | 74.12 | 74.05 | 70.53 | 67.9 | 65.12 | 63.45 | 75.69 |




## Notice
If you want to run our experiment on different numbers of GPUs, you should set the Batch_size * GPUs == 512. For example, one GPU, the Batch size 512 and two GPUs, the Batch size 256 (CIFAR-100 and ImageNet100). If you want to change it, please try to change the hyperparameters.



## Acknowledgement

Our code is heavily based on the great codebase of [Dytox](https://github.com/arthurdouillard/dytox), thanks for its wonderful code frame.

Also, a part of our code is inspired by the [CSCCT](https://github.com/ashok-arjun/CSCCT), thanks for its code.

## Trainer

You can use the following command to run the code like the Dytox: 

```bash
bash train.sh 0,1 
    --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_DKT.yaml 
    --name DKT 
    --data-path MY_PATH_TO_DATASET 
    --output-basedir PATH_TO_SAVE_CHECKPOINTS 
    --memory-size 2000
```


## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@InProceedings{Gao_2023_CVPR, 
    author    = {Gao, Xinyuan and He, Yuhang and Dong, Songlin and Cheng, Jie and Wei, Xing and Gong, Yihong}, 
    title     = {DKT: Diverse Knowledge Transfer Transformer for Class Incremental Learning}, 
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    month     = {June}, 
    year      = {2023}, 
    pages     = {24236-24245} 
}
```
