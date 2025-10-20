## ReshapeIT

<div align=center><img src="figs/ReshapeIT.png"></div>

[**_ReShapeIT: Reliable Shape Interaction with Implicit Template for Medical Anatomy Reconstruction_**]()

> By Minghui Zhang, and Yun Gu
>> Institute of Medical Robotics, Shanghai Jiao Tong University


## Introduction
We present the Reliable Shape Interaction with Implicit Template (ReShapeIT) network, which represents anatomical structures using continuous implicit fields rather than discrete voxel grids. The approach combines a category-specific implicit template field with a deformation field to encode anatomical shapes from training shapes. In addition, a Template Interaction Module (TIM) is designed to refine test cases by aligning learned template shapes with instance-specific latent codes.

## Environment
Detailed environment configurations are listed in the file ```environment.yaml```.


## Pretrained Weights
Pretrained model weights are available in the folder ```checkpoint_models```.

## Usage
To train the model, run:

```
python train.py
```

To fine-tune the model with the specified configuration files, run:

```
python finetune.py
```


