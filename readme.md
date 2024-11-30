# Disentangled Representation Learning For Diabetic Retinopathy

This project provides a framework for disentangled representation learning, including backbone pretraining, disentanglement training, and performance evaluation.

---

## Table of Contents

- [Disentangled Representation Learning For Diabetic Retinopathy](#disentangled-representation-learning-for-diabetic-retinopathy)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Dataset](#dataset)
  - [Usage](#usage)
    - [Training](#training)
      - [Pretrain Backbone](#pretrain-backbone)
      - [Train the Disentanglement Network](#train-the-disentanglement-network)
    - [Testing](#testing)
      - [Evaluate Disentanglement Performance](#evaluate-disentanglement-performance)
      - [Evaluate Classification Performance](#evaluate-classification-performance)
  - [Requirements](#requirements)
  - [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository implements a disentangled representation learning framework. The workflow includes:  
1. **Pretraining the backbone model**: Train a feature extractor using contrastive learning (e.g., MoCo v2).  
2. **Training the disentanglement network**: Fine-tune to separate features into distinct subspaces.  
3. **Evaluation**: Assess disentanglement and feature effectiveness on downstream tasks.  

---

## Directory Structure

```bash
.  
├── data/                          # Dataset folder
│   ├── class_1/                   # Samples for class 1
│   │   ├── sample1.jpg
│   │   ├── sample2.jpg
│   │   └── ...
│   ├── class_2/                   # Samples for class 2
│   │   ├── sample1.jpg
│   │   ├── sample2.jpg
│   │   └── ...
│   └── ...       
├── scripts/                       # Configuration files
│   ├── mocov2_backbone.yaml       # Config for backbone pretraining
│   ├── mocov2_disentangle.yaml    # Config for disentanglement network
│   └── ...
├── solo/                          # Model definitions
├── utils/                         # Utility scripts
├── main_disentangle.py            # Train the disentanglement network
├── main_pretrain.py               # Pretrain the backbone model
├── main_test.py                   # Evaluate performance on classification tasks
├── main_manipulate.py             # Evaluate disentanglement
├── README.md                      
```

## Dataset
The dataset should follow the structure below, with each class organized into a separate folder:

```bash
data/
├── class_1/  # Class 1
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
├── class_2/  # Class 2
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
└── ...
```
Each subfolder corresponds to a class, and it contains the samples (e.g., images) for that class.


## Usage

### Training

#### Pretrain Backbone
To pretrain the backbone network using contrastive learning (e.g., MoCo v2), run:

```bash
python main_pretrain.py --config-path scripts --config-name mocov2_backbone.yaml
```

#### Train the Disentanglement Network
To train the disentanglement network, run:

```bash
python main_disentangle.py --config-path scripts/ --config-name mocov2_disentangle.yaml
```

### Testing
#### Evaluate Disentanglement Performance
To assess the quality of disentangled features:

```bash
python main_manipulate.py
```

#### Evaluate Classification Performance
To evaluate disentangled features on a classification task:

```bash
python main_test.py
```

## Requirements
This project is based on Lightning and Dali. Our Docker environment is coming soon.

## Acknowledgements
This project is based on [solo-learn](https://github.com/vturrisi/solo-learn)