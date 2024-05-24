# Brain Tumor Classification using MRI Images

This repository contains the code and dataset for classifying brain tumors into four classes using MRI images. The four classes are:

1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)


## Overview

This project aims to classify brain tumors from MRI images into four categories using a convolutional neural network (CNN). The dataset contains labeled MRI scans for each category. The model is trained to accurately distinguish between these classes, providing a useful tool for medical diagnostics.

## Dataset

The dataset is organized into four directories, one for each class:
- `glioma/`
- `meningioma/`
- `pituitary_tumor/`
- `no_tumor/`

Each directory contains MRI images in `.jpg` format.

### Download

You can download the dataset from [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri).

### Structure

The dataset structure is as follows:

## Model

The model is a convolutional neural network (CNN) built using TensorFlow/Keras. The architecture includes several convolutional layers followed by max-pooling and dropout layers to prevent overfitting.

### Architecture

- Convolutional Layers
- Max-Pooling Layers
- Dropout Layers
- Fully Connected Layers
- Output Layer with Softmax Activation

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install the requirements using:
```bash
pip install -r requirements.txt

