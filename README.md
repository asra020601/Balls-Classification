
# Multi-Class Sports Balls Image Classification



## Overview

This project focuses on building a machine learning model to classify images of different sports balls into their respective categories. The project utilizes deep learning techniques and convolutional neural networks (CNNs) for accurate image classification.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)


## Dataset

The dataset used for this project consists of various sports balls images categorized into classes like soccer ball, basketball, tennis ball, and baseball. The dataset contains a total of 7328 images for training and 1841 images for testing.

## Installation

To set up and run the project, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Multi-Class-Sports-Balls-Image-Classification.git
   cd Multi-Class-Sports-Balls-Image-Classification
   ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your images for testing in the `test_data` folder.
2. Use the pre-trained model weights or train your model (see [Training](#training)).
3. Run the classification script:

   ```bash
   python classify.py
   ```

## Model Architecture

The classification model architecture consists of a Convolutional Neural Network (CNN) with multiple convolutional and pooling layers, followed by fully connected layers. The model architecture is defined in `model.py`.

## Training

To train the model from scratch:

1. Prepare your dataset by organizing images into class-specific folders within the `train_data` directory.
2. Run the training script:

   ```bash
   python train.py
   ```

3. Trained model weights will be saved in the `saved_models` folder.

## Evaluation

The model's performance is evaluated on the test dataset using accuracy and confusion matrix metrics. The evaluation script can be run using:

```bash
python evaluate.py
```

## Results

Our trained model achieved an accuracy of 79% on the test dataset. The confusion matrix and classification report can be found in the `results` directory.

## Future Enhancements

- Implement data augmentation techniques to improve model generalization.
- Explore transfer learning using pre-trained models like ResNet or VGG.
- Deploy the model as a web application for real-time image classification.



---
