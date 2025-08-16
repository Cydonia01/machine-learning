# Convolutional Neural Network (CNN) for Image Classification

This project implements a Convolutional Neural Network (CNN) for binary image classification (e.g., cats vs. dogs) using TensorFlow/Keras. The model is trained on images organized in directory structures for training and testing.

---

## Overview

- **Task**: Binary image classification (e.g., cat vs. dog)
- **Dataset**: Get the dataset [here](https://sds.courses/cm/dl-az-completeDatasets)
- **Framework**: TensorFlow (Keras API)
- **Input**: 64x64 RGB images
- **Output**: Predicted class (cat or dog)

---

## Steps

1. Load and preprocess images using `image_dataset_from_directory`
2. Data augmentation: rescaling, random flip, zoom, and rotation
3. Build a CNN with two convolutional layers and dense layers
4. Train the model with binary cross-entropy loss and Adam optimizer
5. Evaluate on the test set
6. Make a single prediction on a new image

---

## Usage

```bash
python cnn.py
```

---

## Requirements

- tensorflow
- numpy
- keras

Install with:

```bash
pip install tensorflow numpy keras
```

---

## Directory Structure

```
dataset/
    training_set/
        cats/
        dogs/
    test_set/
        cats/
        dogs/
    single_prediction/
        cat_or_dog.jpg
```

---

## Output

- Prints class indices and the predicted class for a sample image
- Shows training and validation accuracy per epoch

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)
- [Image Classification with Keras](https://keras.io/examples/vision/image_classification_from_scratch/)
