# Three-Color Spiral Data Classifier

A custom neural network project built from scratch using NumPy to classify non-linearly separable spiral data into three distinct categories.

## Project Overview
This repository features a manual implementation of deep learning components without high-level libraries like Keras. It implements fundamental layers, activation functions, optimizers, and regularization techniques to solve the "Spiral Data" classification problem.

## Core Components
The project includes custom Python classes for all parts of the neural network pipeline:
* **Layers**: 
    * `Layer_Dense`: Implements forward and backward passes for fully connected layers with support for L1 and L2 regularization.
    * `Layer_Dropout`: Implements a dropout mechanism to prevent overfitting by randomly deactivating neurons during training.
* **Activation Functions**: 
    * `Activation_ReLu`: Standard Rectified Linear Unit for hidden layers.
    * `Activation_Softmax`: For multi-class probability output in the final layer.
* **Loss Functions**: 
    * `Loss_CategoricalCrossentropy`: Measures performance for classification tasks.
* **Optimizers**: 
    * `Optimizer_Adam`: Advanced optimizer incorporating momentum and learning rate scaling.
    * `Optimizer_SGD`: Stochastic Gradient Descent with momentum and decay options.

## Dataset
* **Source**: Generated using `nnfs.datasets.spiral_data`.
* **Training Set**: 3,000 samples (1,000 per class) across 3 classes.
* **Test Set**: 300 samples (100 per class).

## Model Architecture
The network is structured as follows:
1.  **Input**: 2 features (X, Y coordinates).
2.  **Dense Layer 1**: 64 neurons with L2 weight and bias regularization (`5e-4`).
3.  **ReLU Activation**.
4.  **Dropout Layer**: 10% dropout rate.
5.  **Dense Layer 2**: 3 neurons (representing the 3 classes).
6.  **Softmax Output**.

## Training Details
* **Optimizer**: Adam.
* **Initial Learning Rate**: 0.05.
* **Learning Rate Decay**: 5e-5 per iteration.
* **Epochs**: 10,000.

## Results
This project was made for learning purposes where results were not of main concern.
Results can be improved via hyperparameter tuning.

## Requirements
* `numpy`
* `nnfs` (Neural Networks from Scratch dataset utility)
* `matplotlib`
* `pandas`

## Usage
1.  Open `three_color_classifier.ipynb` in a Jupyter environment.
2.  The notebook includes sections to toggle between training with or without regularization (by commenting/uncommenting specific layer definitions).
3.  Run all cells to see the epoch-by-epoch training logs and final validation metrics.
