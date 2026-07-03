# ECG Heartbeat CNN Classifier

A machine learning project for ECG heartbeat classification using a one-dimensional convolutional neural network.

The workflow covers data preprocessing, class imbalance handling, model tuning, stratified cross-validation, final evaluation, and analysis of precision-recall trade-offs in imbalanced medical signal data.

## Project Overview

The model classifies ECG heartbeat signals into five heartbeat categories. Each heartbeat is represented as a one-dimensional signal, making a 1D CNN suitable for learning local waveform patterns.

The project focuses on the practical challenges of imbalanced classification, where high overall accuracy is not enough to show that minority classes are being handled properly.

## Main Features

- ECG signal preprocessing
- Train/test separation
- Feature scaling
- One-hot encoded class labels
- Class weighting for severe class imbalance
- 1D CNN model architecture
- Hyperparameter tuning
- Stratified cross-validation
- Confusion matrix evaluation
- Per-class precision, recall, and F1-score analysis
- ROC-AUC evaluation
- Discussion of minority-class performance

## Model Approach

The final model uses a 1D convolutional neural network with convolutional blocks, batch normalization, dropout, dense layers, and a softmax output layer for multiclass classification.

Class weights are used to reduce the effect of the dominant normal-heartbeat class and improve learning for underrepresented classes.

## Evaluation Focus

The project evaluates the model using:

- accuracy
- confusion matrices
- per-class precision
- per-class recall
- F1-scores
- macro ROC-AUC
- cross-validation performance

The analysis places more weight on class-specific behaviour than raw accuracy because the dataset is highly imbalanced.

## Repository Notes

The notebook contains the full modelling workflow from preprocessing to final evaluation. The project is designed to show the full experimental process, including model development decisions and validation strategy.
