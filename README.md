# KNN Logistic Regression Classifier
![KNNLOGREG](https://github.com/Torin99/kNN-Logistic-Regression-Classifier/assets/87572723/75d23224-a543-40c0-a27d-8c35ebb2183d)

## Overview:

This repository contains code and analysis for a machine learning project comparing the performance of two classification models, k-nearest Neighbor (KNN) and Logistic Regression, on four distinct datasets. The project also includes experiments to find the best hyperparameters for the models.

## Datasets:

### 1. Ionosphere Dataset:
- **Description:** Radar measurements collected in Goose Bay, Labrador, focusing on radar returns from free electrons in the ionosphere.
- **Preprocessing:** Features converted to floating-point numbers, labels mapped to binary values (0 for "bad" radars, 1 for "good" radars).

### 2. Adult Dataset:
- **Description:** Income dataset predicting whether individuals’ income exceeds $50K/yr based on census data.
- **Preprocessing:** Numerical data converted to floats, labels mapped to binary values (0 for income under $50K, 1 for income over $50K).

### 3. Rice Dataset:
- **Description:** Dataset of rice grain images for two species - Cammeo and Osmancik.
- **Preprocessing:** Labels mapped to binary values (0 for Cammeo, 1 for Osmancik).

### 4. Mushroom Dataset:
- **Description:** Dataset of 23 species of mushrooms classified as poisonous or edible.
- **Preprocessing:** Labels mapped to binary values (1 for edible mushrooms, 0 for poisonous mushrooms).

## Models:

### 1. Logistic Regression:
- **Implementation:** Full-batch gradient descent optimization.
- **Features:** Utilizes learning rates to train the model.
- **Experimentation:** Tested with different learning rates using 5-fold cross-validation.

### 2. K-Nearest Neighbor (KNN):
- **Implementation:** Implemented using custom functions.
- **Features:** Explores different values of K to find the best hyperparameter.
- **Experimentation:** Evaluated using 5-fold cross-validation.

## Results and Analysis:

### Comparison of Models:

- **Accuracy comparison of KNN and Logistic Regression on all datasets:**

| Dataset     | KNN Accuracy | Logistic Regression Accuracy |
|-------------|--------------|------------------------------|
| Ionosphere  | 82.29%       | 77.43%                       |
| Adult       | 76.71%       | 75.92%                       |
| Rice        | 97.22%       | 73.33%                       |
| Mushroom    | 84.20%       | 67.14%                       |

### Best K-value for KNN:

- **Analysis of the best K-value for KNN on each dataset:**

  - Ionosphere Dataset: Best K-value = 3
  - Adult Dataset: Best K-value = 3
  - Rice Dataset: Best K-value = 3
  - Mushroom Dataset: Best K-value = 3

### Learning Rates for Logistic Regression:

- **Experimentation with different learning rates for Logistic Regression and their impact on accuracy:**

| Learning Rate | Average Accuracy | Average Number of Iterations |
|---------------|------------------|------------------------------|
| 5             | 86.0%            | 572.4                        |
| 3             | 84.87%           | 175.4                        |
| 2             | 84.87%           | 238.6                        |
| 1             | 84.29%           | 344.0                        |
| 0.1           | 83.15%           | 597.0                        |
| 0.01          | 77.43%           | 791.0                        |
| 0.001         | 67.71%           | 499.8                        |
| 0.0001        | 67.14%           | 2.0                          |
| 0.00001       | 67.14%           | 2.0                          |

### Performance as Dataset Size Increases:

- **Evaluation of the accuracy of both models as the size of the dataset varies:**

  - *Note: Performance as dataset size increases was not explicitly mentioned in the provided report.*

This section provides a summary of the accuracy comparison between KNN and Logistic Regression on various datasets, analysis of the best K-value for KNN, experimentation with different learning rates for Logistic Regression, and an evaluation of model performance with varying dataset sizes.
