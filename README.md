# Stellar Spectral Classification: Comparing ML Approaches

## Overview

A comparative study of four machine learning approaches applied to stellar spectral classification. The project covers the full pipeline from EDA and preprocessing through to hyperparameter optimisation and final model comparison, deliberately spanning supervised learning, unsupervised learning, and neural network approaches on the same problem.

**Dataset:** Star Type Classification — 240 instances, 6 features (temperature, luminosity, radius, absolute magnitude, star colour, spectral class), 7 target classes (O, B, A, F, G, K, M spectral types)

**Key challenge:** Severe class imbalance — classes G and K are significantly underrepresented. Addressed through data augmentation on the training set with Gaussian noise injection for minority classes.

## Models

| Model | Type | Tuning Method |
|---|---|---|
| Gaussian Naive Bayes | Supervised (probabilistic) | GridSearchCV (var_smoothing) |
| K-Means Clustering | Unsupervised | Manual k selection |
| MLP (Multi-Layer Perceptron) | Neural Network | Optuna (10 trials, K-fold CV) |
| CNN (Convolutional Neural Network) | Neural Network | Optuna (10 trials, K-fold CV) |

## Methodology

**Preprocessing:**
- Star colour standardisation (13 raw values → 4 simplified tint categories)
- Label encoding of categorical features
- Gaussian normalisation of continuous features (StandardScaler)
- Stratified train/test split

**Evaluation:**
- Stratified K-fold cross-validation
- Metrics: accuracy, weighted F1, macro F1, precision, recall
- Confusion matrices for each model

**Hyperparameter search (MLP and CNN via Optuna):**
- Architecture parameters: number of layers, hidden size, activation function, batch normalisation, dropout
- Training parameters: learning rate, batch size, optimiser
- Pruning via MedianPruner to terminate unpromising trials early

## Results

All four models compared on held-out test set. The CNN and Optuna-tuned MLP outperformed both the GNB baseline and K-means. The K-means approach serves as a demonstration of applying unsupervised clustering to a supervised classification problem via majority-label mapping.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── stellar_classification_comparison.ipynb   # Full analysis notebook (85 cells)
```

## Data

The dataset is the [Star Type Classification](https://www.kaggle.com/datasets/deepu1109/star-dataset) dataset available on Kaggle (public domain).

Download `stars_data.csv` from that link and place it in the same directory as the notebook before running. The file is not included in this repository as it must be downloaded directly from Kaggle.

## Running the Notebook

The notebook was developed in Google Colab. To run locally:

```bash
pip install -r requirements.txt
```

Then open `stellar_classification_comparison.ipynb` in Jupyter and ensure `stars_data.csv` is in the same directory.

## Technical Stack

- Python 3.10
- `torch`, `torch.nn` — MLP and CNN implementation
- `scikit-learn` — GNB, K-means, preprocessing, GridSearchCV, cross-validation, metrics
- `optuna` — automated hyperparameter optimisation
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualisation
