# Stroke Risk Prediction Challenge - Deep Learning Classification Models

## Problem Statement

This project assists a public health organization in identifying individuals most at risk of experiencing a stroke using machine learning and deep learning techniques. The goal is to build accurate binary classification models that predict whether a patient will experience a stroke or not based on their health indicators and demographic information.

### Business Context

The health organization has expressed concern about predicting all stroke cases correctly:
- **False Negatives (High Cost):** Missing a stroke prediction can be life-threatening and represents a critical failure
- **False Positives (Lower Cost):** While false alarms waste hospital resources, they are less critical than missing actual cases

This asymmetric cost structure drives the model evaluation strategy, prioritizing **recall** over precision.

## Dataset Overview

**Source:** Stroke Risk Dataset from Kaggle

**Dataset Size:** 5,110 patient records with 11 features

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `id` | Numeric | Unique identifier (removed from model) |
| `gender` | Categorical | Patient's gender (Male, Female, Other) |
| `age` | Numeric | Age of the patient |
| `hypertension` | Binary | Whether patient has hypertension (0/1) |
| `heart_disease` | Binary | History of heart disease (0/1) |
| `ever_married` | Categorical | Marital status (Yes/No) |
| `work_type` | Categorical | Type of employment |
| `Residence_type` | Categorical | Area of residence (Urban/Rural) |
| `avg_glucose_level` | Numeric | Average blood glucose level |
| `bmi` | Numeric | Body Mass Index |
| `smoking_status` | Categorical | Smoking status |
| `stroke` | Binary | **Target:** Whether patient experienced stroke (0/1) |

### Data Characteristics

- **Total Records:** 5,110
- **Missing Values:** 201 missing BMI values (handled via mean imputation)
- **Class Imbalance:** 95% negative class (4,861), 5% positive class (249)
- **Duplicates:** None detected

### Feature Correlations with Stroke

Top correlated features with stroke outcome:
1. Age (r = 0.245) - Strongest predictor
2. Heart Disease (r = 0.135)
3. Average Glucose Level (r = 0.132)
4. Hypertension (r = 0.128)
5. Ever Married (r = 0.108)

## Key Assumptions

### Data Processing Strategy

1. **ID Removal:** Removed the `id` column as high index values could mislead the model
2. **Missing Value Handling:** Filled missing BMI values with the column mean (28.89)
3. **Categorical Encoding:** Applied label encoding to all categorical features
4. **Class Imbalance:** Applied SMOTENC (Synthetic Minority Over-sampling Technique) on training data only
   - Original training: 4,088 samples
   - Resampled training: 7,802 samples (balanced class distribution)

### Evaluation Philosophy

- **Primary Metric:** Recall (sensitivity) - to minimize false negatives
- **Secondary Metrics:** Precision, F1-score, and ROC-AUC
- **Decision Threshold:** All models use 0.49 as classification threshold (slightly below 0.5)

## Data Preparation

### Steps Followed

1. **Loaded and explored** the dataset structure
2. **Removed** the `id` column (high values affect model learning)
3. **Separated** features (X) and target (Y)
4. **Train-Test Split:** 80% training, 20% testing (random_state=42)
5. **SMOTENC Resampling:** Applied to training set only (preserves test set imbalance)
6. **Categorical Indexing:** Identified 7 categorical features for SMOTENC
7. **Data Standardization:** Applied StandardScaler to all features during preprocessing

### Distribution Insights

- **Age:** Relatively even distribution (0.08 - 82 years)
- **BMI & Glucose:** Positive skewed distributions (addressed via standardization)
- **Stroke Cases:** Higher glucose levels show bimodal patterns for stroke vs. non-stroke groups

## Model Development

This project implements four progressively sophisticated models, starting from a baseline and advancing to hyperparameter-tuned deep learning:

### Model 1: Logistic Regression Baseline

**Architecture:** Simple logistic regression classifier

**Key Parameters:**
- Algorithm: L-BFGS solver
- Max iterations: 100 (default)
- Training data: SMOTENC resampled (7,802 samples)

**Results:**
```
Accuracy:    72.6%
Recall:      70.97%  (71% of stroke cases detected)
Precision:   14.4%
F1-Score:    23.9%

Confusion Matrix:
                Predicted 0    Predicted 1
Actual 0            698            262
Actual 1             18             44
```

**Analysis:** High recall but extremely low precision. The model detects most strokes but generates many false positives. Suitable for sensitivity analysis but not clinical deployment due to resource waste.

**Limitations:** Convergence warning due to small training set size relative to feature space; suggests need for scaling.


### Model 2: Simple Neural Network

**Architecture:**
```
Input (10 features)
  ↓
Dense(16, relu)
  ↓
Dense(8, relu)
  ↓
Dense(1, sigmoid) → Output
```

**Key Parameters:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 120
- Batch Size: 32
- Validation Split: 10%
- Total Parameters: 965

**Results:**
```
Accuracy:    73.3%
Recall:      64.52%  (65% of stroke cases detected)
Precision:   13.7%
F1-Score:    22.7%

Confusion Matrix:
                Predicted 0    Predicted 1
Actual 0            709            251
Actual 1             22             40
```

**Analysis:** Slightly higher accuracy but lower recall than Model 1. The neural network's added complexity doesn't compensate for class imbalance without regularization.

**Observations:** Model shows overfitting tendency; regularization techniques needed.

### Model 3: Regularized Neural Network with Dropout

**Architecture:**
```
Input (10 features)
  ↓
Dense(16, relu) + L2(0.1)
  ↓
Dropout(0.2)
  ↓
Dense(8, relu) + L2(0.1)
  ↓
Dropout(0.2)
  ↓
Dense(1, sigmoid) → Output
```

**Key Parameters:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 150
- Batch Size: 16
- Dropout Rate: 0.2 (after each hidden layer)
- L2 Regularization: 0.1
- Total Parameters: 965

**Results:**
```
Accuracy:    67.3%
Recall:      88.71%  ⭐ (89% of stroke cases detected!)
Precision:   14.4%
F1-Score:    24.8%

Confusion Matrix:
                Predicted 0    Predicted 1
Actual 0            633            327
Actual 1              7             55
```

**Analysis:** **Recommended model.** Despite lower overall accuracy, this model excels at the critical task: detecting actual stroke cases. Only 7 false negatives (missed strokes) compared to 18-22 in previous models. The regularization successfully prevents overfitting while maintaining high sensitivity to the minority class.

**Clinical Value:** From a public health perspective, this model's 89% recall rate is preferable—it catches almost all at-risk patients, with false positives being manageable through clinical review.

### Model 4: Hyperparameter-Tuned Neural Network (GridSearchCV)

**Architecture:** Same as Model 3 (Dropout + L2 Regularization)

**GridSearchCV Parameters Tested:**
```
L2 Regularization: [0.1]
Learning Rate: [0.0001, 0.001]
Batch Size: [16]
Epochs: [150]
Cross-Validation: 3-fold
Scoring Metric (refit): Recall
```

**Best Parameters Found:**
- L2 Value: 0.1
- Learning Rate: 0.001
- Batch Size: 16
- Epochs: 150

**Results:**
```
Accuracy:    68.2%
Recall:      88.71%  ⭐ (89% of stroke cases detected)
Precision:   14.7%
F1-Score:    25.3%

Confusion Matrix:
                Predicted 0    Predicted 1
Actual 0            642            318
Actual 1              7             55
```

**Analysis:** GridSearchCV optimization yields marginal improvement over Model 3. The 0.001 learning rate with 0.1 L2 regularization proves to be the optimal configuration. Same high recall (89%) with slightly better specificity (fewer false positives: 318 vs 327).

**Key Insight:** The hyperparameter tuning converges on conservative learning, prioritizing stable minority class detection.

## Results Comparison

### Performance Metrics Summary

| Metric | Model 1 | Model 2 | Model 3 | Model 4 |
|--------|---------|---------|---------|---------|
| **Accuracy** | 72.6% | 73.3% | 67.3% | 68.2% |
| **Recall (Class 1)** | 70.97% | 64.52% | **88.71%** | **88.71%** |
| **Precision (Class 1)** | 14.4% | 13.7% | 14.4% | 14.7% |
| **F1-Score** | 23.9% | 22.7% | 24.8% | 25.3% |
| **False Negatives** | 18 | 22 | **7** | **7** |
| **False Positives** | 262 | 251 | 327 | 318 |

### Key Findings

1. **Recall Trade-off:** Models 3 & 4 sacrifice overall accuracy (67-68%) to achieve superior recall (89%). This is intentional and appropriate for stroke prediction.

2. **False Negative Reduction:** The jump from 18-22 false negatives (Models 1-2) to 7 false negatives (Models 3-4) is clinically significant—potentially saving lives.

3. **Precision Limitation:** All models show low precision (~14%), indicating many false positives. This reflects the challenge of extreme class imbalance even with SMOTENC.

4. **Hyperparameter Impact:** GridSearchCV provides minimal gains, suggesting Model 3's design is near-optimal for this problem.

### Recommendation

**Model 4 (Regularized Neural Network with Dropout)** is recommended for production deployment because:
- Highest recall for the minority class (89%)
- Minimal false negatives (only 7 missed strokes)
- Robust generalization via L2 regularization and dropout

## Project Structure

```
deepLearning_stroke_prediction/
├── README.md                                    # This file
├── healthcare-dataset-stroke-data.csv          # Original dataset
└── Stroke_Risk_Prediction_Challenge_Module_4.ipynb  # Main notebook

---

## Future Improvements

1. **Threshold Optimization:** Experiment with different decision thresholds (currently 0.49)
2. **Feature Selection:** Use permutation importance to reduce dimensionality
3. **Deep Learning:** Experiment with additional layers and activation functions

---

## References

- Dataset: [Kaggle - Healthcare Dataset - Stroke Data](https://www.kaggle.com/)
- SMOTENC: [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- Keras/TensorFlow: [Official Documentation](https://keras.io/)
- Medical Context: Stroke prediction importance in public health

---

## License

This project is submitted as part of the ZAKA Machine Learning Track. All rights reserved.

---

## Contact

**Author:** Alaa Almouiz F. Moh.  
**ID:** S2026_176  
**Organization:** ZAKA ©

For questions or contributions, please refer to the GitHub repository.

---

**Last Updated:** February 2026  
**Status:** Complete - Ready for Deployment (Model 3 Recommended)
