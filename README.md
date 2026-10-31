<div align="center">

# Employee Attrition Classification

**A case study in the accuracy paradox: three of five models score about 80% accuracy and never find a single employee who leaves.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/imbalance%20fix-not%20done%20yet-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## At a glance

| | |
|---|---|
| **Task** | Predict `Attrition` (Yes/No) for 10,000 employees from 25 HR features |
| **Models** | KNN, Gaussian Naive Bayes, Decision Tree, Random Forest, Keras neural network |
| **Finding** | Naive Bayes, Random Forest and the neural network reach 0.795 to 0.796 accuracy with **0.0 precision and recall**, because they always predict "no attrition" |
| **Only useful model** | The Decision Tree: lowest accuracy (0.672), but the only one with real recall (0.257) |
| **Not done** | Retraining with class weights or SMOTE. Needs the CSV, which is not in this repo ([details](#next-steps-not-done-yet)) |

## Contents

[The result in one picture](#the-result-in-one-picture) · [Data](#data) · [Pipeline](#pipeline) · [Results](#results-as-logged-by-the-notebook) · [Why accuracy misleads](#an-instructive-result-reported-honestly) · [Next steps](#next-steps-not-done-yet)

## The result in one picture

| Model | Accuracy | Recall (finds leavers) |
|---|---|---|
| Naive Bayes | `████████████████░░░░` 0.796 | `░░░░░░░░░░░░░░░░░░░░` 0.000 |
| Random Forest | `████████████████░░░░` 0.796 | `░░░░░░░░░░░░░░░░░░░░` 0.000 |
| Deep Learning | `████████████████░░░░` 0.795 | `░░░░░░░░░░░░░░░░░░░░` 0.000 |
| KNN | `███████████████░░░░░` 0.765 | `░░░░░░░░░░░░░░░░░░░░` 0.039 |
| Decision Tree | `█████████████░░░░░░░` 0.672 | `█████░░░░░░░░░░░░░░░` 0.257 |

High accuracy next to empty recall is the signature of a model that has learned to say "no" to everyone, because about 80% of employees stay.

## Data

The notebook reads `employee_attrition_dataset_10000.csv`: 10,000 employees and 26 columns (age, gender, marital status, department, job role and level, monthly income, hourly rate, tenure and promotion history, satisfaction and performance scores, overtime, project count, weekly hours, absenteeism, distance from home, number of previous companies, and the `Attrition` label). The file came with the course and is **not included in this repository**, so the notebook cannot be re-run from a fresh clone until you place that CSV next to it. A first, smaller dataset (`employee_attrition_dataset.csv`) is referenced only in commented-out code.

To re-run it: `pip install pandas numpy scikit-learn matplotlib seaborn tensorflow jupyter`, put the CSV in the same folder, and run all cells. The outputs saved in the notebook are the ones reported below.

## Pipeline

1. **Dataset analysis & preparation:** shape, dtypes, categorical value counts, duplicate removal, missing-value imputation (median/mode), type conversion, grouped aggregation (mean salary by department/role), equal-width binning.
2. **Exploration:** histograms, scatterplots, StandardScaler normalization, PCA to 2 components, correlation heatmap.
3. **Modeling:** five approaches trained and evaluated on the same train/test split — KNN, Gaussian Naive Bayes, Decision Tree, Random Forest, and a deep learning model (Keras).

## Results (as logged by the notebook)

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| KNN | 0.765 | 0.170 | 0.039 |
| Naive Bayes | 0.796 | 0.0 | 0.0 |
| Decision Tree | 0.672 | 0.229 | 0.257 |
| Random Forest | 0.796 | 0.0 | 0.0 |
| Deep Learning | 0.795 | 0.0 | 0.0 |

## An instructive result, reported honestly

Three of the five models (Naive Bayes, Random Forest, Deep Learning) land at ~79.5–79.6% accuracy with **0.0 precision and recall** — their confusion matrices show they're predicting the majority class ("no attrition") for nearly every record, never actually identifying an attrition case. Since roughly 80% of the dataset is "no attrition," a model that always predicts "no" scores ~80% accuracy while being completely useless at the task it was built for.

This is left in deliberately rather than cherry-picking the headline accuracy number: it's a clean, real example of the accuracy paradox on an imbalanced dataset, and exactly why precision/recall (or techniques like class weighting, SMOTE, or choosing a cost-sensitive threshold) matter more than accuracy alone for imbalanced classification — the Decision Tree model, despite its *lower* accuracy (0.672), is the only one actually identifying some true attrition cases (recall 0.257).

## Next steps (not done yet)

The honest follow-up to the result above is to retrain with the imbalance handled: class weights or SMOTE on the training split only, a lower decision threshold chosen from the precision-recall curve, and PR-AUC or F1 as the headline metric instead of accuracy. That needs the dataset, so it is left as a to-do rather than claimed.

## Repository layout

```text
.
├── employee_attrition_classification.ipynb   # EDA, preprocessing, five models
└── LICENSE
```

*Two versions of this assignment existed in the original coursework archive; this is the more complete one, which extends the shared EDA and preprocessing with the full model comparison.*

## Skills demonstrated

The full classical ML workflow across several algorithm families, and, more importantly, reading evaluation metrics correctly instead of reporting accuracy alone on an imbalanced problem.

## License

[MIT](LICENSE) © 2026 Abdelrahman Shaheen
