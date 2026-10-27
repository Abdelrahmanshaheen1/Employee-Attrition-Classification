# Employee Attrition Classification

Predicts employee attrition (Yes/No) from an HR dataset, covering the full workflow from exploratory data analysis through comparing four classification approaches: KNN, Naive Bayes, Decision Tree, Random Forest, plus a deep learning model.

*(Two versions of this assignment existed in the original coursework archive — this is the more complete one, which extends the shared EDA/preprocessing with the full model comparison.)*

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

## Skills demonstrated

Full classical ML workflow across multiple algorithm families, and — importantly — correctly reading evaluation metrics rather than reporting accuracy in isolation on an imbalanced classification problem.
