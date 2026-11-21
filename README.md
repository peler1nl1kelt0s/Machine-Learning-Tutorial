# Machine Learning

🇹🇷 [Türkçe](README.tr.md)

A structured collection of study notes and hands-on implementations covering foundational machine learning algorithms. Each topic is organized in its own folder with a dedicated markdown file explaining the intuition, the math behind it (with step-by-step derivations and figures built from the datasets in this repo), and scikit-learn code snippets.

---

## Contents

### 📈 [Regression](./Regression/Regression.md)
Predicting continuous numerical values from input features.

| # | Algorithm | Key Idea |
|---|-----------|----------|
| 1 | Simple Linear Regression | Fit a straight line between one input and one output |
| 2 | Multiple Linear Regression | Extend linear regression to multiple input features |
| 3 | Polynomial Regression | Model curved relationships by adding powers of features |
| 4 | Decision Tree Regression | Recursively split the feature space; predict a constant per leaf |
| 5 | Random Forest Regression | Ensemble of decision trees averaged for more stable predictions |
| 6 | Model Evaluation | R², MSE, SSR, SST — measuring how well the model fits |

### 🔷 [Classification](./Classification/Classification.md)
Predicting which category a data point belongs to.

| # | Algorithm | Key Idea |
|---|-----------|----------|
| 1 | Logistic Regression | Binary classification via sigmoid; implemented from scratch + sklearn |
| 2 | K-Nearest Neighbours (KNN) | Classify by majority vote among the K closest training samples |
| 3 | Support Vector Machine (SVM) | Find the maximum-margin hyperplane separating two classes |
| 4 | Naive Bayes | Probabilistic classifier using Bayes' theorem with feature independence assumption |
| 5 | Decision Tree Classification | Split on the feature and threshold that reduce entropy (or Gini) the most |
| 6 | Random Forest Classification | Many randomized trees on bootstrap sub-samples, combined by majority vote |
| 7 | Confusion Matrix & Metrics | Accuracy, precision, recall, F1, ROC/AUC — measuring which mistakes a classifier makes |
| 8 | K-means Clustering | Assign unlabelled points to centroids and reduce within-cluster squared error |
| 9 | Hierarchical Clustering | Merge groups by linkage and cut a dendrogram |
| 10 | NLP: Sentiment Classification | Turn text into sparse BoW/TF-IDF features and compare models with CV |
| 11 | PCA (Principal Component Analysis) | Compress high-dimensional data onto orthogonal directions of maximum variance |
| 12 | Model Selection (K-Fold & Grid Search CV) | Tune hyperparameters cleanly with Stratified K-Fold CV inside a Pipeline |
| 13 | Recommendation Systems | Content-based, cosine collaborative filtering, and SVD latent factors |

**[Classification: 10 executed notebooks, code, figures and setup](Classification/README.md).** The clustering chapters cover unsupervised learning. Notebooks include English/Turkish guidance and saved real outputs.

---

## Tools & Libraries

- **Language:** Python 3
- **Main Library:** [scikit-learn](https://scikit-learn.org/)
- **Utilities:** NumPy, Matplotlib, Pandas

---

> Each algorithm folder contains a Jupyter notebook with step-by-step code and inline explanations, alongside the theory reference in the folder's `.md` file.
