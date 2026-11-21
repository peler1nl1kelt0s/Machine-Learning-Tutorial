# Classification — executed notebook index

🇹🇷 [Türkçe](README.tr.md)

All 10 chapters have executed notebooks with code, explanations, tables and real outputs. Open an `.ipynb` file on GitHub to see the results without running it. Each notebook contains its complete implementation; no hidden helper module is required.

**Theory:** [English](Classification.md) · [Türkçe](Classification.tr.md). Notebook explanations include Turkish guidance alongside English; the code is not duplicated by language.

| # | Executed notebook | Contents | Saved result |
|---|---|---|---|
| 1 | [Logistic Regression](1%29%20%20Logistic%20Regression%20Classification/Logistic%20Regression.ipynb) | NumPy gradients, stable loss, sklearn, confusion matrix | Test accuracy 97.37% |
| 2 | [K-Nearest Neighbours](2%29%20K-Nearest%20Neighbour%20%28KNN%29%20Classification/K-Nearest%20Neighbour%20%28KNN%29.ipynb) | Train-only scaling, CV choice of K, inspect a neighbour vote | Test accuracy 95.61% |
| 3 | [Support Vector Machine](3%29%20Support%20Vector%20Machine%20%28SVM%29%20Classification/Support%20Vector%20Machine%20%28SVM%29.ipynb) | Linear/RBF kernel selection, support vectors, margin plot | Test accuracy 97.37% |
| 4 | [Naive Bayes](4%29%20Naive%20Bayes%20Classification/Naive%20Bayes.ipynb) | Gaussian likelihoods, priors, manual log posterior | Test accuracy 93.86% |
| 5 | [Decision Tree](5%29%20Decision%20Tree%20Classification/Decision%20Tree.ipynb) | CV depth/leaf selection, entropy, root information gain | Test accuracy 92.98% |
| 6 | [Random Forest](6%29%20Random%20Forest%20Classification/Random%20Forest.ipynb) | Bootstrap, OOB, tree probability averaging, importance | Test accuracy 96.49% |
| 7 | [Evaluation & algorithm comparison](7%29%20Model%20Evaluation/Model%20Evaluation.ipynb) | Six models on shared folds, OOF threshold, confusion/ROC/PR | CV-selected SVM: test 98.25% |
| 8 | [K-means clustering](8%29%20K-Means%20Clustering/K-Means.ipynb) | Lloyd from scratch, K-means++, elbow, silhouette, limitations | WCSS 150,669.43; silhouette 0.7444 |
| 9 | [Hierarchical clustering](9%29%20Hierarchical%20Clustering/Hierarchical%20Clustering.ipynb) | Hand example, Ward SSE/height, dendrograms, linkage comparison | 3 clusters; SciPy/sklearn ARI = 1 |
| 10 | [NLP: sentiment classification](10%29%20Natural%20Language%20Processing/NLP%20Sentiment%20Classification.ipynb) | Parsing, BoW, manual TF-IDF, CSR, six pipelines, interpretation | Test accuracy 83.89% |

## Setup and re-execution

Use Python **3.10–3.12**; the executed artifacts were verified with Python 3.10. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Classification/requirements.txt
python Classification/run_notebooks.py
```

On Windows, activate with `.venv\Scripts\activate`.

Run one chapter or a subset:

```bash
python Classification/run_notebooks.py --chapter 10
python Classification/run_notebooks.py --chapter 8 --chapter 9
```

The runner starts each notebook in a **fresh kernel using the current Python interpreter**, executes every cell in order and saves outputs into the same `.ipynb`. It stops on an error and leaves that notebook’s previous saved version intact. It also collects measured summaries in [notebook_results.json](notebook_results.json). Re-execution can change notebook outputs in Git.

In an interactive notebook editor, select this virtual environment’s kernel and use **Restart Kernel → Run All**. Notebooks do not depend on variables from one another. No data download or API key is needed.

## How to read the measurements

- **1–7:** Local Wisconsin data; one shared stratified 80/20 split, `random_state=42`. Scalers fit training rows only and fit separately inside each CV training fold. Test rows do not choose hyperparameters. Chapters 2, 3 and 5 tune settings with CV; chapter 7 compares six fixed configurations. Its SVM therefore need not match chapter 3’s result. A higher CV score does not guarantee a higher score on a single held-out split.

- **8–9:** 3,000 unlabelled synthetic points from the same seed. WCSS and silhouette describe geometry; they are not accuracy. Chapter 9’s ARI compares two implementations’ partitions, not predictions against true labels.

- **10:** [UCI sentiment data](datasets/nlp_sentiment/README.md); 3,000 raw records become 2,979 unique records, split into 2,383 training and 596 test rows. Word+character TF-IDF + logistic regression, selected by training CV macro-F1, reaches **83.89%** test accuracy. This does not reproduce the earlier reported 48% on a different dataset/target.

Older reference figures and worked tables in theory chapters 1–7 may use different splits; they describe the experiments in their captions. The current shared-protocol results are saved in the notebooks above. Error bars show variation across folds or shuffles, not population confidence intervals.

## Other Python files

- [run_notebooks.py](run_notebooks.py): executes all notebooks or selected chapters.
- [nlp_text_classification.py](nlp_text_classification.py): repeats chapter 10’s sentiment experiment from the command line.
- [generate_clustering_figures.py](generate_clustering_figures.py) and [generate_nlp_figures.py](generate_nlp_figures.py): regenerate the bilingual theory figures for chapters 8–10 in `images/`. Run `nlp_text_classification.py` before generating the NLP figures.
