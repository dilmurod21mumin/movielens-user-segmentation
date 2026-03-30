# 🎬 Movie User Taste Segmentation
### Unsupervised Clustering on the MovieLens Dataset

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

This project segments MovieLens users into distinct **taste profiles** based on their movie genre preferences — without any labels. By applying and comparing four unsupervised clustering algorithms, the project identifies the most effective model for powering a personalized movie recommendation engine.

> **Best model:** K-Means (k=5) with a Davies–Bouldin Index of **1.6467**

---

## 📂 Project Structure

```
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── final_project.ipynb       # Main analysis notebook
├── report.md                 # Full analytical report
└── README.md
```

---

## 📊 Dataset

**[MovieLens Small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)** — GroupLens Research, University of Minnesota

| File | Records | Description |
|------|---------|-------------|
| `movies.csv` | 9,742 | Movie titles and genres |
| `ratings.csv` | 100,836 | User–movie ratings (1–5 scale) |
| `tags.csv` | — | User-generated movie tags |
| `links.csv` | — | IMDb / TMDb ID mappings |

> Download the dataset and place the CSV files inside a `data/` folder before running the notebook.

---

## ⚙️ Methodology

### Feature Engineering
- Merged ratings, movies, and tags into a unified dataframe (102,677 rows)
- Built a **user–genre pivot table** (671 users × 951 genre combinations) using mean ratings per user per genre
- Standardized features with `StandardScaler`
- Applied **PCA** for dimensionality reduction (2D/3D visualization & 10-component preprocessing)

### Clustering Algorithms

| Algorithm | Key Config | Clusters Found |
|-----------|-----------|----------------|
| K-Means | k=5, Elbow Method | 5 |
| Mean Shift | bandwidth=28.19 (auto) | 4 |
| DBSCAN | eps=3, min_samples=5, PCA(10) | 3 + 164 noise pts |
| Hierarchical (HAC) | Ward linkage, dendrogram, PCA(10) | 4 |

### Evaluation
Models were compared using the **Davies–Bouldin Index (DBI)** — an internal metric requiring no ground-truth labels. Lower is better.

| Algorithm | DBI Score |
|-----------|-----------|
| **K-Means** ✅ | **1.6467** |
| Mean Shift | 1.7676 |
| Hierarchical Clustering | 1.7676 |
| DBSCAN | 4.0447 |

---

## 🔍 Key Results

- **K-Means (k=5)** achieved the best DBI score, producing 5 compact and well-separated user segments
- **Cluster profiles** revealed distinct taste groups corresponding to genre combinations like Action/Adventure, Comedy/Romance, and Drama
- **DBSCAN underperformed** due to the sparse, high-dimensional nature of the feature space — ~24% of users were classified as noise
- **PCA preprocessing** was essential for density- and distance-based algorithms to function reliably

---

## 🛠️ Tech Stack

- **Python 3.13**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — clustering (KMeans, MeanShift, DBSCAN, AgglomerativeClustering), preprocessing, evaluation
- `scipy` — hierarchical linkage and dendrogram
- `matplotlib` — 2D/3D cluster visualization
- `sklearn.manifold.TSNE` — non-linear dimensionality reduction for visualization

---


## 🔮 Future Work

- Add **Silhouette Score** and **Calinski-Harabasz Index** for multi-metric evaluation
- Enrich features with **TF-IDF tag embeddings**
- Explore **Gaussian Mixture Models (GMM)** for soft cluster assignments
- Apply **matrix factorization** (SVD, NMF) for latent factor extraction
- Fine-tune DBSCAN with grid search over `eps` and `min_samples`

---

*Dataset provided by [GroupLens Research](https://grouplens.org/datasets/movielens/), University of Minnesota.*