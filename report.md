# User Taste Segmentation Using Unsupervised Learning — MovieLens Dataset

---

## 1. Objective

The goal of this analysis is to **segment MovieLens users by their movie genre preferences** using unsupervised clustering algorithms. By grouping users with similar tastes, we can power a personalized recommendation engine that does not rely on explicit labels or supervised signals.

Specifically, the analysis aims to:

- Build a user–genre preference matrix from ratings data.
- Apply and compare four unsupervised clustering models: K-Means, Mean Shift, DBSCAN, and Hierarchical Agglomerative Clustering (HAC).
- Identify the best-performing model using an internal evaluation metric (Davies–Bouldin Index).
- Derive actionable insights about user taste groups.

---

## 2. Data Description

The dataset used is the **[MovieLens Small dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)** provided by GroupLens Research, University of Minnesota. It contains real movie ratings collected from the MovieLens website.

### Files and Structure

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `movies.csv` | 9,742 | 3 | `movieId`, `title`, `genres` |
| `ratings.csv` | 100,836 | 4 | `userId`, `movieId`, `rating`, `timestamp` |
| `tags.csv` | — | 4 | `userId`, `movieId`, `tag`, `timestamp` |
| `links.csv` | — | 3 | `movieId`, `imdbId`, `tmdbId` |

### Feature Engineering

To prepare data for clustering:

1. **Merging** — Ratings were joined with movies and tags into a single dataframe of **102,677 rows × 8 columns**.
2. **User–Genre Pivot Table** — A matrix was built with `userId` as rows and `genres` as columns, using the mean rating per genre per user. Unrated genres were filled with 0, producing a **671 users × 951 genre combinations** feature matrix.
3. **Standardization** — `StandardScaler` was applied so each genre feature contributes equally regardless of rating scale differences.
4. **Dimensionality Reduction** — PCA was used both for visualization (2D, 3D) and as a preprocessing step before DBSCAN and HAC (10 components to reduce noise). No null or NaN values were present in the engineered feature matrix.

---

## 3. Model Comparison

Four unsupervised clustering algorithms were implemented and evaluated.

### 3.1 K-Means (k = 5)

K-Means partitions users into k clusters by minimizing intra-cluster variance. The number of clusters k was initially set to 5 and validated with the **Elbow Method** (k = 2 to 10), which confirmed k = 5 as the point of diminishing inertia reduction.

- **Clusters found:** 5
- **Visualization:** 2D PCA, 3D PCA, t-SNE — clusters showed visible separation.
- **Cluster profiles** (mean genre ratings per cluster) were analyzed to characterize each user group.

### 3.2 Mean Shift

Mean Shift is a non-parametric algorithm that finds cluster centers by shifting data points toward regions of higher density. It requires no manual specification of k.

- **Bandwidth estimated automatically:** 28.19 (quantile = 0.2, n_samples = 500)
- **Clusters found:** 4
- Suitable for datasets where the number of clusters is unknown in advance.

### 3.3 DBSCAN (Density-Based Spatial Clustering)

DBSCAN groups points that are closely packed and marks low-density points as noise. It was applied on **PCA-reduced data (10 components)** to reduce the curse of dimensionality.

- **Parameters:** `eps = 3`, `min_samples = 5`
- **Clusters found:** 3
- **Noise points:** 164 users (≈24%) were labeled as outliers (`-1`), indicating that a significant portion of users did not fit into any dense cluster.
- A k-distance graph was plotted to guide `eps` selection.

### 3.4 Hierarchical Agglomerative Clustering (HAC)

HAC builds a tree of clusters (dendrogram) bottom-up using Ward linkage, which minimizes within-cluster variance at each merge step. Applied on PCA-reduced data (10 components).

- **Clusters found:** 4 (determined from dendrogram, truncated at level 5)
- Ward linkage produced compact, balanced clusters similar to K-Means but without requiring an initial centroid guess.

### Summary Comparison Table

| Algorithm | Clusters Found | Requires k? | Handles Noise? | DBI Score |
|-----------|---------------|-------------|----------------|-----------|
| **K-Means** | 5 | Yes (Elbow) | No | **1.6467** ✅ |
| Mean Shift | 4 | No (auto) | No | 1.7676 |
| HAC | 4 | Yes (dendrogram) | No | 1.7676 |
| DBSCAN | 3 | No | Yes (164 pts) | 4.0447 |

---

## 4. Evaluation

### Davies–Bouldin Index (DBI)

The **Davies–Bouldin Index** is an internal clustering evaluation metric for unlabeled data. It measures the ratio of within-cluster scatter to between-cluster separation — a **lower score means better, more compact and well-separated clusters**.

| Algorithm | DBI Score | Rank |
|-----------|-----------|------|
| **K-Means** | **1.6467** | 🥇 1st |
| Mean Shift | 1.7676 | 🥈 2nd |
| Hierarchical Clustering | 1.7676 | 🥈 2nd |
| DBSCAN | 4.0447 | 🥉 4th |

---

## 5. Key Findings and Insights

- **K-Means (k=5) produced the best cluster quality**, achieving the lowest DBI of 1.65. The five clusters correspond to five meaningfully distinct user taste profiles based on genre preferences.

- **Mean Shift and HAC performed comparably** (DBI = 1.77), both converging on 4 clusters. Their similar scores suggest a natural 4–5 cluster structure in the data.

- **DBSCAN was poorly suited to this dataset**. The high-dimensional genre-rating space with many zero values creates a sparse, uniform density distribution that DBSCAN cannot handle effectively. Nearly a quarter of users (164/671) were classified as noise, and the resulting DBI of 4.04 indicates poor cluster cohesion.

- **PCA dimensionality reduction was critical** for DBSCAN and HAC. Without it, distance metrics in 951-dimensional space become unreliable due to the curse of dimensionality.

- **The Elbow Method confirmed k=5** as optimal for K-Means, showing a clear inflection point in the inertia curve and validating the initial assumption.

- **Cluster profile analysis** revealed that different clusters correspond to users who predominantly enjoy specific genre combinations (e.g., Action/Adventure, Comedy/Romance, Drama), enabling targeted content recommendations per segment.

---

## 6. Limitations and Next Steps

### Limitations

- **Sparse feature matrix:** Most users have not rated movies in most genre combinations, resulting in a feature matrix dominated by zeros. This sparsity affects distance-based algorithms — especially DBSCAN — and may cause clusters to reflect rating activity levels rather than true taste differences.

- **Genre-only features:** The analysis relies solely on raw genre labels. More nuanced user preferences — such as favorite directors, actors, or narrative themes — are not captured.

- **Static snapshot:** The dataset represents a fixed time window. User tastes evolve over time, and the model does not account for temporal drift in preferences.

- **DBI as sole metric:** DBI is a good internal metric, but it favors convex, similarly-sized clusters, which advantages K-Means by design. Using additional metrics like Silhouette Score or Calinski-Harabasz Index would give a more balanced evaluation.

- **DBSCAN parameter sensitivity:** The `eps = 3` and `min_samples = 5` values were chosen manually. A systematic grid search over these parameters may yield better density-based clustering results.

### Next Steps

- **Add Silhouette Score and Calinski-Harabasz Index** alongside DBI for a more robust multi-metric model comparison.
- **Incorporate tag-based features** using TF-IDF on user tags to enrich the feature space beyond genre labels.
- **Apply Gaussian Mixture Models (GMM)** for soft/probabilistic cluster assignments, allowing users to partially belong to multiple taste segments.
- **Experiment with matrix factorization** (SVD, NMF) to extract latent user–movie factors as features instead of raw genre averages.
- **Tune DBSCAN** more rigorously using the k-distance graph and a grid search over `eps` and `min_samples`.
- **Deploy K-Means** as a real-time user segmentation service to feed cluster labels into a recommendation engine, with periodic retraining as new ratings arrive.

---

*Dataset source: [MovieLens Small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) — GroupLens Research, University of Minnesota.*