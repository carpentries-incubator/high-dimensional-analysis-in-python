---
title: "Addressing challenges in high-dimensional clustering"  
teaching: "20"  
exercises: "10"  
keypoints: 
- "Exploring techniques to mitigate the challenges of high-dimensional clustering."
- "Applying dimensionality reduction techniques."
- "Using specialized clustering algorithms."
- "Visualizing clustering results."

objectives:
- "Apply dimensionality reduction techniques to high-dimensional data."
- "Implement specialized clustering algorithms for high-dimensional data."
- "Evaluate the impact of these techniques on clustering performance."
- "Visualize clustering results."

questions:
- "How can dimensionality reduction help in high-dimensional clustering?"
- "What are specialized clustering algorithms for high-dimensional data?"
- "How can we visualize high-dimensional data?"
- "What insights can we gain from visualizing clusters?"

---

# Addressing Challenges in High-Dimensional Clustering

## Dimensionality Reduction Techniques

### Principal Component Analysis (PCA)

PCA reduces the dimensionality of the data by transforming it into a new set of variables that are linear combinations of the original variables. It aims to capture the maximum variance with the fewest number of components.

- **Strengths:**
  - Reduces dimensionality while preserving variance.
  - Simplifies visualization.
  - Computationally efficient.

- **Weaknesses:**
  - Linear method, may not capture complex relationships.
  - Can be sensitive to scaling and outliers.

- **When to use:**
  - When you need to reduce dimensionality for visualization.
  - When you have a large number of features and need to reduce them for computational efficiency.

```python
from sklearn.decomposition import PCA

# Applying PCA
pca = PCA(n_components=10)  # Reduce to 10 dimensions
df_pca = pca.fit_transform(df_scaled)
```

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear dimensionality reduction technique particularly well-suited for embedding high-dimensional data in a low-dimensional space. It is effective in visualizing high-dimensional data.

- **Strengths:**
  - Captures complex structures in the data.
  - Excellent for visualization of high-dimensional data.

- **Weaknesses:**
  - Computationally intensive.
  - Does not preserve global structures as well as local structures.

- **When to use:**
  - When the primary goal is visualization and understanding local structure in high-dimensional data.

```python
from sklearn.manifold import TSNE

# Applying t-SNE
tsne = TSNE(n_components=2)
df_tsne = tsne.fit_transform(df_scaled)
```

### Pairwise Controlled Manifold Approximation Projection (PacMAP)

PacMAP is a dimensionality reduction technique that focuses on preserving both global and local structures in the data, providing a balance between t-SNE's focus on local structure and PCA's focus on global structure.

- **Strengths:**
  - Balances local and global structure preservation.
  - Effective for visualizing clusters and maintaining data topology.

- **Weaknesses:**
  - Still relatively new and less tested compared to t-SNE and PCA.
  - Computationally intensive.

- **When to use:**
  - When you need a good balance between preserving local and global structures.
  - When you need to visualize clusters with better preservation of overall data topology.

```python
from pacmap import PaCMAP

# Applying PacMAP
pacmap = PaCMAP(n_components=2)
df_pacmap = pacmap.fit_transform(df_scaled)
```

> ## Exercise 1: Compare Silhouette Scores with PCA
>
> Compare the silhouette scores of K-means clustering on the original high-dimensional data and the data reduced using PCA. What do you observe?
>
> ```python
> # K-means on PCA-reduced data
> kmeans_pca = KMeans(n_clusters=5, random_state=42)
> kmeans_pca.fit(df_pca)
> labels_kmeans_pca = kmeans_pca.labels_
> silhouette_avg_kmeans_pca = silhouette_score(df_pca, labels_kmeans_pca)
> print(f"K-means on PCA-reduced data Silhouette Score: {silhouette_avg_kmeans_pca}")
> ```
>
> {:.challenge}

> ## Exercise 2: Understanding Global vs Local Structure
>
> Compare t-SNE and PacMAP on the same high-dimensional data. Visualize the results and discuss how each technique handles local and global structures.
>
> ```python
> import matplotlib.pyplot as plt
>
> # Visualizing t-SNE Clustering
> plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=labels_kmeans, cmap='viridis')
> plt.title("t-SNE Clustering")
> plt.show()
>
> # Visualizing PacMAP Clustering
> plt.scatter(df_pacmap[:, 0], df_pacmap[:, 1], c=labels_kmeans, cmap='viridis')
> plt.title("PacMAP Clustering")
> plt.show()
> ```
>
> Discuss the following:
> - How does t-SNE's focus on local structures affect the visualization?
> - How does PacMAP's balance of local and global structures compare to t-SNE?
> - Which method provides a better visualization for understanding the overall data topology?
>
> {:.challenge}

## Specialized Clustering Algorithms

### Spectral Clustering

Spectral Clustering uses the eigenvalues of a similarity matrix to perform dimensionality reduction before clustering. It is useful for capturing complex, non-linear relationships.

- **Strengths:**
  - Effective in detecting non-convex clusters.
  - Can work well with small datasets.

- **Weaknesses:**
  - Requires the computation of a similarity matrix, which can be computationally expensive for large datasets.
  - Sensitive to the choice of parameters.

- **When to use:**
  - When clusters are expected to have complex, non-linear shapes.
  - When other clustering algorithms fail to capture the structure of the data.

```python
from sklearn.cluster import SpectralClustering

# Implementing Spectral Clustering
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
labels_spectral = spectral.fit_predict(df_scaled)

# Evaluating Spectral Clustering
silhouette_avg_spectral = silhouette_score(df_scaled, labels_spectral)
print(f"Spectral Clustering Silhouette Score: {silhouette_avg_spectral}")
```

> ## Exercise 3: Implement Spectral Clustering on PCA Data
>
> Implement and evaluate Spectral Clustering on the PCA-reduced data. How does its performance compare to other methods?
>
> ```python
> # Spectral Clustering on PCA-reduced data
> labels_spectral_pca = spectral.fit_predict(df_pca)
> silhouette_avg_spectral_pca = silhouette_score(df_pca, labels_spectral_pca)
> print(f"Spectral Clustering on PCA-reduced data Silhouette Score: {silhouette_avg_spectral_pca}")
> ```
>
> {:.challenge}

## Visualizing Clustering Results

### Dimensionality Reduction using PCA

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Applying PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualizing K-means Clustering
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels_kmeans, cmap='viridis')
plt.title("K-means Clustering")
plt.show()
```

### Comparing Clustering Results

```python
# Visualizing Hierarchical Clustering
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels_hierarchical, cmap='viridis')
plt.title("Hierarchical Clustering")
plt.show()

# Visualizing DBSCAN Clustering
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels_dbscan, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.show()
```

### Visualizing t-SNE and PacMAP Results

```python
# Visualizing t-SNE Clustering
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=labels_kmeans, cmap='viridis')
plt.title("t-SNE Clustering")
plt.show()

# Visualizing PacMAP Clustering
plt.scatter(df_pacmap[:, 0], df_pacmap[:, 1], c=labels_kmeans, cmap='viridis')
plt.title("PacMAP Clustering")
plt.show()
```

> ## Exercise 4: Compare Clustering Visualizations
>
> Compare the visualizations of different clustering algorithms. Which algorithm do you think performed best and why?
>
> {:.challenge}

---

### Summary and Q&A

## Recap

- **Clustering:** Grouping similar data points together.
- **High-Dimensional Challenges:** Curse of dimensionality, sparsity, and visualization difficulties.
- **Algorithms:** K-means, Hierarchical, DBSCAN.
- **Evaluation:** Silhouette score to assess the quality of clusters.
- **Dimensionality Reduction:** PCA, t-SNE, PacMAP.
- **Specialized Algorithms:** Spectral Clustering.

## Q&A

Feel free to ask any questions or share your thoughts on today's lesson. 

---

**Follow-up Materials:**

- Documentation on clustering algorithms: [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- Further reading on high-dimensional data: [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
- Suggested exercises: Experiment with other clustering algorithms and datasets.
