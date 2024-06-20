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
High-dimensional clustering often requires reducing the number of dimensions to make the problem more tractable. The choice of dimensionality reduction technique can significantly impact the clustering results. Here, we will explore several popular techniques and discuss their strengths and weaknesses.

### Principal Component Analysis (PCA)

PCA reduces the dimensionality of the data by transforming it into a new set of variables that are linear combinations of the original variables. It aims to capture the maximum variance with the fewest number of components.

- **Strengths:**
  - Reduces dimensionality while preserving variance.
  - Simplifies visualization.
  - Computationally efficient.

- **Weaknesses:**
  - PCA prioritizes directions of maximum variance across the entire dataset, often capturing global patterns at the expense of local relationships. This means that while PCA can capture large-scale trends and overall data distribution, it might not maintain the subtle, local relationships between nearby data points​ .
  - Linear method, may not capture complex nonlinear relationships.
  - PCA can be significantly influenced by outliers since these points can disproportionately affect the variance calculations. This further distorts the local structures, as outliers may dominate the principal components and obscure the true local relationships within the data

- **When to use:**
  - When you need to reduce dimensionality for visualization.
  - When you have a large number of features and need to reduce them for computational efficiency.

#### Local vs global structure
When reducing high-dimensional data to lower dimensions, it's important to consider both local and global structures. Local structure refers to the relationships and distances between nearby data points. Preserving local structure ensures that points that were close together in the high-dimensional space remain close in the lower-dimensional representation. This is crucial for identifying clusters or neighborhoods within the data. Global structure, on the other hand, refers to the overall arrangement and distances between clusters or distant points in the dataset. Preserving global structure ensures that the broader data topology and the relative positioning of different clusters are maintained. 

```python
from sklearn.decomposition import PCA

# Applying PCA
pca = PCA(n_components=10)  # Reduce to 10 dimensions
df_pca = pca.fit_transform(df_scaled)
```

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear dimensionality reduction technique particularly well-suited for embedding high-dimensional data in a low-dimensional space. It is effective in visualizing high-dimensional data. Techniques like t-SNE excel at preserving local structure, making them great for detailed cluster visualization, but they may distort the global arrangement of clusters. 

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
Determining whether to prioritize local or global structure in a research clustering context depends on the goals of your analysis and the nature of the data. Here are some key considerations:

## Global vs Local Structure Deep Dive
Understanding and choosing the right technique based on the need to preserve either local or global structure (or both) can significantly impact the insights drawn from the data visualization.

### When to Care More About Local Structure

1. **Cluster Identification and Separation**:
   - If your primary goal is to identify and separate distinct clusters within your data, preserving local structure is crucial. Techniques that focus on local structure, such as t-SNE, ensure that points that are close in high-dimensional space remain close in the reduced space, making clusters more discernible.
   - **Example**: In gene expression data, where the goal is to identify distinct groups of genes or samples with similar expression patterns, preserving local neighborhoods is essential for accurate clustering.

2. **Neighborhood Analysis**:
   - When the analysis requires examining the relationships between nearby data points, preserving local structure becomes important. This is common in studies where understanding local patterns or small-scale variations is key.
   - **Example**: In image recognition tasks, local structure preservation helps in identifying small groups of similar images, which can be crucial for tasks like facial recognition or object detection.

3. **Anomaly Detection**:
   - For tasks like anomaly detection, where identifying outliers or unusual patterns within small regions of the data is important, maintaining local structure ensures that these patterns are not lost during dimensionality reduction.
   - **Example**: In network security, preserving local structure helps in detecting abnormal user behavior or network activity that deviates from typical patterns.

### When to Care More About Global Structure

1. **Overall Data Topology**:
   - If understanding the broad, overall arrangement of data points is critical, preserving global structure is essential. This helps in maintaining the relative distances and relationships between distant points, providing a comprehensive view of the data's topology.
   - **Example**: In geographical data analysis, maintaining global structure can help in understanding the broader spatial distribution of features like climate patterns or population density.

2. **Data Integration and Comparison**:
   - When integrating multiple datasets or comparing data across different conditions, preserving global structure helps in maintaining consistency and comparability across the entire dataset.
   - **Example**: In multi-omics studies, where different types of biological data (e.g., genomics, proteomics) are integrated, preserving global structure ensures that the overall relationships between data types are maintained.

3. **Data Compression and Visualization**:
   - For tasks that require data compression or large-scale visualization, preserving global structure can help in maintaining the integrity of the dataset while reducing its dimensionality.
   - **Example**: In large-scale data visualization, techniques that preserve global structure, such as PCA, help in creating interpretable visual summaries that reflect the overall data distribution.

### Balancing Local and Global Structures

In many cases, it may be necessary to strike a balance between preserving local and global structures. Techniques like PacMAP offer a compromise by maintaining both local and global relationships, making them suitable for applications where both detailed clustering and overall data topology are important.

**Key Questions to Consider**:
- What is the primary goal of your clustering analysis? (e.g., identifying distinct clusters, understanding overall data distribution)
- What is the nature of your data? (e.g., high-dimensional, non-linear structures, presence of noise or outliers)
- Are you integrating multiple datasets or focusing on a single dataset?
- Is detailed local information more critical than understanding broad patterns, or vice versa?

By carefully considering these factors, you can determine the appropriate emphasis on local or global structure for your specific research context.

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
