---
title: "Introduction to High-Dimensional Clustering"
teaching: 20 minutes  
exercises: 10 minutes 
questions:
- "What is clustering?"
- "Why is clustering important?"
- "What are the challenges of clustering in high-dimensional spaces?"
- "How do we implement K-means clustering in Python?"
- "How do we evaluate the results of clustering?"
objectives:
- "Explain what clustering is."
- "Discuss common clustering algorithms."
- "Highlight the challenges of clustering in high-dimensional spaces."
- "Implement and evaluate basic clustering algorithms."
keypoints: 
- "Understanding the basics of clustering and its importance."
- "Challenges in high-dimensional spaces."
- "Implementing basic clustering algorithms."
---

# Introduction to High-Dimensional Clustering

## What is Clustering?

Clustering is a method of unsupervised learning that groups a set of objects in such a way that objects in the same group (called a cluster) are more similar to each other than to those in other groups. It is used in various applications such as market segmentation, document clustering, image segmentation, and more.

## Common Clustering Algorithms

- **K-means:** Divides the data into K clusters by minimizing the variance within each cluster.
- **Hierarchical Clustering:** Builds a hierarchy of clusters either from the bottom up (agglomerative) or from the top down (divisive).
- **DBSCAN:** Density-Based Spatial Clustering of Applications with Noise, groups together points that are closely packed together while marking points that are in low-density regions as outliers.

## Challenges in High-Dimensional Spaces

High-dimensional spaces present unique challenges for clustering:

1. **Curse of Dimensionality:** As the number of dimensions increases, the distance between points becomes less meaningful, making it difficult to distinguish between clusters.
2. **Sparsity:** In high-dimensional spaces, data points tend to be sparse, which can hinder the performance of clustering algorithms.
3. **Visualization:** Visualizing clusters in high-dimensional spaces is challenging since we can only plot in two or three dimensions.

**Exercise:** Discuss with your neighbor some applications of clustering you might encounter in your research or daily life. Why is it important?

## Data Preparation for Clustering

### Loading and Inspecting the Dataset

We'll use the Ames Housing dataset for our examples. This dataset contains various features about houses in Ames, Iowa.

```python
from sklearn.datasets import fetch_openml

# Load the dataset
housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')

df = housing.data.copy(deep=True)
df = df.astype({'Id': int})
df = df.set_index('Id')
df.head()
```

### Preprocessing the Data

#### Handling Missing Values

```python
# Handling missing values
df.fillna(df.mean(), inplace=True)
```

#### Scaling Features

Clustering algorithms are sensitive to the scale of the data. We'll use StandardScaler to scale our features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

**Exercise:** Inspect the first few rows of the scaled dataset. Why is scaling important for clustering?

```python
import pandas as pd

pd.DataFrame(df_scaled, columns=df.columns).head()
```

## Implementing Clustering Algorithms

### K-means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Implementing K-means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_scaled)
labels_kmeans = kmeans.labels_

# Evaluating K-means
silhouette_avg_kmeans = silhouette_score(df_scaled, labels_kmeans)
print(f"K-means Silhouette Score: {silhouette_avg_kmeans}")
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering

# Implementing Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
labels_hierarchical = hierarchical.fit_predict(df_scaled)

# Evaluating Hierarchical Clustering
silhouette_avg_hierarchical = silhouette_score(df_scaled, labels_hierarchical)
print(f"Hierarchical Clustering Silhouette Score: {silhouette_avg_hierarchical}")
```

### DBSCAN Clustering

```python
from sklearn.cluster import DBSCAN

# Implementing DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(df_scaled)

# Evaluating DBSCAN
silhouette_avg_dbscan = silhouette_score(df_scaled, labels_dbscan)
print(f"DBSCAN Silhouette Score: {silhouette_avg_dbscan}")
```

**Exercise:** Modify the parameters of the DBSCAN algorithm and observe how the silhouette score changes. Why do you think the results differ?



