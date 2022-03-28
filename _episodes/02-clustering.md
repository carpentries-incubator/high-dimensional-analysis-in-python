---
title: "Exploring Data and Clustering"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective. (FIXME)"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME

1. Clustering in low dimensions via K-means.
    1. Generate and plot 2 blobs;
    2. Walk through steps of k-means.
    3. Live coding of k means 2 dim, with k=2.
        1. Use sklearn.datasets.load_breast_cancer() for 2 clusters +/- PCA
    4. Live coding: Example where clustering doesn’t work in low dimensions.
2. Clustering in high dimensions:
    1. Live coding: Example where k means does reveal an aspect of the signal.
    2. Live coding: Example where it doesn’t  and explain why clustering isn’t a good tool there
    3. example: [https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e](https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e) 
    4. Explore different clustering algorithms here? Probably not. The focus here is more on the dimensionality of the data and not on exploring every possible clustering algorithms
3. Provide cluster quality metrics
    1. Silhouette or inertia (sklearn)
4. Cluster on lower-dimensional version of high-dim data
    1. Explore PCA once again
        1. Remove 50% of variance and show that clustering fails
5. Misc points
    1. outliers: can exclude noise with clustering


{% include links.md %}
