---
title: "Exploring High Dimensional Data"
teaching: 0
exercises: 5
questions:
- "Key question (FIXME)"
objectives:
- "Provide intellectual access to discussions of information-age high dimensional data(sets)"
- "Define, identify, and give examples of high dimensional datasets"
- "Summarize the dimensionality of a dataset"
- "Explain best practices for how to organize / structure high dim data for reuse"
- "Demonstrate at least one method to visualize, and explore a high-dimensional dataset"
- "Describe how high dimensional data visualization and analysis can reveal a research story in noisy data."
- "Explain how to form lower dimensional descriptions/abstractions of high dimensional data"
- "Identify and explain at least one possible method and use-case for reducing dimensionality"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

## Outline
1. Define, identify, and give examples of high dimensional datasets
    1. Introduction - what is high dimensional data?
    2. What is data? 'the quantities, characters, or symbols on which operations are performed by a computer' - literally anything.
        1. (how is it formatted? structured, semi-structured, unstructured: flat file, json, raw text)
        2. There is a conversion to numerical representation happening here
        3. Original data set not rectangular, might require conversion that produces high dimensional rectangular data set. 
        4. We’re discussing structured, rectangular data only today.
    3. What is a dimension? - 'A measurable extent of some kind', 'an aspect of feature of a situation, problem, or thing' (Oxford Languages)
        5. A column of that rectangular data.
            1. Each row is an observation.
            2. There are other ways to arrange the data
    4. Provide dataset examples:
        6. 1d data set - number line, likert scale
        7. 2d data set - scatter plot, 2 columns, survey answer
        8. 3d data set - surface, scatter plot, consecutive pulses of CP 1919, 2d black and white image data set
        9. 4d data set - surface plus coloration, 2d full color image data set, a movie
        10. 30 dimensional dataset - customer feature table
        11. 20000 dimensional dataset - gene expression 
        12. Summarize the dimensionality of a dataset 
    5. Live coding: describe data set
        13. sklearn.datasets.load_breast_cancer() dataset
        14. demonstrate features
            3. description
            4. Dimensions
        15. View raw data
    6. Exercise: describe a data set
        16. Copy live coding example
        17. What’s one thing you can learn about the data?
2. Demonstrate at least one method to visualize, and explore a high-dimensional dataset
    7. Live code plotting with one of the following provided functions
    8. Exercise: Plotting of discovered dimensions (provide functions)
        18. Normalization
        19. Correlation heat map
        20. Scatter matrix
        21. facet plots
3. Describe how high dimensional data visualization and analysis can reveal a research story in noisy data.
    9. Exercise - Using your plots or new plots How can you grasp what signal is present in a 13 dimensional dataset? What does each dimension contribute?
    10. Discussion: what story or stories seem to be in this data?
4. Explain how to form lower dimensional descriptions/abstractions of high dimensional data
    11. Exercise - how would you simplify a dataset? 
        22. Are there dimensions that don’t seem to add anything?
        23. What new representations could you choose?
        24.  movie, radio pulses, customer dataset
            5. Solution
                1. Movie
                    1. Average color value of every frame’s px
                    2. reviews of the movie
                2. radio signals - just take max amplitude?
                3. Customer table - choose specific features?
            6. Note these approaches are lossy compression
5. Identify and explain at least one possible method and use-case for reducing dimensionality
    12. PCA concept slides
    13. Visualization webpage: [https://setosa.io/ev/principal-component-analysis/](https://setosa.io/ev/principal-component-analysis/)
    14. Live coding: \
	Apply PCA to breast cancer data set
    15. Exercise
        25. Apply PCA to breast cancer data set
        26. Vary parameters
        27. What effects do you notice
6. Describe how PCA can help you tell a story about a high dimensional dataset
    16. Show story/signal with PCA result
        28. malignant/benign definition.

## How to Apply PCA
~~~
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

bc = datasets.load_breast_cancer()
df = pd.DataFrame(bc.data)
df = df.set_axis(bc.feature_names, axis=1)
df['labels'] = bc.target

p = PCA(n_components=2)
components = p.fit_transform()

pca_df = pd.DataFrame(components, columns=[1,2])
pca_df['labels'] = df['labels']
df_0 = pca_df[pca_df['labels'] == 0]
df_1 = pca_df[pca_df['labels'] == 1]

fig = plt.figure(figsize(7,7))
labels  = df['labels'].tolist()
s = 10
alpha = 1
plt.scatter(df_0[1], df_0[2], s=s, label = '0', c='lightblue', alpha=alpha, zorder=2)
plt.scatter(df_1[1], df_1[2], s=s, label = '1', c='orange', alpha=alpha, zorder=1)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('Breast cancer wisconsin (diagnostic) dataset 2 component PCA')
plt.legend()
plt.show()
~~~
{:.language-python}

![PCA result](../../fig/day_1/pca_result.png)



{% include links.md %}
