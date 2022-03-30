---
title: "Exploring Data and Clustering"
teaching: 0
exercises: 0
questions:
- "How can I cluster points in a low dimensional space?"
- "What's different about clustering points in high dimensions? What
are some things to be aware of?"

objectives:
- "Use `sklearn` to cluster points."
- "Construct an example that illustrates why clustering in high
dimensions might fail due to the so-called 'curse of dimensionality.'"
- "Demonstrate one way of applying clustering to work-around the
problems in high dimensions." 

keypoints:

- "Clustering is a heuristic way of looking at data that, when applied
with the appropriate caveats, can lead to insights and suggest more
formal analyses."

---

> Fix objective #3.

> Make sure the key point is aligned with what we develop in the lesson.

## Outline

1. K-means clustering in low dimensions.

    1. Generate and plot 2 blobs;
    2. Walk through steps of k-means.
    3. Live coding of k means 2 dim, with k=2.
        1. Use sklearn.datasets.load_breast_cancer() for 2 clusters +/- PCA

2. Clustering in high dimensions:
    2. Live coding: Example where it doesn’t  and explain why clustering isn’t a good tool there
    3. example: [https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e](https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e) 

3. Provide cluster quality metrics
    1. Silhouette or inertia (sklearn)
4. Cluster on lower-dimensional version of high-dim data
    1. Explore PCA once again
        1. Remove 50% of variance and show that clustering fails
5. Misc points
    1. outliers: can exclude noise with clustering

## Clustering: A first look

First, let's generate two *blobs* and cluster them using the k-means algorithm.

~~~
# create two blobs  (pseudo code)
blob1 = makeBlobs(100 points in d=2, y mean = 1, x mean = 1
blob2 = makeBlobs(100 points in d=2, y mean = 1, x mean = 5

plot blob1 and blob2, using different shapes

~~~
{: .language-python}

Great.  Now let's first run the k means algorithm and then we'll go
over what it does.

~~~
load sklearn;
run kmeans k=2 on the blobs and color them by group
plot result;
~~~
{: .language-python}

Exercise: Look at the result. Compare the shapes and colors. Explain
what's happening.
Note: If the blobs are random picks, your plot will differ slightly
from others'.

### What is k-means?

Include a couple of slides or code block that explain 2-means clustering.
~~~


~~~
{: .language-python}


~~~

~~~
{: .language-python}



{% include links.md %}
