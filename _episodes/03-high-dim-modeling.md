---
title: "High-dimensional Modeling and Feature Selection"
teaching: 0
exercises: 3
questions:
- "Are more features always better when trying to fit a model to your data?"
- "What does a model's level of bias or variance indicate?"
- "What are some of the popular methods to avoid overfitting when training on high-dimensional data?"
- "How can one determine which features are most relevant to a model's predictions?"
objectives:
- "Understand the challenges associated with modeling high-dimensional data"
- "Understand the importance of feature selection as a tool for modeling high-dimensional data"
- "Identify and understand some of the possible ways to perform feature selection"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME

{% include links.md %}

# Introduction
- **Curse of dimensionality**: 
- The phrase, attributed to Richard Bellman, was coined to express the difficulty of using brute force (a.k.a. grid search) to optimize a function with too many input variables.
- In machine learning, the Curse of Dimensionality refers to a set of problems that arise when working with high-dimensional data.
  - Compute Time: Generally speaking, as the number of features in a model increases, so does the amount of time required to opitmize the model
  - Overfitting: When there are more features than training observations, overfitting is gauranteed
  - Interpretability: Models become more difficult to interpret as the numbers of features increase
  - Clustering: Too many dimensions causes every observation in your dataset to appear equidistant from all the others — making it difficult to form clusters on the data. If the distances are all approximately equal, then all the observations appear equally alike (as well as equally different), and no meaningful clusters can be formed.


- **Bias VS Variance**, 10X as many datapoints as features as general rule of thumb
- Methods for feature selection

> ## Exercise - demonstrate overfitting (might just show this in intro, but leaving here as placeholder for now)
> 
> 
>
> > ## Solution
> >  
> >  
> {: .solution}
{: .challenge}

> ## Exercise
> In this example, we would like to classify images of cats versus dogs. In every image example, a cat or a dog appears at the center of the image with some background imagery present as well. There are two example images provided below. Instead of training our model on every pixel present in each image, what could we do to help the model hone in on the important aspects of the images that relate to how dogs and cats differ?
> 
>
> > ## Solution
> >  - Include only the center of each image--where a dog or a cat appears (i.e. remove the constants)
> >  - Include only pixels that contain the head of the animal--where differences are more noticeable between the species.
> {: .solution}
{: .challenge}

> ## Exercise
> In this example, 
> 
>
> > ## Solution
> >  - Include 
> {: .solution}
{: .challenge}

# Filter Methods

# Automated Feature Selection
What if don't know which features are important?

## Wrapper Methods
Forward Selection
Backward Elimination
Recursive Feature Elimination

## Embedded Methods 
lasso (L1 regularization)
ridge (L2 regularization)
