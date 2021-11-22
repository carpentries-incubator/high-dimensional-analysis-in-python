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
- "Identify and understand different approaches/methods for feature selection"
- "Understand the limitations of feature selection techniques and ways to assess model bias/variance tradeoff"
- "Learn how to fit and interpret univariate and multivariate linear models"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
{% include links.md %}
# Schedule Outline (internal use; will remove this later)
1. Intro/overview of ML
      * What goes into a model/estimator
      * What is the goal of the model (prediction or classification)
      * How is a model trained?
      * Live-coding: Training a univariate linear model
2. The curse of dimensionality
      * Overfitting, bias vs. variance trade-off discussion
      * Live-coding: Use same model with additional predictor variables and demonstrate overfitting
3. Methods for feature selection
      * Intro to feature selection
      * Filter methods
          * Overview
          * Live-coding: implement varianceThreshold and selectKbest (F-test and mutual information tests) using sklearn
      * Wrapper methods
          * Overview
          * Live-coding: 
      * Embedded methods
          * Overview
          * Live-coding: 

# Introduction
## Machine learning — a brief overview
Machine learning is the process of learning (optimizing) a function that is able to map some input to a desired output.
input data --> cool algorithms --> prediction forecast, recommendation ,decision

From Intro to ML lesson:
"The techniques breakdown into two broad categories, predictors and classifiers. Predictors are used to predict a value (or set of value) given a set of inputs, for example trying to predict the cost of something given the economic conditions and the cost of raw materials or predicting a country’s GDP given its life expectancy. Classifiers try to classify data into different categories, for example deciding what characters are visible in a picture of some writing or if a message is spam or not."

In this lesson, we will focus on predicting continuous values when fitting models to high-dimensional data.


## Training data 
* The text below is borrowed form this lesson: https://carpentries-incubator.github.io/machine-learning-novice-sklearn/01-introduction/index.html)
* 
Many (but not all) machine learning systems “learn” by taking a series of input data and output data and using it to form a model. The maths behind the machine learning doesn’t care what the data is as long as it can represented numerically or categorised. Some examples might include:

- predicting a person’s weight based on their height
- predicting commute times given traffic conditions
- predicting house prices given stock market prices
- classifying if an email is spam or not
- classifying what if an image contains a person or not

Typically we will need to train our models with hundreds, thousands or even millions of examples before they work well enough to do any useful predictions or classifications with them.

Some systems will do training as a one shot process which produces a model. Others might try to continuosly refine their training through the real use of the system and human feedback to it. For example every time you mark an email as spam or not spam you are probably contributing to further training of your spam filter’s model.

## **The curse of dimensionality**
- The phrase, attributed to Richard Bellman, was coined to express the difficulty of using brute force (a.k.a. grid search) to optimize a function with too many input variables.
- In machine learning, the Curse of Dimensionality refers to a set of problems that arise when working with high-dimensional data.
  - Compute Time: Generally speaking, as the number of features in a model increases, so does the amount of time required to opitmize the model
  - Overfitting: When there are more features than training observations, overfitting is gauranteed
  - Interpretability: Models become more difficult to interpret as the numbers of features increase
  - Clustering: Too many dimensions causes every observation in your dataset to appear equidistant from all the others — making it difficult to form clusters on the data. If the distances are all approximately equal, then all the observations appear equally alike (as well as equally different), and no meaningful clusters can be formed.

## **Bias-Variance Tradeoff (Underftting vs Overfitting)**
- Train error:
- Test error: 
- Overfitting: 

**Bias**: Bias is the model's average prediction error (i.e. model's predicted values minus actual values). A model with high bias pays very little attention to the training data and oversimplifies the model. It leads to a high error on both the train and test datasets.

**Variance**: Variance is the variability of the model's predictions. It indicates the overal spread of the model's predictions. A model with high variance pays too much attention to the training data (i.e. overfits) 

10X as many datapoints as features as general rule of thumb
![image alt text](https://upload.wikimedia.org/wikipedia/commons/9/9f/Bias_and_variance_contributing_to_total_error.svg)
![need to find equivalent image w/ creative commons license](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/bias_variance/bullseye.png)

> ## Exercise - demonstrate overfitting (might just show this in intro, but leaving here as placeholder for now)
> 
> 
>
> > ## Solution
> >  
> >  
> {: .solution}
{: .challenge}
# **Feature Selection**
From: https://en.wikipedia.org/wiki/Feature_selection
"In machine learning and statistics, feature selection, also known as variable selection, attribute selection or variable subset selection, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction. Feature selection techniques are used for several reasons:
1. simplification of models to make them easier to interpret by researchers/users,[1]
2. shorter training times,[2]
3. to avoid overfitting and subsequently improve the accuracy of the predictive models
4. improve data's compatibility with a learning model class,[4]
5. encode inherent symmetries present in the input space.[5][6][7][8]

# **Methods for feature selection**
## Filter Methods (Preprocessing Step)
From: https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/
"Filters methods belong to the category of feature selection methods that select features independently of the machine learning algorithm model. This is one of the biggest advantages of filter methods. Features selected using filter methods can be used as an input to any machine learning models. Another advantage of filter methods is that they are very fast. Filter methods are generally the first step in any feature selection pipeline.

Filter methods can be broadly categorized into two categories: Univariate Filter Methods and Multivariate filter methods.
The univariate filter methods are the type of methods where individual features are ranked according to specific criteria. The top N features are then selected. Different types of ranking criteria are used for univariate filter methods, for example fisher score, mutual information, and variance of the feature.

One of the major disadvantage of univariate filter methods is that they may select redundant features because the relationship between individual features is not taken into account while making decisions. Univariate filter methods are ideal for removing constant and quasi-constant features from the data.

Multivariate filter methods are capable of removing redundant features from the data since they take the mutual relationship between the features into account. Multivariate filter methods can be used to remove duplicate and correlated features from the data."

### Implementing Filter Methods using Scikit Learn 
* make note about applying filter methods using training set data
- Pearson's Correlation
- ANOVA
- Remove constants

## Wrapper Methods
With wrapper methods, we train many models on various subsets of the feature space. Since this method requires training and testing numerous models, it tends to be very computationally expensive. Some wrapper methods include:
- Forward Selection
- Backward Elimination
- Recursive Feature Elimination

## Embedded Methods 
- lasso (L1 regularization)
- ridge (L2 regularization)




> ## Exercise
> In this example, 
> 
>
> > ## Solution
> >  - Include 
> {: .solution}
{: .challenge}


