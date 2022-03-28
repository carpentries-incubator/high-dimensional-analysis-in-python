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



1. Intro/overview of modeling as a tool to tell a story about data
    1. Example of pre-prepared model inference with Ames housing data.
        1. What goes into a model/estimator
        2. What is the goal of the model (prediction or classification)
        3. How is a model trained? Optimizing error functions.
            1. be clear about model training vs inference
            2. be clear about why we care about modeling, and how it differs from clustering and data exploration
        4. Provide linear modeling visualization 
        5. Live-coding: Training a univariate linear model; 
        6. Exercise: Have attendees select their own variable to predict housing prices and discuss the result
        7. Exercise: Have attendees interpret their univariate models
2. Feature selection for univariate linear model
    1. Cover a filter or wrapper method here. We’ll repeat this once we get to multivariate linear models.
    2. Compare and contrast high vs. low variance predictors
3. Multivariate models and the curse of dimensionality
    1. Overfitting, bias vs. variance trade-off discussion
    2. How to assess overfitting
    3. Live-coding: Use same model given in intro section but add additional (all) predictor variables
        1. Demonstrate overfitting
        2. Comment on interpretability and training time compared to univariate
    4. Exercise: ?
4. Methods for feature selection & regularization
    1. Intro to feature selection; exercises to illustrate the concept of feature selection
    2. Filter methods
        1. Overview
        2. Live-coding: implement varianceThreshold and selectKbest (F-test and mutual information tests) using sklearn
    3. Wrapper methods
        1. Overview of forward selection
        2. Live-coding: implement forward selection using mlxtend library (built in wrapper methods
        3. Overview of backward elimination
        4. Live-coding: implement backward elimination using mlxtend library (built in wrapper methods)
    4. **Embedded methods (primary focus)**
        1. Overview of ridge and lasso regression
        2. Live-coding: implement both ridge and lasso using sklearn. Ask students to compare performance of model to unregularized model
    5. Compare and contrast feature selection with feature transformation (PCA)
5. Model interpretability / wrap-up 
    1. How actionable is the model?
        1. Can outliers be detected?
        2. Exercise: Which features might be helpful for predicting housing prices? Are any features surprising (hint: compare to feature correlation heat map)?
    2. Final Exercise
        1. Compare performance across all feature selection methods using a new dataset.
        2. Questions to consider....
            3. Which feature selection method yielded the best performing model?
            4. Which features contributed the most to your best model’s predictions?
            5. When might you choose one feature selection over another?


