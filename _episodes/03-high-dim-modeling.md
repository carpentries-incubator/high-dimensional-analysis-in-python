---
title: "Regression with many features"
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

# Ames Housing Data
In the previous lesson, we applied PCA to the Ames housing dataset to better understand X properties of the data, including:
- degree of feature redundancy
- underlying factors present in the data

In this section, we will attempt to predict housing prices from the features recorded in the Ames Housing dataset (e.g., overall quality of house, size of garage, construction date, etc.). 




