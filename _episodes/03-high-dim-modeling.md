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

# Modeling Sales Prices of Houses 
A common goal associated with high-dimensional datasets is to determine if one variable of the data (e.g., sale price of house) can be predicted using other observed variables (e.g., overall quality of house, size of garage, construction date, etc.). 

In this section, we will learn how to appropriately approach modeling high-dimensional datasets using multivariate linear regression. Specifically, we will use the Ames Housing dataset to predict the sale prices of individual houses.

TODO: Explain why we are studying high-dimensional modeling through the lens of linear modeling (fast, easy to interpret, etc.)

## Import Essential Packages
~~~
import numpy as np
import pandas as pd
~~~
{: .language-python}

## Load Ames Housing Data




