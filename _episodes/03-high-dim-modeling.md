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

##### TODO: Explain why we are studying high-dimensional modeling through the lens of linear modeling (fast, easy to interpret, etc.)

### Import Essential Packages
~~~
import numpy as np
import pandas as pd
~~~
{: .language-python}

## Load the Ames Housing Data
Let's read in and briefly explore the data for this section.

See here python documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

~~~
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
~~~
{: .language-python}

 Also see here for more thorough documentation regarding the feature set: 
https://www.openml.org/d/42165


> ## Load the Ames Housing Data
>
> Spend 5 minutes exploring the Ames Housing data. Specifically, please answer the following:
> 1. What kind of object is "housing" stored as?
> 2. How many observations and features are there in the data?
> 3. What are some of the features available?
> 4. What is the name of the target feature?
>
> > ## Solution
> >
> >
> > ~~~
> > # 1. What kind of object is "housing" stored as?
> > print(type(housing)) # <class 'sklearn.utils.Bunch'>
> > # 2. How many observations and features are there in the data?
> > print(housing.keys()) # keys used to store info in housing variable
> > print(housing['data'].shape) # 80 features total, 1460 observations
> > # 3. What are some of the features available?
> > feat_names=housing['feature_names'] # get feature names
> > print(feat_names)
> > # 4. What is the name of the target feature?
> > targets=housing['target_names'] # get target name
> > print(targets) # Sale price
> > ~~~
> > {: .language-python}
> > ~~~
> > <class 'sklearn.utils.Bunch'>
> > dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
> > (1460, 80)
> > ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
> > ['SalePrice']
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}
