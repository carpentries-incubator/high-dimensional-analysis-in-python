---
title: High-dimensional data preparation
teaching: 45
exercises: 2
keypoints:
- ""
objectives:
- "Know how to deal with skewed a target variable, outliers, missing values, and categorical data"
- "Understand the importance of standarizing predictor variables"
questions:
- ""
---

### Load the data


```python
## Ames housing dataset. 

# See here for thorough documentation regarding the feature set: 
# https://www.openml.org/d/42165
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
```


```python
# add blanks between 
print(f"housing['data'].shape = {housing['data'].shape}\n") # 80 features total, 1460 observations
print(f"housing['feature_names'] = {housing['feature_names']}\n")
print(f"housing['target_names'] = {housing['target_names']}\n")
```

    housing['data'].shape = (1460, 80)
    
    housing['feature_names'] = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
    
    housing['target_names'] = ['SalePrice']
    
    


```python
# Extract X and y
X=housing['data']
y=housing['target']

```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>60.0</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>20.0</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2007.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>60.0</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>70.0</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2006.0</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>60.0</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



#### Plotting the target variable distribution
Explore the distribution of sale prices. Is the distribution uniform or skewed left/right?


```python
# plot histogram of housing sales, show mean and std of prices as well
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.distplot(y)
title = plt.title('House Price Distribution')
print('Mean Sale Price:', np.mean(y))
print('Standard Deviation:', np.std(y))
```

    C:\Users\Endemann\anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    

    Mean Sale Price: 180921.19589041095
    Standard Deviation: 79415.29188606751
    


    

    


#### Skewed Target Variable
This distribution has a long right tail, suggesting that it is skewed. 

Why do we care if the data is skewed? If the response variable is right skewed, the model will be trained on a much larger number of moderately priced homes, and will be less likely to successfully predict the price for the most expensive houses. In addition, the presence of a highly skewed target varible can, more likely, influence the distribution of residuals making them, in turn, non-normal. Normal residuals are required for hypothesis testing.

The concept is the same as training a model on imbalanced categorical classes. If the values of a certain independent variable (feature) are skewed, depending on the model, skewness may violate model assumptions (e.g. logistic regression) or may impair the interpretation of feature importance.


```python
# To quantitatively assess a distribution's skewness, we can use pandas' skew() function
y.skew() 
```




    1.8828757597682129



· If the skewness is between -0.5 and 0.5, the data are fairly symmetrical

· If the skewness is between -1 and — 0.5 or between 0.5 and 1, the data are moderately skewed

· If the skewness is less than -1 or greater than 1, the data are highly skewed

#### Correcting skewed target variable using log transformation
We can correct for a skewed variable by adjusting the scale of the variable. One commonly used rescaling technique that can correct for skew is applying a log transformation. 


```python
# Correct for skew using log transformation
y = np.log(y)
plt.figure(figsize=(8,6))
sns.distplot(y)
title = plt.title("House Price Distribution")
print(np.mean(y))
print(np.std(y))
y.skew()
```

    12.024050901109373
    0.3993150462437029
    

    C:\Users\Endemann\anaconda3\lib\site-packages\seaborn\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    




    0.12133506220520406




    

    


Our data now appears to be normal and has a skew value of only .399 — meaning the data is now fairly symmetrical. When we correct the target variable skew using a log transformation but not the predictors, the resulting model fit to this data is a log-linear model, meaning a log dependent variable with linear explanatory variables. 

**Note on other skew correction methods**: While a log transformation is probably the most common way to fix a skewed variable, there are other rescaling methods available to explore, e.g., Box Cox transformation.

#### Skewed Predictor Variables
What happens if our predictor variables are also skewed? Does this have any impact on the model, and should we correct for predictor variable skew?

Technically speaking, the only distributional assumption we have to look out for when doing hypothesis testing with linear models is that the model's residuals are normally distributed. As long as this is true, the underlying independent variable can be as non-normal as you like. However, sometimes the presence of skewed predictors can lead to less stable model predictions because long tails or outliers in the predictor variable distrutions require an analsyis of leverage (i.e. how much these outliers impact on the estimate of the regression coefficients).

Thus, for very skewed variables it might be a good idea to transform the data to eliminate the harmful effects. If there's just a small amount of skew, you likely are fine to move forward with the data as is. We will return to the impact of skewed predictor variables later in the lesson. For now, we will leave all predictor variables as they are.

### Code all nominal and dichotomous variables as "dummy variables" via one-hot encoding


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>60.0</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>20.0</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2007.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>60.0</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>70.0</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2006.0</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>60.0</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260.0</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>2008.0</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
from helper_functions import encode_predictors_housing_data
X_encoded = X.copy(deep=True) 
X_encoded = encode_predictors_housing_data(X_encoded)
X_encoded.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass_20.0</th>
      <th>MSSubClass_30.0</th>
      <th>MSSubClass_40.0</th>
      <th>MSSubClass_45.0</th>
      <th>MSSubClass_50.0</th>
      <th>MSSubClass_60.0</th>
      <th>MSSubClass_70.0</th>
      <th>MSSubClass_75.0</th>
      <th>MSSubClass_80.0</th>
      <th>MSSubClass_85.0</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>Street</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>61.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>298.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2007.0</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008.0</td>
      <td>9.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>272.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2006.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>192.0</td>
      <td>84.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 215 columns</p>
</div>



### Remove columns/predictors that meet any of the following criteria...
* Presence of one or more NaN value (note: interpolation is also an option here)
* Column is a constant or nearly constant (i.e., one value is present across 90% or more of the rows). Constant predictors have no prediction power, and low-variance predictors have very little prediction power.


```python
# Remove variables that have NaNs as observations and vars that have a constant value across all observations
from helper_functions import remove_bad_cols
X_encoded_good = remove_bad_cols(X_encoded, .95)
X_encoded_good.head()

```

    MSSubClass_30.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_40.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_45.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_70.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_75.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_80.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_85.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_90.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_160.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_180.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSSubClass_190.0 removed due to lack of variance ( >95.0% rows have the same value value)
    MSZoning_C (all) removed due to lack of variance ( >95.0% rows have the same value value)
    MSZoning_FV removed due to lack of variance ( >95.0% rows have the same value value)
    MSZoning_RH removed due to lack of variance ( >95.0% rows have the same value value)
    Alley_Grvl removed due to lack of variance ( >95.0% rows have the same value value)
    Alley_Pave removed due to lack of variance ( >95.0% rows have the same value value)
    LandContour_Bnk removed due to lack of variance ( >95.0% rows have the same value value)
    LandContour_HLS removed due to lack of variance ( >95.0% rows have the same value value)
    LandContour_Low removed due to lack of variance ( >95.0% rows have the same value value)
    LotConfig_FR2 removed due to lack of variance ( >95.0% rows have the same value value)
    LotConfig_FR3 removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_Blmngtn removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_Blueste removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_BrDale removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_BrkSide removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_ClearCr removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_Crawfor removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_IDOTRR removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_MeadowV removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_Mitchel removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_NPkVill removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_NoRidge removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_SWISU removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_SawyerW removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_StoneBr removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_Timber removed due to lack of variance ( >95.0% rows have the same value value)
    Neighborhood_Veenker removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_Artery removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_PosA removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_PosN removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_RRAe removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_RRAn removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_RRNe removed due to lack of variance ( >95.0% rows have the same value value)
    Condition1_RRNn removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_Artery removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_Feedr removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_Norm removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_PosA removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_PosN removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_RRAe removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_RRAn removed due to lack of variance ( >95.0% rows have the same value value)
    Condition2_RRNn removed due to lack of variance ( >95.0% rows have the same value value)
    BldgType_2fmCon removed due to lack of variance ( >95.0% rows have the same value value)
    BldgType_Duplex removed due to lack of variance ( >95.0% rows have the same value value)
    BldgType_Twnhs removed due to lack of variance ( >95.0% rows have the same value value)
    HouseStyle_1.5Unf removed due to lack of variance ( >95.0% rows have the same value value)
    HouseStyle_2.5Fin removed due to lack of variance ( >95.0% rows have the same value value)
    HouseStyle_2.5Unf removed due to lack of variance ( >95.0% rows have the same value value)
    HouseStyle_SFoyer removed due to lack of variance ( >95.0% rows have the same value value)
    HouseStyle_SLvl removed due to lack of variance ( >95.0% rows have the same value value)
    RoofStyle_Flat removed due to lack of variance ( >95.0% rows have the same value value)
    RoofStyle_Gambrel removed due to lack of variance ( >95.0% rows have the same value value)
    RoofStyle_Mansard removed due to lack of variance ( >95.0% rows have the same value value)
    RoofStyle_Shed removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_ClyTile removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_CompShg removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_Membran removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_Metal removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_Roll removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_Tar&Grv removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_WdShake removed due to lack of variance ( >95.0% rows have the same value value)
    RoofMatl_WdShngl removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_AsbShng removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_AsphShn removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_BrkComm removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_BrkFace removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_CBlock removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_CemntBd removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_ImStucc removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_Stone removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_Stucco removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior1st_WdShing removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_AsbShng removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_AsphShn removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_Brk Cmn removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_BrkFace removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_CBlock removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_CmentBd removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_ImStucc removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_Other removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_Stone removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_Stucco removed due to lack of variance ( >95.0% rows have the same value value)
    Exterior2nd_Wd Shng removed due to lack of variance ( >95.0% rows have the same value value)
    MasVnrType_BrkCmn removed due to lack of variance ( >95.0% rows have the same value value)
    Foundation_Slab removed due to lack of variance ( >95.0% rows have the same value value)
    Foundation_Stone removed due to lack of variance ( >95.0% rows have the same value value)
    Foundation_Wood removed due to lack of variance ( >95.0% rows have the same value value)
    Heating_Floor removed due to lack of variance ( >95.0% rows have the same value value)
    Heating_GasA removed due to lack of variance ( >95.0% rows have the same value value)
    Heating_GasW removed due to lack of variance ( >95.0% rows have the same value value)
    Heating_Grav removed due to lack of variance ( >95.0% rows have the same value value)
    Heating_OthW removed due to lack of variance ( >95.0% rows have the same value value)
    Heating_Wall removed due to lack of variance ( >95.0% rows have the same value value)
    Electrical_FuseF removed due to lack of variance ( >95.0% rows have the same value value)
    Electrical_FuseP removed due to lack of variance ( >95.0% rows have the same value value)
    Electrical_Mix removed due to lack of variance ( >95.0% rows have the same value value)
    GarageType_2Types removed due to lack of variance ( >95.0% rows have the same value value)
    GarageType_Basment removed due to lack of variance ( >95.0% rows have the same value value)
    GarageType_CarPort removed due to lack of variance ( >95.0% rows have the same value value)
    MiscFeature_Gar2 removed due to lack of variance ( >95.0% rows have the same value value)
    MiscFeature_Othr removed due to lack of variance ( >95.0% rows have the same value value)
    MiscFeature_Shed removed due to lack of variance ( >95.0% rows have the same value value)
    MiscFeature_TenC removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_COD removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_CWD removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_Con removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_ConLD removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_ConLI removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_ConLw removed due to lack of variance ( >95.0% rows have the same value value)
    SaleType_Oth removed due to lack of variance ( >95.0% rows have the same value value)
    SaleCondition_AdjLand removed due to lack of variance ( >95.0% rows have the same value value)
    SaleCondition_Alloca removed due to lack of variance ( >95.0% rows have the same value value)
    SaleCondition_Family removed due to lack of variance ( >95.0% rows have the same value value)
    Utilities_AllPub removed due to lack of variance ( >95.0% rows have the same value value)
    Utilities_NoSeWa removed due to lack of variance ( >95.0% rows have the same value value)
    LotFrontage removed due to presence of NaNs (sum of nans = 259)
    MasVnrArea removed due to presence of NaNs (sum of nans = 8)
    LowQualFinSF removed due to lack of variance ( >95.0% rows have the same value value)
    KitchenAbvGr removed due to lack of variance ( >95.0% rows have the same value value)
    GarageYrBlt removed due to presence of NaNs (sum of nans = 81)
    3SsnPorch removed due to lack of variance ( >95.0% rows have the same value value)
    PoolArea removed due to lack of variance ( >95.0% rows have the same value value)
    Street removed due to lack of variance ( >95.0% rows have the same value value)
    All columns removed: ['MSSubClass_30.0', 'MSSubClass_40.0', 'MSSubClass_45.0', 'MSSubClass_70.0', 'MSSubClass_75.0', 'MSSubClass_80.0', 'MSSubClass_85.0', 'MSSubClass_90.0', 'MSSubClass_160.0', 'MSSubClass_180.0', 'MSSubClass_190.0', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'Alley_Grvl', 'Alley_Pave', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LotConfig_FR2', 'LotConfig_FR3', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_Crawfor', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NPkVill', 'Neighborhood_NoRidge', 'Neighborhood_SWISU', 'Neighborhood_SawyerW', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gambrel', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_ImStucc', 'Exterior2nd_Other', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'GarageType_2Types', 'GarageType_Basment', 'GarageType_CarPort', 'MiscFeature_Gar2', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_Oth', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'Utilities_AllPub', 'Utilities_NoSeWa', 'LotFrontage', 'MasVnrArea', 'LowQualFinSF', 'KitchenAbvGr', 'GarageYrBlt', '3SsnPorch', 'PoolArea', 'Street']
    # of columns removed: 133
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass_20.0</th>
      <th>MSSubClass_50.0</th>
      <th>MSSubClass_60.0</th>
      <th>MSSubClass_120.0</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>LandContour_Lvl</th>
      <th>LotConfig_Corner</th>
      <th>LotConfig_CulDSac</th>
      <th>LotConfig_Inside</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>ScreenPorch</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>548.0</td>
      <td>0.0</td>
      <td>61.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008.0</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>460.0</td>
      <td>298.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2007.0</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>608.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008.0</td>
      <td>9.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>642.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>272.0</td>
      <td>0.0</td>
      <td>2006.0</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>836.0</td>
      <td>192.0</td>
      <td>84.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2008.0</td>
      <td>12.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



**Note**: can replace NaNs with some interpolation instead of dropping columns. Your definition of bad may and possibly should differ!

#### End of data cleaning stage
At this point, we have prepared our X dataframe to contain all independent variables of interest. Our Y variable (pands series object) contains only the response/dependent variable we are trying to predict -- housing prices. 


```python
# quick check of data-types and dimensions post-cleaning efforts
print(X_encoded_good.shape)
print(y.shape)
```

    (1460, 82)
    (1460,)
    

#### Use means and stds from training data to zscore test data


```python
def zscore_test(test_df, train_means, train_stds):
    cols = test_df.columns
    for col in cols:
        test_df.loc[:,col] = (test_df[col] - train_means[col])/train_stds[col]
        
    return test_df
```


```python
X_test_zscore = X_test.copy(deep=True) 
X_test_zscore = zscore_test(X_test_zscore, train_means, train_stds)
X_test_zscore.head()

# add plot of distribution before/after
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass_20.0</th>
      <th>MSSubClass_30.0</th>
      <th>MSSubClass_40.0</th>
      <th>MSSubClass_45.0</th>
      <th>MSSubClass_50.0</th>
      <th>MSSubClass_60.0</th>
      <th>MSSubClass_70.0</th>
      <th>MSSubClass_75.0</th>
      <th>MSSubClass_80.0</th>
      <th>MSSubClass_85.0</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>Street</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>529</th>
      <td>1.295274</td>
      <td>-0.216936</td>
      <td>-0.055442</td>
      <td>-0.096325</td>
      <td>-0.327838</td>
      <td>-0.514713</td>
      <td>-0.214342</td>
      <td>-0.096325</td>
      <td>-0.189684</td>
      <td>-0.116007</td>
      <td>...</td>
      <td>-0.742245</td>
      <td>-0.705987</td>
      <td>2.735143</td>
      <td>-0.117585</td>
      <td>-0.275025</td>
      <td>-0.054775</td>
      <td>-0.615835</td>
      <td>-1.221678</td>
      <td>0.071648</td>
      <td>0.266685</td>
    </tr>
    <tr>
      <th>491</th>
      <td>-0.771248</td>
      <td>-0.216936</td>
      <td>-0.055442</td>
      <td>-0.096325</td>
      <td>3.047168</td>
      <td>-0.514713</td>
      <td>-0.214342</td>
      <td>-0.096325</td>
      <td>-0.189684</td>
      <td>-0.116007</td>
      <td>...</td>
      <td>-0.742245</td>
      <td>-0.705987</td>
      <td>0.129185</td>
      <td>-0.117585</td>
      <td>-0.275025</td>
      <td>-0.054775</td>
      <td>-1.357568</td>
      <td>0.640543</td>
      <td>0.071648</td>
      <td>0.266685</td>
    </tr>
    <tr>
      <th>459</th>
      <td>-0.771248</td>
      <td>-0.216936</td>
      <td>-0.055442</td>
      <td>-0.096325</td>
      <td>3.047168</td>
      <td>-0.514713</td>
      <td>-0.214342</td>
      <td>-0.096325</td>
      <td>-0.189684</td>
      <td>-0.116007</td>
      <td>...</td>
      <td>-0.742245</td>
      <td>-0.705987</td>
      <td>3.479702</td>
      <td>-0.117585</td>
      <td>-0.275025</td>
      <td>-0.054775</td>
      <td>0.867630</td>
      <td>0.268099</td>
      <td>0.071648</td>
      <td>0.266685</td>
    </tr>
    <tr>
      <th>279</th>
      <td>-0.771248</td>
      <td>-0.216936</td>
      <td>-0.055442</td>
      <td>-0.096325</td>
      <td>-0.327838</td>
      <td>1.940844</td>
      <td>-0.214342</td>
      <td>-0.096325</td>
      <td>-0.189684</td>
      <td>-0.116007</td>
      <td>...</td>
      <td>1.500759</td>
      <td>1.020040</td>
      <td>-0.367189</td>
      <td>-0.117585</td>
      <td>-0.275025</td>
      <td>-0.054775</td>
      <td>0.125897</td>
      <td>-1.221678</td>
      <td>0.071648</td>
      <td>0.266685</td>
    </tr>
    <tr>
      <th>655</th>
      <td>-0.771248</td>
      <td>-0.216936</td>
      <td>-0.055442</td>
      <td>-0.096325</td>
      <td>-0.327838</td>
      <td>-0.514713</td>
      <td>-0.214342</td>
      <td>-0.096325</td>
      <td>-0.189684</td>
      <td>-0.116007</td>
      <td>...</td>
      <td>-0.742245</td>
      <td>-0.705987</td>
      <td>-0.367189</td>
      <td>-0.117585</td>
      <td>-0.275025</td>
      <td>-0.054775</td>
      <td>1.609362</td>
      <td>-1.221678</td>
      <td>0.071648</td>
      <td>0.266685</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 215 columns</p>
</div>



#### Should we standardize the target variable as well?
In the context of linear modeling using OLS, standardizing the target varible is not necessary. Standardization of target variables is a common practice used when models make use of gradient descent to solve for the model parameters. Gradient descent tends to converge much faster if the target variable has a smaller range. 


## Train univariate models

Define a function `train_linear_model` to help us fit a linear model to data using a model_type argument to specify which model to use


```python
# TODO: toy example before getting into the functions
reg = LinearRegression().fit(X_train,y_train)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-18-debd002b698a> in <module>
          1 # TODO: toy example before getting into the functions
    ----> 2 reg = LinearRegression().fit(X_train,y_train)
    

    NameError: name 'LinearRegression' is not defined



```python
# sklearn version
from sklearn.linear_model import LinearRegression

# stats model version (for hypothesis testing)
from statsmodels.formula.api import ols

def train_linear_model(X_train, y_train, model_type):
    if model_type == "unregularized":
        reg = LinearRegression().fit(X_train,y_train)
#         reg = ols("dist ~ speed", data=cars).fit()
    else:
        raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)
  
    # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
    print('# model coefs = ' + str(len(reg.coef_)+1))

    return reg


```

Define a function `measure_model_err` to help us measure the model's performance (train/test RMSE)


```python
from sklearn import metrics

def measure_model_err(X_train, X_test, y_train, y_test, reg):
    
    # 1) get model predicitons based on transformed (z-scored) predictor vars
    y_pred_train=reg.predict(X_train)
    y_pred_test=reg.predict(X_test)
    
    # 2) reverse log transformation (exponential)
    y_pred_train=np.exp(y_pred_train)
    y_pred_test=np.exp(y_pred_test)
    y_train=np.exp(y_train)
    y_test=np.exp(y_test)
    
    # 3) calculate RMSE for train and test sets
    RMSE_train = metrics.mean_squared_error(y_train, y_pred_train,squared=False) # squared=False to get RMSE instead of MSE
    R2_train = reg.score(X_train, y_train) # returns R^2 ("coef of determination")
    RMSE_test = metrics.mean_squared_error(y_test, y_pred_test,squared=False) 
    R2_test = reg.score(X_test, y_test) # returns R^2 ("coef of determination")

    return RMSE_train, RMSE_test, R2_train, R2_test
```

Define a function `fit_eval_model` that will call both `train_linear_model` and `measure_model_err` and report back on model performance.


```python
def fit_eval_model(X_train, y_train, X_test, y_test, predictor_vars, model_type):
    '''This function uses the predictor vars specified by predictor_vars to predict housing price. Function returns RMSE for both train and test data'''
    # Convert response vectors from pandas series to numpy arrays. 
    # This is necessary for downstream analyses (required format for linear regression fucntion we'll use).
    y_train=np.array(y_train) 
    y_test=np.array(y_test) 

    # Index specific predictor vars. Use reshape to handle case of just one predictor var (convert to shape=[numRows,numvars] rather than shape=[numRows,] )
    X_train=np.array(X_train[predictor_vars]).reshape(-1, len(predictor_vars)) # index subset of predictor vars
    X_test=np.array(X_test[predictor_vars]).reshape(-1, len(predictor_vars)) # do the same for test set

    # report predictor var if there's only one
    if len(predictor_vars)==1:
        preview_predict_var = ' (' + predictor_vars[0] + ')'
    else:
        preview_predict_var = ''

    # print number of observations in train/test sets as well as number of features used to predict housing price
    print('# of predictor vars = ' + str(len(predictor_vars)) + preview_predict_var)
    print('# of train observations = ' + str(X_train.shape[0]))
    print('# of test observations = ' + str(X_test.shape[0]))
  
    # fit model to training data
    reg = train_linear_model(X_train, y_train, model_type)

    # get train and test set RMSE
    RMSE_train, RMSE_test = measure_model_err(X_train, X_test, y_train, y_test, reg)

    # print results
    print('Train RMSE:', RMSE_train)
    print('Test RMSE:', RMSE_test)
    perc_diff = (RMSE_test-RMSE_train)/RMSE_train
    perc_diff = "{:.0%}".format(perc_diff)
    print('(Test-Train)/Train:', perc_diff)
    return RMSE_train, RMSE_test

```

#### Determine which single variable is most predictive of housing prices


```python
import pandas as pd 

all_feats=X_train.columns
RMSE_train_list=[None] * len(all_feats)
RMSE_test_list=[None] * len(all_feats)

feat_index=0
for feat in all_feats:  
    # fit univariate model and return train/test RMSE
    RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, 
                                           X_test, y_test,
                                           [feat],'unregularized')
    print('')
    # store model errors
    RMSE_train_list[feat_index] = RMSE_train
    RMSE_test_list[feat_index] = RMSE_test#metrics.mean_squared_error(y_test, predicted_test,squared=False) # squared=False to get RMSE instead of MSE
    feat_index+=1
    
# store errors in pandas dataframe for ease of access downstream
df_model_err = pd.DataFrame()
df_model_err['Predictor Variable'] = all_feats
df_model_err['Train RMSE'] = RMSE_train_list
df_model_err['Test RMSE'] = RMSE_test_list

```

    # of predictor vars = 1 (MSSubClass_20.0)
    # of train observations = 978
    # of test observations = 482
    # model coefs = 2
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-63-e4a90b63b167> in <module>
          8 for feat in all_feats:
          9     # fit univariate model and return train/test RMSE
    ---> 10     RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, 
         11                                            X_test, y_test,
         12                                            [feat],'unregularized')
    

    <ipython-input-62-b8c93fc000b0> in fit_eval_model(X_train, y_train, X_test, y_test, predictor_vars, model_type)
         25 
         26     # get train and test set RMSE
    ---> 27     RMSE_train, RMSE_test = measure_model_err(X_train, X_test, y_train, y_test, reg)
         28 
         29     # print results
    

    ValueError: too many values to unpack (expected 2)


### Plot out train/test error vs predictor var


```python
# Let's take a closer look at the results by sorting the test error from best feature to worst. We'll then plot performance by feature for both train and test data.
RMSE_test = np.asarray(df_model_err['Test RMSE'])
sort_inds=[i[0] for i in sorted(enumerate(RMSE_test), key=lambda x:x[1])]
sort_inds = np.array(sort_inds)

# now that we have the sort indices based on test set performance, we'll sort the trainErr, testErr, and feature name vectors
RMSE_train = np.asarray(df_model_err['Train RMSE'])
all_feats = df_model_err['Predictor Variable']

RMSE_train=RMSE_train[sort_inds]
RMSE_test=RMSE_test[sort_inds]
labels=all_feats[sort_inds]

print(labels)
print(len(labels))
```

    57    OverallQual
    73     GarageCars
    55      YearBuilt
    68       FullBath
    74     GarageArea
             ...     
    58    OverallCond
    59     BsmtFinSF1
    65      GrLivArea
    63       1stFlrSF
    62    TotalBsmtSF
    Name: Predictor Variable, Length: 82, dtype: object
    82
    


```python
# plot out top 10 features based on RMSE; try tight layout or set fig size 
import matplotlib.pyplot as plt
num_feats_plot=30#len(labels)
fig, ax = plt.subplots()
ax.plot(RMSE_train[0:num_feats_plot], linestyle='--', marker='o', color='b')
ax.plot(RMSE_test[0:num_feats_plot], linestyle='--', marker='o', color='r')
ax.set_xticklabels(labels[0:num_feats_plot]);
plt.xticks(list(range(0,num_feats_plot)),rotation = 45,ha='right'); # Rotates X-Axis Ticks by 45-degrees, ha='right' is used to make rotated labels show up in a clean format
# plt.xticks(list(range(0,num_feats_plot)),rotation = 90); # Rotates X-Axis Ticks by 45-degrees
ax.set_ylabel('RMSE')
ax.legend(['train','test']);
# increase fig size a bit
fig = plt.gcf()
fig.set_size_inches(14, 7) 
# remind ourselves of train/test error for top-performing predictor variable
print(RMSE_train[0])
print(RMSE_test[0])

# add title, colomn chart, maybe start y-axis at zero
```

    <ipython-input-34-637d614428be>:7: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(labels[0:num_feats_plot]);
    

    45534.349409507675
    44762.77229823456
    


    

    


### Discuss results



```python
# print and look at descriptions of top 5 features
best_feats_combined=labels[0:num_feats_plot]
print(best_feats_combined)

# OverallQual: Rates the overall material and finish of the house
# GarageCars: Size of garage in car capacity
# YearBuilt: Original construction date
# FullBath: Full bathrooms above grade (i.e., not in the basement)
# GarageArea: Size of garage in square feet

# Also see here for more thorough documentation regarding the feature set: 
# https://www.openml.org/d/42165
```

    3                OverallQual
    12                 GrLivArea
    20                GarageCars
    9                   1stFlrSF
    8                TotalBsmtSF
    21                GarageArea
    15                  FullBath
    1                  YearBuilt
    171         Foundation_PConc
    18              TotRmsAbvGrd
    2               YearRemodAdd
    19                Fireplaces
    81      Neighborhood_NridgHt
    191        GarageType_Detchd
    170        Foundation_CBlock
    187        GarageType_Attchd
    22                WoodDeckSF
    5                 BsmtFinSF1
    167          MasVnrType_None
    39           MSSubClass_60.0
    23               OpenPorchSF
    202             SaleType_New
    53               MSZoning_RM
    210    SaleCondition_Partial
    10                  2ndFlrSF
    80      Neighborhood_NoRidge
    146      Exterior1st_VinylSd
    162      Exterior2nd_VinylSd
    168         MasVnrType_Stone
    16                  HalfBath
    Name: Predictor Variable, dtype: object
    

#### Exercise: Discussion of results
**1. How comparable is the train error to the test error for each feature? Do these results indicate overfitting?** 
- For most predictors/features, the test set error is very comparable to train set error. These predictors do not seem to encounter overfitting.

- For a handful of the predictors, we see a very large difference between train set and test set error
Test set error is between .02 to 1.75% larger than train set error. This suggests that we have successfully avoided overfitting.
    
**2. Which feature appears to perform the best in predicting housing prices?** 

OverallQual
    
**3. Write a sentence summarizing how to interpret the test RMSE for the best predictor.** 

On average, the overallQual feature predicts housing prices within +/- $44,762 from the true price 


```python
# View data frame and do exercise...
train_test_ratio = (df_model_err['Test RMSE']-df_model_err['Train RMSE'])/df_model_err['Train RMSE']*100
plt.hist(abs(train_test_ratio))
print(abs(train_test_ratio).min())
print(abs(train_test_ratio).max())


```

    0.02905866309520297
    174.37966498830312
    


    

    


## Hypothesis testing of univariate models


#### Hypotheses in linear modeling
* H_0 (Null hypothesis): m = 0 (i.e., slope is flat)
* H_A (Alternative hypothesis): m != 0 (i.e.., slope is not completely flat) 

#### The 4 Assumptions for Linear Regression Hypothesis Testing
1. Linearity: There is a linear relation between Y and X
2. Normality: The error terms (residuals) are normally distributed
3. Homoscedasticity: The variance of the error terms is constant over all X values (homoscedasticity)
    - calculate residuals and show their distribution
    - build an ad hoc plot to test normality using a qq-plot
    - Shapiro-Wilk Test
4. Independence: The error terms are independent




```python

```

## Fit multivariate model using all predictor vars

#### Predictive Models VS Interpretable Models
* **Interpretable models**: Models trained with linear regression are the most interpretable kind of regression models available - meaning it’s easier to take action from the results of a linear regression model. However, if the assumptions are not satisfied, the interpretation of the results will not always be valid. This can be very dangerous depending on the application.

#### Assumptions of multivariate regression (for statistical/hypothesis testing)
1. Independence: All observations are independent
2. Linearity: The relationship between the dependent variable and the independent variables should be linear

    a. **Note**: In practice, linear models are often used to model nonlinear relationships due to complexity (number of model parameters/coefs that need to be estimated) of nonlinear models. When using a linear model to model nonlinear relationships, it usually best to use resulting model for predictive purposes only. 
3. Normality: For each value of the dependent variable, the distribution of the dependent variable must be normal.
4. Homoscedasticity: The residuals of a good model should be normally and randomly distributed i.e. the unknown does not depend on X ("homoscedasticity")


```python
print(len(labels)) 
```

    213
    


```python
help(fit_eval_model)
```

    Help on function fit_eval_model in module __main__:
    
    fit_eval_model(X_train, y_train, X_test, y_test, predictor_vars, model_type)
        This function uses the predictor vars specified by predictor_vars to predict housing price. Function returns RMSE for both train and test data
    
    


```python
# fit model using all features/predictors available
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'unregularized')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    # model coefs = 214
    Train RMSE: 21981.654614715466
    Test RMSE: 3562241001.482347
    (Test-Train)/Train: 16205418%
    

### Discuss
Is this a good model? Does this model encounter overfitting?

Flesh this out. How many features, how many observations, how many model coefs

## Regularized regression: ridge, lasso, elastic net


### Ridge and RidgeCV
- Show ridge optimization equation
- Default CV is Leave-One-Out. In this form of CV, all samples in the data except for one are used as the inital training set. The left out sample is used a validation set.
- One alpha value used for entire model; larger alphas give more weight to the penalty/regularization term of the loss function

Edit function below to use multiple regression techniques (add model_type input)






```python
# edit train_linear_model to train ridge models as well
def train_linear_model(X_train, y_train, model_type):
  if model_type == "unregularized":
    reg = LinearRegression().fit(X_train,y_train)
  elif model_type == 'ridge':
    reg = RidgeCV(alphas=[1e-3,1e-2,1e-1,1,10,100,1000], store_cv_values=True).fit(X_train,y_train)
    print(reg.cv_values_.shape) # num_datapoints x num_alphas
    print(np.mean(reg.cv_values_, axis=0))
    print(reg.alpha_)
  else:
    raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)

  # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
  print('# model coefs = ' + str(len(reg.coef_)+1))

  return reg


```


```python
# import sklearn's ridge model with built-in cross-validation
from sklearn.linear_model import RidgeCV 

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'ridge')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    (978, 7)
    [1.01586692e+09 1.01401918e+09 9.99400573e+08 9.57029390e+08
     9.43452552e+08 1.02279420e+09 1.21826389e+09]
    10.0
    # model coefs = 214
    Train RMSE: 25463.82775189401
    Test RMSE: 39003.787373887266
    (Test-Train)/Train: 53%
    

- What is the model's train and test error? How does this compare to the unregularized model we fit using all predictor variables? How does this model compare to the best univariate model we fit?
  - The ridge model does much better (i.e., in terms of Test RMSE) than the unregularized model that uses all predictor vars.
  - Unregularized_all_predictors_testRMSE: 3562241001
  - Unregularized_best_univariate_testRMSE: 48243
  - Regularized_all_predictors_testRMSE: 39004

- What alpha value was selected using RidgeCV? Is it a lower or higher value? What does this value tell you about the model?
  - This model is highly regularized/penalized since it has a large alpha value



### LASSO
- explain why there's a random state param in LASSO but not ridge



```python
# edit train_linear_model to train ridge models as well
def train_linear_model(X_train, y_train, model_type):
  if model_type == "unregularized":
    reg = LinearRegression().fit(X_train,y_train)
  elif model_type == 'ridge':
    reg = RidgeCV(alphas=[1e-3,1e-2,1e-1,1,10,100,1000], store_cv_values=True).fit(X_train,y_train)
    print(reg.cv_values_.shape) # num_datapoints x num_alphas
    print(np.mean(reg.cv_values_, axis=0))
    print(reg.alpha_)
  elif model_type == 'lasso':
    reg = LassoCV(random_state=0, alphas=[1e-3,1e-2,1e-1,1,10,100,1000], max_iter=100000, tol=1e-3).fit(X_train,y_train)
    print(reg.alpha_)
    print(reg.alphas_)

  else:
    raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)

  # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
  print('# model coefs = ' + str(len(reg.coef_)+1))

  return reg


```


```python
# import sklearn's lasso model with built-in cross-validation
from sklearn.linear_model import LassoCV 

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'lasso')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 8284193614.777222, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 39790829366.316284, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 38718390339.6676, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 38647786272.54297, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 38613966518.50275, tolerance: 4900223000.335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 51724109695.34778, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 49532515142.88132, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 49422113573.12036, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 49474867747.30087, tolerance: 4985081801.251732
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 7255607704.328613, tolerance: 5231563918.29538
      positive,
    

    100.0
    [1.e+03 1.e+02 1.e+01 1.e+00 1.e-01 1.e-02 1.e-03]
    # model coefs = 214
    Train RMSE: 23844.411315377245
    Test RMSE: 41216.23320716389
    (Test-Train)/Train: 73%
    

Add elastic net option to function


```python
# edit train_linear_model to train ridge models as well
def train_linear_model(X_train, y_train, model_type):
  if model_type == "unregularized":
    reg = LinearRegression().fit(X_train,y_train)
  elif model_type == 'ridge':
    reg = RidgeCV(alphas=[1e-3,1e-2,1e-1,1,10,100,1000], store_cv_values=True).fit(X_train,y_train)
    print(reg.cv_values_.shape) # num_datapoints x num_alphas
    print(np.mean(reg.cv_values_, axis=0))
    print('alpha:', reg.alpha_)
  elif model_type == 'lasso':
    reg = LassoCV(random_state=0, alphas=[1e-3,1e-2,1e-1,1,10,100,1000], max_iter=100000, tol=1e-3).fit(X_train,y_train)
    print('alpha:', reg.alpha_)
    print('alphas:', reg.alphas_)
  elif model_type == 'elastic':
    reg = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],alphas=[1e-5,1e-4,1e-3,1e-2,1e-1,1,10]).fit(X_train,y_train)
    print('alpha:', reg.alpha_)
    print('l1_ratio:', reg.l1_ratio_)
  else:
    raise ValueError('Unexpected model_type encountered; model_type = ' + model_type)

  # print number of estimated model coefficients. Need to add one to account for y-intercept (not included in reg.coef_ call)
  print('# model coefs = ' + str(len(reg.coef_)+1))

  return reg


```


```python
from sklearn.linear_model import ElasticNetCV

# fit model using multivariate_model_feats and ridge regression
RMSE_train, RMSE_test = fit_eval_model(X_train, y_train, X_test, y_test, labels, 'elastic')
```

    # of predictor vars = 213
    # of train observations = 978
    # of test observations = 482
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 466501538000.58923, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 403270137725.9388, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 301577300752.58386, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 230081693672.962, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 191261132823.84335, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180015211573.03418, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178123858792.64304, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 517170940383.7549, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 455149703947.2829, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 356760757816.2706, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 271287951936.91687, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 208621382681.19177, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179934405122.62006, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174598046989.29358, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 497139127553.7737, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 439200852649.58246, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 339907195700.6849, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 258953594195.75394, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 200882179797.22876, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173374398849.39014, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168308116883.32767, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 531036794639.032, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 469827343258.0895, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 368933133080.9624, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 280605522374.60645, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 213947936192.50116, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 182087724825.03278, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176055602385.14227, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 572629384126.2748, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 501555480332.6684, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 391631272775.97107, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 298678305300.5673, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 227982526042.6555, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 195157942278.29935, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 189024550168.52182, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 444949892409.9793, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 365094575321.7053, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 271536336788.16092, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 211633099311.5671, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 186635909873.2761, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179121358447.05002, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178021230383.19012, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 493368405846.16003, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 414829399434.4205, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 322242581988.8907, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 246597333224.66333, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 196334081147.4532, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177416221956.2069, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174308462866.78204, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 480198849240.9201, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 405428439002.9113, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 309501576674.2682, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 236001205429.3958, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190783090645.77213, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 170975409642.3658, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168035191309.2294, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 508714548362.01385, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 430015798460.9209, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 331451029685.684, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 253623789109.70438, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 201491454104.8407, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179241665013.0187, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175728390202.92365, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 547814708459.1266, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 460655259599.63354, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 357768526357.03766, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 272796444290.12274, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 214743064981.45016, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 192268977910.40497, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188690555192.06982, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 416677230542.26807, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 326470938261.0464, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 244818755605.2463, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 196565697156.34995, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 183909103175.64297, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178646713525.2646, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177969529190.60886, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 462432485621.19946, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 373224181082.7571, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 289057115458.06903, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 224721439217.2333, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190549133808.97656, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176074535061.96317, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174162572900.09378, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 457774371126.0421, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 371054861296.9792, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 281551992503.0566, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 215505859575.48795, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 183583514120.1759, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 169702772421.62424, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167897783053.45456, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 479421550321.16235, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 388323215242.1754, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 296292818625.5564, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 228663127248.45554, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 194020395148.21582, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177724632997.83142, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175563562620.36926, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 514485326942.4092, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 421611389032.36316, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 326924241461.59247, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 250711795302.28946, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 207239240911.33615, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190726062457.20798, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188522236475.86008, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 311797142313.5558, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 209862077232.94275, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179799225897.31848, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 162976250066.2721, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180226399106.9014, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178149146521.00806, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177917216985.7658, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 348771325443.47424, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 262676140169.38663, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 213606238041.7038, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 169325997730.12842, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180532958271.4085, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174670065202.66556, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174015846743.25925, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 369288213632.16943, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 286971076623.6909, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 215203533299.0186, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167078183835.83978, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173945243870.41464, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168375348268.6882, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167759303838.29395, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 369631488837.6472, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 274426855741.68402, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 212201514272.23502, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 154631300697.99414, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 182763227707.45428, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176136919103.90308, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175397761945.90024, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 399732561591.90906, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 325717700010.3299, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 256351885785.4635, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 201853296477.35535, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 195843376102.29, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 189107464528.57724, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188352732190.63144, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 220081626718.3236, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 155730175381.8516, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 154449120821.4552, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 181827917422.58365, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179122177438.0314, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178020909027.6598, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177904013510.02463, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 248813839151.22867, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 184846475897.15, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 156427193949.10315, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 163724362231.61304, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177417710095.71313, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174308423162.75894, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173978973823.79193, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 308635567974.98413, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 227821113321.46594, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 172002330293.1055, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188712555568.589, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 170975235250.40717, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168034122123.34964, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167724165476.4443, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 270167884662.31128, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 185075807794.65875, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 150140928105.46667, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 184244212674.26703, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 179243266269.3932, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175728272974.75, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175356095927.54163, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 289957702789.10443, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 262688983586.6175, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 211292658771.88013, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 194232544058.9558, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 192270712977.04324, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188690421311.04477, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188310203297.35104, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 80443289126.99353, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 33940528950.43747, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 93696965405.23439, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180202740645.64984, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 178146838446.83603, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177914128720.72086, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177890539075.50424, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 73876773879.9613, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 41155109697.85321, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 30215275736.665314, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 180533254979.0658, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174670642077.96503, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174014803401.21664, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173948413577.51135, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168155841759.9858, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 124721798412.78943, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 96673988127.99841, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173936407575.40323, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 168369125621.13373, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167752779273.1091, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167690449923.0291, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 88457878160.88037, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 44078712149.98291, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 43413779781.612, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 182757574338.25684, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176136877100.2423, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175396024287.3029, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175321047229.12817, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 194579646116.2381, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 154082096434.86902, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 124051165230.44101, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 195842065111.45575, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 189106603639.84344, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188350172728.8122, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188273563349.11105, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 69964040921.6865, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 175528407923.5064, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176837080816.7248, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176867290496.7877, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177240763902.08405, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 177003775798.41608, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176997329865.3684, tolerance: 490022300.0335935
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 11322111222.890259, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 88359906052.406, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 156822284363.6528, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173347208032.45923, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173349890034.86722, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173360600099.71875, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 173369456092.77753, tolerance: 462319524.2429815
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 114946317314.01196, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 169417380573.14777, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 166843406276.167, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167022131326.93918, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167004371179.29108, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167002615777.21475, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 167002447844.92758, tolerance: 459829356.68389124
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 40265289368.52884, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 176974601700.9984, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174920867566.5073, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174921819973.986, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174901178365.3262, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174899116163.26465, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 174898918506.41833, tolerance: 498508180.12517315
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 81923366998.92291, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 190410356419.97, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188360311598.58383, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188204026079.0411, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188181829342.89905, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188179608568.28705, tolerance: 523156391.829538
      positive,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:644: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 188179387336.04922, tolerance: 523156391.829538
      positive,
    

    0.1
    0.95
    # model coefs = 214
    Train RMSE: 24561.74964572943
    Test RMSE: 39333.83728843832
    (Test-Train)/Train: 60%
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.409e+11, tolerance: 6.087e+08
      coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# Diabetes dataset

# from sklearn import datasets
# example datasets from sklean: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes
# iris_X, iris_y = datasets.load_iris(return_X_y=True)
# more info on diabetes dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
# diabetes = datasets.load_diabetes(return_X_y=False,as_frame=False)
# print(type(diabetes))
# feat_names=diabetes['feature_names']
# print(feat_names)
# data=diabetes['data']
# target=diabetes['target'] # the target is a quantitative measure of disease progression one year after baseline
# print(data.shape)
# print(target.shape)
# print(diabetes_X.shape) # 442 observations, 10 features
# diabetes_y

# California housing dataset

# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()
# # housing
# feat_names=housing['feature_names']
# print(feat_names)
# print(len(feat_names))
```
