---
title: Model validity - relevant predictors
teaching: 45
exercises: 2
keypoints:
- "All models are wrong, but some are useful."
- "Before reading into a model's estimated coefficients, modelers must take care to account for essential predictor variables"
- "Models that do not account for essential predictor variables can produce distorted pictures of reality due to omitted variable bias and confounding effects."
objectives:
- "Understand the importance of including relevant predictors in a model."
questions:
- "What are the benfits/costs of including additional predictors in a regression model?"
---

### Model Validity And Interpretation
While using models strictly for predictive purposes is a completely valid approach for some domains and problems, researchers typically care more about being able to interpret their models such that interesting relationships between predictor(s) and target can be discovered and measured. When interpretting a linear regression model, we can look at the model's estimated coefficients and p-values associated with each predictor to better understand the model. The coefficient's magnitude can inform us of the effect size associated with a predictor, and the p-value tells us whether or not a predictor has a consistent (statistically significant) effect on the target.

**Before we can blindly accept the model's estimated coefficients and p-values, we must answer three questions that will help us determine whether or not our model is valid.**

#### Model Validity Assessments
1. **Accounting for relevant predictors**: Have we included all relevant predictors in the model?
2. **Bias/variance or under/overfitting**: Does the model capture the variability of the target variable well? Does the model generalize well?
3. **Model assumptions**: Does the fitted model follow the 5 assumptions of linear regression?

We will discuss the first two assessments in detail throughout this episode.

### 1. Relevant predictors

> ## Benefits and drawbacks of including all relevant predcitors
> What do you think might be some benefits of including all relevant predictors in a model that you intend to use to **explain** relationships? Are there any drawbacks you can think of?
> > ## Solution
> >
> > Including all relevant predictor variables in a model is important for several reasons:
> > 
> > 1. **Improving model interpretability**: Leaving out relevant predictors can result in *model misspecification*. Misspecification refers to a situation where the model structure or functional form does not accurately reflect the underlying relationship between the predictors and the outcome. If a relevant predictor is omitted from the model, the coefficients of the remaining predictors may be biased. This occurs because the omitted predictor may have a direct or indirect relationship with the outcome variable, and its effect is not accounted for in the model. Consequently, the estimated coefficients of other predictors may capture some of the omitted predictor's effect, leading to biased estimates.
> > 
> > 2. **Improving predicitive accuracy and reducing residual variance**: Omitting relevant predictors can increase the residual variance in the model. Residual variance represents the unexplained variation in the outcome variable after accounting for the predictors in the model. If a relevant predictor is left out, the model may not fully capture the systematic variation in the data, resulting in larger residuals and reduced model fit. While it is true that, in a research setting, we typically care more about being able to interpret our model than being able to perfectly predict the target variable, a model that severely underfits is still a cause for concern since the model won't be capturing the variability of the data well enough to form any conclusions.
> > 
> > 3. **Robustness to future changes**: This benefit only applies to predictive modeling tasks where models are often being fit to new data. By including all relevant predictors, the model becomes more robust to changes in the data or the underlying system. If a predictor becomes important in the future due to changes in the environment or additional data, including it from the start ensures that the model is already equipped to capture its influence.
> > 
> > **Drawbacks to including all relevant predictors:** While one should always aim to include as many relevant predictors as possible, this goal needs to be balanced with overfitting concerns. If we include too many predictors in the model and train on a limited number of observations, the model may simply memorize the nuances/noise in the data rather than capturing the underlying trend in the data.
> {:.solution}
{:.challenge}


#### Example
Let's consider a regression model where we want to evaluate the relationship between FullBath (number of bathrooms) and SalePrice.


```python
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True, parser='auto') #
y=housing['target']
X=housing['data']['FullBath']
print(X.shape)
X.head()
```

    (1460,)





    0    2
    1    2
    2    2
    3    1
    4    2
    Name: FullBath, dtype: int64



It's always a good idea to start by plotting the predictor vs the target variable to get a sense of the underlying relationship.


```python
import matplotlib.pyplot as plt
plt.scatter(X,y,alpha=.3);
# plt.savefig('..//fig//regression//scatterplot_fullBath_vs_salePrice.png', bbox_inches='tight', dpi=300, facecolor='white');
```







<img src="../fig/regression/scatterplot_fullBath_vs_salePrice.png"  align="center" width="30%" height="30%">

Since the relationship doesn't appear to be quite as linear as we were hoping, we will try a log transformation as we did in the previous episode.


```python
import numpy as np
y_log = y.apply(np.log)
plt.scatter(X,y_log, alpha=.3);
# plt.savefig('..//fig//regression//scatterplot_fullBath_vs_logSalePrice.png', bbox_inches='tight', dpi=300, facecolor='white');
```







<img src="../fig/regression/scatterplot_fullBath_vs_logSalePrice.png"  align="center" width="30%" height="30%">

The log transform improves the linear relationship substantially! Next, we will import the statsmodels package which is an R-style modeling package that has some convenient functions for rigorously testing and running stats on linear models.

We'll compare the coefficients estimated from this model to an additional univariate model. To make this comparison more straightforward, we will z-score the predictor. If you don't standardize the scale of all predictors being compared, the coefficient size will be a function of the scale of each specific predictor rather than a measure of each predictor's overall influence on the target.


```python
X = (X - X.mean())/X.std()
X.head()
```




    0    0.789470
    1    0.789470
    2    0.789470
    3   -1.025689
    4    0.789470
    Name: FullBath, dtype: float64



For efficiency, we will skip train/test splits in this episode. Recall that train/test splits aren't as essential when working with only a handful or predictors (i.e., when the ratio between number of training observations and model parameters/coefficients is at least 10).

Fit the model. Since we are now turning our attention towards explanatory models, we will use the statsmodels library isntead of sklearn. Statsmodels comes with a variety of functions which make it easier to interpret the model and ultimately run hypothesis tests. It closely mirrors the way R builds linear models.


```python
import statsmodels.api as sm

# Add a constant column to the predictor variables dataframe
X = sm.add_constant(X)

# Fit the multivariate regression model
model = sm.OLS(y_log, X)
results = model.fit()
```

Let's print the coefs from this model. In addition, we can quickly extract R-squared from the statsmodel model object using...


```python
print(results.params,'\n')
print(results.pvalues,'\n')
print('R-squared:', results.rsquared)
```

    const       12.024051
    FullBath     0.237582
    dtype: float64

    const        0.000000e+00
    FullBath    2.118958e-140
    dtype: float64

    R-squared: 0.3537519976399338


You can also call results.summary() for a detailed overview of the model's estimates and resulting statistics.


```python
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>SalePrice</td>    <th>  R-squared:         </th> <td>   0.354</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.353</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   798.1</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 04 Aug 2023</td> <th>  Prob (F-statistic):</th> <td>2.12e-140</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:53:41</td>     <th>  Log-Likelihood:    </th> <td> -412.67</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1460</td>      <th>  AIC:               </th> <td>   829.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1458</td>      <th>  BIC:               </th> <td>   839.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>
</tr>
<tr>
  <th>const</th>    <td>   12.0241</td> <td>    0.008</td> <td> 1430.258</td> <td> 0.000</td> <td>   12.008</td> <td>   12.041</td>
</tr>
<tr>
  <th>FullBath</th> <td>    0.2376</td> <td>    0.008</td> <td>   28.251</td> <td> 0.000</td> <td>    0.221</td> <td>    0.254</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>51.781</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 141.501</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.016</td> <th>  Prob(JB):          </th> <td>1.88e-31</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.525</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Based on the R-squared, this model explains 35.4% of the variance in the SalePrice target variable.

The model coefficient estimated for the "FullBath" predictor is 0.24. Recall that we fit this model to a log scaled version of the SalePrice. In other words, increasing the FullBath predictor by 1 standard deviation increases the log(SalePrice) by 0.24. While this explanation is completely valid, it is often useful to interpret the coefficient in terms of the original scale of the target variable.

##### Transforming the coefficient to the original scale of the data.
Exponentiate the coefficient to reverse the log transformation. This gives the multiplicative factor for every one-unit increase in the independent variable. In our model (run code below), for every standard devation increase in the predictor, our target variable increases by a factor of about 1.27, or 27%. Recall that multiplying a number by 1.27 is the same as increasing the number by 27%. Likewise, multiplying a number by, say 0.3, is the same as decreasing the number by 1 – 0.3 = 0.7, or 70%.


```python
np.exp(results.params[1]) # First param is the estimated coef for the y-intercept / "const". The second param is the estimated coef for FullBath.
```




    1.2681792421553808



When transformed to the original data scale, this coefficient tells us that increasing bathroom count by 1 standard deviation increases the sale price, on average, by 27%. While bathrooms are a very hot commodity to find in a house, they likely don't deserve this much credit. Let's do some further digging by comparing another predictor which likely has a large impact on SalePrice — the total square footage of the house (excluding the basement).


```python
X=housing['data']['GrLivArea']
plt.scatter(X, y_log);
# plt.savefig('..//fig//regression//scatterplot_GrLivArea_vs_logSalePrice.png', bbox_inches='tight', dpi=300, facecolor='white');
```







<img src="../fig/regression/scatterplot_GrLivArea_vs_logSalePrice.png"  align="center" width="30%" height="30%">

As before, we will z-score the predictor. This is a critical step when comparing coefficient estimates since the estimates are a function of the scale of the predictor.


```python
X = (X - X.mean())/X.std()
X.head()
```




    0    0.370207
    1   -0.482347
    2    0.514836
    3    0.383528
    4    1.298881
    Name: GrLivArea, dtype: float64



Fit the model and print coefs/R-squared.


```python
# Add a constant column to the predictor variables dataframe
X = sm.add_constant(X)
print(X.head())
# Fit the multivariate regression model
model = sm.OLS(y_log, X)
results = model.fit()
print(results.params)
print('R-squared:', results.rsquared)
```

       const  GrLivArea
    0    1.0   0.370207
    1    1.0  -0.482347
    2    1.0   0.514836
    3    1.0   0.383528
    4    1.0   1.298881
    const        12.024051
    GrLivArea     0.279986
    dtype: float64
    R-squared: 0.49129817224671934


Based on the R-squared, this model explains 49.1% of the variance in the target variable (higher than FullBath which is to be expected). Let's convert the coef to the original scale of the target data before reading into it.


```python
np.exp(results.params[1]) # First param is the estimated coef for the y-intercept / "const". The second param is the estimated coef for FullBath.
```




    1.3231118984358705



For every one standard devation increase in the predictor (GrLivArea), our target variable (SalePrice) increases by a factor of about 1.32, or 32%.

Let's compare our findings with a multivariate regression model that includes both predictors.


```python
predictors = ['GrLivArea', 'FullBath']
X=housing['data'][predictors]
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
      <th>GrLivArea</th>
      <th>FullBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1710</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1786</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1717</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2198</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



##### Standardization


```python
X = (X - X.mean())/X.std()
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
      <th>GrLivArea</th>
      <th>FullBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.370207</td>
      <td>0.789470</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.482347</td>
      <td>0.789470</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.514836</td>
      <td>0.789470</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.383528</td>
      <td>-1.025689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.298881</td>
      <td>0.789470</td>
    </tr>
  </tbody>
</table>
</div>



Add constant for modeling y-intercept


```python
# Fit the multivariate regression model
X = sm.add_constant(X)
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
      <th>const</th>
      <th>GrLivArea</th>
      <th>FullBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.370207</td>
      <td>0.789470</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.482347</td>
      <td>0.789470</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.514836</td>
      <td>0.789470</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.383528</td>
      <td>-1.025689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.298881</td>
      <td>0.789470</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = sm.OLS(y_log, X)
results = model.fit()
print(results.params)
print('R-squared:', results.rsquared)
```

    const        12.024051
    GrLivArea     0.216067
    FullBath      0.101457
    dtype: float64
    R-squared: 0.530204241994317


> ## Comparing results
> 1. How does the R-squared of this model compare to the univariate models? Is the variance explained by the multivariate model equal to the sum of R-squared of each univariate model? Why or why not?
> 2. Convert the coefficients to the original scale of the target variable as we did earlier in this episode. How much does SalePrice increase with a 1 standard deviation increase in each predictor?
> 3. How do the coefficient estimates compare to the univariate model estimates? Is there any difference? If so, what might be the cause?
> 
> > ## Solution
> >
> > **How does the R-squared of this model compare to the univariate models? Is the variance explained by the multivariate model equal to the sum of R-squared of each univariate model? Why or why not?**
> > 
> > The R-squared value in the multivariate model (53.0%) is somewhat larger than each of the univariate models (GrLivArea=49.1%, FullBath=35.4%) which illustrates one of the benefits of includign multiple predictors. When we add the R-squared values of the univariate models, we get 49.1 + 35.4 = 84.5%. This value is much larger than what we observe in the multivariate model. The reason we can't simply add the R-squared values together is because each univariate model fails to account for at least one relevant predictor. When we omit one of the predictors, the model assumes the observed relationship is only due to the remaining predictor. This causes the impact of each individual predictor to appear inflated (R-squared and coef magnitude) in the univariate models.
> > 
> > **Convert the coefficients to the original scale of the target variable as we did earlier in this episode. How much does SalePrice increase with a 1 standard deviation increase in each predictor?**
> > 
> > First we'll convert the coefficients to the original scale of the target variable using the exp() function (the inverse of log).
> > 
> > ~~~
> > print('GrLivArea:', np.exp(.216))
> > print('FullBath:', np.exp(.101))
> > ~~~
> > {: .language-python}
> > ~~~
> > GrLivArea: 1.2411023790006717
> > FullBath: 1.1062766417634236
> > ~~~
> > {: .output}
> > 
> > Based on these results, increasing the GrLivArea by 1 standard deviation increases SalePrice by 24.1% (univariate = 32.3%), while increasing FullBath by 1 standard deviation increases SalePrice by only 10.6% (univariate = 26.8%).
> > 
> > **How do the coefficient estimates compare to the univariate model estimates? Is there any difference? If so, what might be the cause?**
> > 
> > When using a multivariate model, the coefficients were reduced to a considerable degree compared to the univariate models. Why does this happen? Both SalePrice and FullBath linearly relate to SalePrice. If we model SalePrice while considering only one of these effects, the model will think that only one predictor is doing the work of multiple predictors. We call this effect *omitted-variable bias* or *omitted-predictor bias*. Omitted-variable bias leads to *model misspecification*, where the model structure or functional form does not accurately reflect the underlying relationship between the predictors and the outcome. If you want a more truthful model, it is critical that you include as many relevant predictors as possible.
> > 
> {:.solution}
{:.challenge}


### Including ALL predictors - overfitting concerns
While researchers should always strive to include as many many relevant predictors as possible, this must also be balanced with overfitting concerns. That is, it is often the case that SOME of the relevant predictors must be left out in order to ensure that overfitting does not occur. If we include too many predictors in the model and train on a limited number of observations, the model may simply memorize the nuances/noise in the data rather than capturing the underlying trend in the data.

Let's see how this plays out with the Ames housing dataset.

We'll first load and prep the full high-dimensional dataset. The following helper function...
1. loads the full Ames housing dataset
2. Encodes all categorical predictors appropriately (we'll discuss this step in detail in the next episode)
3. Removes sparse predictors with little to no variability (we'll discuss this step in detail in the next episode)
3. log scales the target variable, SalePrice
4. train/test splits the data


```python
# the below code will be converted to a helper function (prep_full_data or something named similar)
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True, parser='auto') #
y=housing['target']
X=housing['data']
X.head()
import numpy as np
y_log = np.log(y)
from preprocessing import encode_predictors_housing_data
X_encoded = X.copy(deep=True)
X_encoded = encode_predictors_housing_data(X_encoded)
X_encoded.head()

from preprocessing import remove_bad_cols
X_encoded_good = remove_bad_cols(X_encoded, .95)
X_encoded_good.head()

print(X_encoded_good.shape)
print(y.shape)
```

    # of columns removed: 133
    Columns removed: ['GarageType_2Types', 'GarageType_Basment', 'GarageType_CarPort', 'RoofStyle_Flat', 'RoofStyle_Gambrel', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_WdShing', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'MiscFeature_Gar2', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_Oth', 'RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', "MSZoning_'C (all)'", 'MSZoning_FV', 'MSZoning_RH', "Exterior2nd_'Brk Cmn'", "Exterior2nd_'Wd Shng'", 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_ImStucc', 'Exterior2nd_Other', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_Crawfor', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NPkVill', 'Neighborhood_NoRidge', 'Neighborhood_SWISU', 'Neighborhood_SawyerW', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'Alley_Grvl', 'Alley_Pave', 'Utilities_AllPub', 'Utilities_NoSeWa', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'LotConfig_FR2', 'LotConfig_FR3', 'Condition1_Artery', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'MasVnrType_BrkCmn', 'KitchenAbvGr', 'LotFrontage', 'PoolArea', 'GarageYrBlt', 'LowQualFinSF', 'MasVnrArea', '3SsnPorch', 'Street']
    (1460, 82)
    (1460,)



```python
X_encoded_good.head()
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
      <th>GarageType_Attchd</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_Detchd</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Hip</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_50</th>
      <th>MSSubClass_60</th>
      <th>MSSubClass_120</th>
      <th>Exterior1st_'Wd Sdng'</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>BsmtFinSF1</th>
      <th>ScreenPorch</th>
      <th>GrLivArea</th>
      <th>MoSold</th>
      <th>OverallQual</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>706</td>
      <td>0</td>
      <td>1710</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>978</td>
      <td>0</td>
      <td>1262</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>486</td>
      <td>0</td>
      <td>1786</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>216</td>
      <td>0</td>
      <td>1717</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>655</td>
      <td>0</td>
      <td>2198</td>
      <td>12</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



Next, we will perform a train/test split so that we can test for overfitting effects. We will limit the size of the training set here to illustrate a point. Typically, you'd want to use around 30% of the data for testing.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded_good, y_log, test_size=0.90)
```

#### Zscoring all predictors
Since we're now working with multiple predictors, we will zscore our data such that we can compare coefficient estimates across predictors.

There is some additional nuance to this step when working with train/test splits. For instance, you might wonder which of the following procedures is most appropriate...

1. Zscore the full dataset prior to train/test splittng
2. Zscore the train and test sets separately, using each subset's mean and standard deviation

As it turns out, both are incorrect. Instead, it is best to use only the training set to derive the means and standard deviations used to zscore both the training and test sets. The reason for this is to prevent **data leakage**, which can occur if you calculate the mean and standard deviation for the entire dataset (both training and test sets) together. This would give the test set information about the distribution of the training set, leading to biased and inaccurate performance evaluation. The test set should be treated as unseen data during the preprocessing steps.

#### To standardize your data correctly:

1. Calculate the mean and standard deviation of each feature on the training set.
2. Use these calculated means and standard deviations to standardize both the training and test sets.


```python
import pandas as pd

def zscore(df: pd.DataFrame, train_means: pd.Series, train_stds: pd.Series) -> pd.DataFrame:
    """return z-scored dataframe"""
    return (df - train_means) / train_stds
```


```python
# get means and stds
train_means = X_train.mean()
train_stds = X_train.std()
```


```python
X_train_z = zscore(df=X_train, train_means=train_means, train_stds=train_stds)
X_test_z = zscore(df=X_test, train_means=train_means, train_stds=train_stds)
X_train_z.head()
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
      <th>GarageType_Attchd</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_Detchd</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Hip</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_50</th>
      <th>MSSubClass_60</th>
      <th>MSSubClass_120</th>
      <th>Exterior1st_'Wd Sdng'</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>BsmtFinSF1</th>
      <th>ScreenPorch</th>
      <th>GrLivArea</th>
      <th>MoSold</th>
      <th>OverallQual</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>2.84066</td>
      <td>-0.517416</td>
      <td>-0.255428</td>
      <td>-0.419767</td>
      <td>...</td>
      <td>-1.145210</td>
      <td>0.216482</td>
      <td>0.599222</td>
      <td>-0.270798</td>
      <td>-0.392862</td>
      <td>1.171898</td>
      <td>-0.841802</td>
      <td>1.159982</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>403</th>
      <td>-1.339892</td>
      <td>5.292150</td>
      <td>-0.538549</td>
      <td>-1.774741</td>
      <td>1.808744</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>-0.419767</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-1.050998</td>
      <td>-0.270798</td>
      <td>1.261357</td>
      <td>0.106978</td>
      <td>1.277217</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>138</th>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>-0.419767</td>
      <td>...</td>
      <td>1.922692</td>
      <td>1.480736</td>
      <td>0.454941</td>
      <td>-0.270798</td>
      <td>0.506769</td>
      <td>1.881844</td>
      <td>1.277217</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>1051</th>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>1.281879</td>
      <td>-0.34962</td>
      <td>-0.517416</td>
      <td>-0.255428</td>
      <td>-0.419767</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-1.050998</td>
      <td>-0.270798</td>
      <td>-0.477318</td>
      <td>1.171898</td>
      <td>0.570877</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>221</th>
      <td>-1.339892</td>
      <td>5.292150</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>-0.419767</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-1.050998</td>
      <td>-0.270798</td>
      <td>1.268701</td>
      <td>1.881844</td>
      <td>-0.135462</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



#### Fit the model and measure train/test performance


```python
# Fit the multivariate regression model
X_train_z = sm.add_constant(X_train_z)
X_train_z.head()
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
      <th>const</th>
      <th>GarageType_Attchd</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_Detchd</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Hip</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_50</th>
      <th>MSSubClass_60</th>
      <th>MSSubClass_120</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>BsmtFinSF1</th>
      <th>ScreenPorch</th>
      <th>GrLivArea</th>
      <th>MoSold</th>
      <th>OverallQual</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>2.84066</td>
      <td>-0.517416</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>-1.145210</td>
      <td>0.216482</td>
      <td>0.599222</td>
      <td>-0.270798</td>
      <td>-0.392862</td>
      <td>1.171898</td>
      <td>-0.841802</td>
      <td>1.159982</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>403</th>
      <td>1.0</td>
      <td>-1.339892</td>
      <td>5.292150</td>
      <td>-0.538549</td>
      <td>-1.774741</td>
      <td>1.808744</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-1.050998</td>
      <td>-0.270798</td>
      <td>1.261357</td>
      <td>0.106978</td>
      <td>1.277217</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>138</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>1.922692</td>
      <td>1.480736</td>
      <td>0.454941</td>
      <td>-0.270798</td>
      <td>0.506769</td>
      <td>1.881844</td>
      <td>1.277217</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>1051</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>1.281879</td>
      <td>-0.34962</td>
      <td>-0.517416</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-1.050998</td>
      <td>-0.270798</td>
      <td>-0.477318</td>
      <td>1.171898</td>
      <td>0.570877</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>221</th>
      <td>1.0</td>
      <td>-1.339892</td>
      <td>5.292150</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-1.050998</td>
      <td>-0.270798</td>
      <td>1.268701</td>
      <td>1.881844</td>
      <td>-0.135462</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>



We'll add the constant to the test set as well so that we can feed the test data to the model for prediction.


```python
X_test_z = sm.add_constant(X_test_z)
X_test_z.head()
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
      <th>const</th>
      <th>GarageType_Attchd</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_Detchd</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Hip</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_50</th>
      <th>MSSubClass_60</th>
      <th>MSSubClass_120</th>
      <th>...</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>BsmtFinSF1</th>
      <th>ScreenPorch</th>
      <th>GrLivArea</th>
      <th>MoSold</th>
      <th>OverallQual</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>CentralAir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>0.559603</td>
      <td>-0.549083</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>1.480736</td>
      <td>0.425634</td>
      <td>-0.270798</td>
      <td>1.142018</td>
      <td>1.881844</td>
      <td>1.277217</td>
      <td>1.159982</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>-1.774741</td>
      <td>1.808744</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>0.216482</td>
      <td>-0.054553</td>
      <td>-0.270798</td>
      <td>0.550832</td>
      <td>0.106978</td>
      <td>-0.135462</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>515</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>-1.774741</td>
      <td>1.808744</td>
      <td>1.281879</td>
      <td>-0.34962</td>
      <td>-0.517416</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>1.480736</td>
      <td>2.186319</td>
      <td>-0.270798</td>
      <td>0.815214</td>
      <td>0.816924</td>
      <td>2.689897</td>
      <td>1.159982</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>526</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>-1.774741</td>
      <td>1.808744</td>
      <td>1.281879</td>
      <td>-0.34962</td>
      <td>-0.517416</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>-1.145210</td>
      <td>-1.047772</td>
      <td>-0.201089</td>
      <td>-0.270798</td>
      <td>-1.189678</td>
      <td>-0.247995</td>
      <td>-0.841802</td>
      <td>-0.856177</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
    <tr>
      <th>527</th>
      <td>1.0</td>
      <td>0.741217</td>
      <td>-0.187665</td>
      <td>-0.538549</td>
      <td>-1.774741</td>
      <td>1.808744</td>
      <td>-0.774762</td>
      <td>-0.34962</td>
      <td>1.919445</td>
      <td>-0.255428</td>
      <td>...</td>
      <td>0.388741</td>
      <td>1.480736</td>
      <td>1.947353</td>
      <td>-0.270798</td>
      <td>2.087549</td>
      <td>1.526871</td>
      <td>1.983557</td>
      <td>1.159982</td>
      <td>-0.22364</td>
      <td>0.239946</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>



Train the model.


```python
model = sm.OLS(y_train, X_train_z)
trained_model = model.fit()
```

Get model predictions on train and test sets.


```python
y_pred_train=trained_model.predict(X_train_z)
y_pred_test=trained_model.predict(X_test_z)
```

Compare train/test R-squared.


```python
from sklearn import metrics
R2_train = metrics.r2_score(y_train, y_pred_train)
R2_test = metrics.r2_score(y_test, y_pred_test)
print(R2_train)
print(R2_test)
```

    0.9665357480273615
    0.7321665998421403


We can see that this model exhibits signs of overfitting. That is, the test set performance is substantially lower than train set performance. Since the model does not generalize well to other datasets — we shouldn't read too much into the model's estimated coefficients and p-values. An overfit model is a model that learns nuances/noise in the training data, and the coefficients/p-values may be biased towards uninteresting patterns in the data (i.e., patterns that don't generalize).

Why do we see overfitting here? Let's quickly calculate the ratio between number of observations used to train the model and number of coefficients that need to be esitmated.


```python
X_train_z.shape[0]/X_train_z.shape[1]
```




    1.7590361445783131



As the number of observations begins to approach the number of model parameters (i.e., coefficients being estimated), the model will simply memorize the training data rather than learn anything useful. As a general rule of thumb, obtaining reliable estimates from linear regression models requires that you have at least 10X as many observations than model coefficients/predictors. The exact ratio may change depending on the variability of your data and whether or not each observation is truly independent (time-series models, for instance, often require much more data since observations are rarely independent).

#### "All models are wrong, but some are useful" - George Box
Because of these opposing forces, it's important to remember the following sage wisdom: **All models are wrong, but some are useful.**.  This famous quote by the statistician George E. P. Box conveys an essential concept in the field of statistics and modeling.

In essence, the phrase means that no model can perfectly capture the complexities and nuances of real-world data and phenomena. Models are simplifications of reality and are, by their nature, imperfect representations. Therefore, all models are considered "wrong" to some extent because they do not fully encompass the entire reality they attempt to describe.

However, despite their imperfections, some models can still be valuable and "useful" for specific purposes. A useful model is one that provides valuable insights, makes accurate predictions, or aids in decision-making, even if it does not perfectly match the underlying reality. The key is to understand the limitations of the model and interpret its results accordingly. Users of the model should be aware of its assumptions, potential biases, and areas where it might not perform well. Skilled data analysts and scientists know how to leverage models effectively, acknowledging their limitations and using them to gain insights or solve problems within the scope of their applicability.

#### Feature selection methods
Throughout this workshop, we will explore a couple of feature (predictor) selection methods that can help you simplify your high-dimensional data — making it possible to avoid overfitting concerns. These methods can involve either (A) mathematically combining features to reduce dimensionality or (B) selecting only the most "interesting" predictors, where the definition of interesting varies based on the exact method of choice.

### Summary
In summary, leaving out relevant predictors can lead to biased coefficient estimates and model misspecification. Without including the most essential predictors, the model will place too much focus on the predictors included and over/underestimate their contributions to the target variable.

In addition, while researchers should strive to include as many relevant predictors in their models as possible, this must be balanced with overfitting concerns. Obtaining good coefficient estimates can become difficult as the number of predictors increases. As a general rule of thumb, obtaining reliable estimates from linear regression models requires that you have at least 10X as many observations than model coefficients/predictors. The exact ratio may change depending on the variability of your data and whether or not each observation is truly independent (time-series models, for instance, often require much more data since observations are rarely independent).

#### Other considerations
So far, we've explored the importance of including relevant predictors and checking for overfitting before we attempt to read too far into the model's estimates. However, recall that there are three critical questions we must ask before we can read too far into our model's estimations
1. **Accounting for relevant predictors**: Have we included all relevant predictors in the model?
2. **Bias/variance or under/overfitting**: Does the model capture the variability of the target variable well? Does the model generalize well?
3. **Model assumptions**: Does the fitted model follow the 5 assumptions of linear regression?

In the next episode, we'll review a handful of assumptions that must be evaluated prior to running any hypothesis tests on a regression model.