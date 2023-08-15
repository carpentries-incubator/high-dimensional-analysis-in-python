---
title: Regularization methods - lasso, ridge, and elastic net
teaching: 45
exercises: 2
keypoints:
- ""
objectives:
- ""
questions:
- "How can LASSO regularization be used as a feature selection method?"
---

## Introduction to the LASSO Model in high-dimensional data analysis
In the realm of high-dimensional data analysis, where the number of predictors begins to approach or exceed the number of observations, traditional regression methods can become challenging to implement and interpret. The Least Absolute Shrinkage and Selection Operator (LASSO) offers a powerful solution to address the complexities of high-dimensional datasets. This technique, introduced by Robert Tibshirani in 1996, has gained immense popularity due to its ability to provide both effective prediction and feature selection.

The LASSO model is a regularization technique designed to combat overfitting by adding a penalty term to the regression equation. The essence of the LASSO lies in its ability to shrink the coefficients of less relevant predictors towards zero, effectively "shrinking" them out of the model. This not only enhances model interpretability by identifying the most important predictors but also reduces the risk of multicollinearity and improves predictive accuracy.

LASSO's impact on high-dimensional data analysis is profound. It provides several benefits:

* Feature Selection / Interpretability: The LASSO identifies and retains the most relevant predictors. With a reduced set of predictors, the model becomes more interpretable, enabling researchers to understand the driving factors behind the predictions.

* Regularization / Dimensionality Reduction: The L1 penalty prevents overfitting by constraining the coefficients, even in cases with a large number of predictors. The L1 penality inherently reduces the dimensionality of the model, making it suitable for settings where the number of predictors is much larger than the sample size.

* Improved Generalization: Related to the above point, LASSO's feature selection capabilities contribute to better generalization and prediction performance on unseen data.

* Data Efficiency: LASSO excels when working with limited samples, offering meaningful insights despite limited observations.

### The L1 penalty
The key concept behind the LASSO is its use of the L1 penalty, which is defined as the sum of the absolute values of the coefficients (parameters) of the model, multiplied by a regularization parameter (usually denoted as λ or alpha).

In the context of linear regression, the L1 penalty can be incorporated into the ordinary least squares (OLS) loss function as follows:

![LASSO Model](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Lasso.png)


Where:

* λ (lambda) is the regularization parameter that controls the strength of the penalty. Higher values of λ lead to stronger regularization and more coefficients being pushed towards zero.
* βi is the coefficient associated with the i-th predictor.

The L1 penalty has a unique property that it promotes sparsity. This means that it encourages some coefficients to be exactly zero, effectively performing feature selection. In contrast to the L2 penalty (Ridge penalty), which squares the coefficients and promotes small but non-zero values, the L1 penalty tends to lead to sparse solutions where only a subset of predictors are chosen. As a result, the LASSO automatically performs feature selection, which is especially advantageous when dealing with high-dimensional datasets where many predictors may have negligible effects on the outcome.

## Compare full-dim and LASSO results

### Load full dim, zscored, data
We'll use most of the data for the test set so that this dataset's dimensionality begins to approach the number of observations. Regularization techniques such as LASSO tend to shine when working in this context. If you have plenty of data to estimate each coefficient, you will typically find that an unregularized model performs better.


```python
from preprocessing import prep_fulldim_zdata
X_train_z, X_test_z, y_train, y_test, y = prep_fulldim_zdata(const_thresh= 86, test_size=.93, y_log_scaled=True)
X_train_z.head()
```

    164 columns removed, 51 remaining.
    Columns removed: ['LotFrontage', 'LowQualFinSF', 'ScreenPorch', 'MasVnrArea', 'BsmtHalfBath', 'PoolArea', 'KitchenAbvGr', '3SsnPorch', 'GarageYrBlt', 'BsmtFinSF2', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Partial', 'GarageType_2Types', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_ImStucc', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_WdShing', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'Alley_Grvl', 'Alley_Pave', 'RoofStyle_Flat', 'RoofStyle_Gambrel', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'Foundation_BrkTil', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'Utilities_AllPub', 'Utilities_NoSeWa', "Exterior2nd_'Brk Cmn'", "Exterior2nd_'Wd Sdng'", "Exterior2nd_'Wd Shng'", 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_ImStucc', 'Exterior2nd_Other', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'MasVnrType_BrkCmn', 'MasVnrType_Stone', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'MiscFeature_Gar2', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', "MSZoning_'C (all)'", 'MSZoning_FV', 'MSZoning_RH', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSSubClass_120', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'CentralAir', 'Street']
    




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
      <th>1stFlrSF</th>
      <th>HalfBath</th>
      <th>EnclosedPorch</th>
      <th>YrSold</th>
      <th>Fireplaces</th>
      <th>MoSold</th>
      <th>YearBuilt</th>
      <th>TotalBsmtSF</th>
      <th>OverallCond</th>
      <th>BsmtFullBath</th>
      <th>...</th>
      <th>Exterior2nd_MetalSd</th>
      <th>Exterior2nd_VinylSd</th>
      <th>MasVnrType_BrkFace</th>
      <th>MasVnrType_None</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>LotConfig_Corner</th>
      <th>LotConfig_Inside</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>368</th>
      <td>0.362455</td>
      <td>-0.832549</td>
      <td>-0.252468</td>
      <td>1.549150</td>
      <td>0.611180</td>
      <td>-1.088988</td>
      <td>-0.929703</td>
      <td>0.634862</td>
      <td>0.523613</td>
      <td>-0.834015</td>
      <td>...</td>
      <td>-0.429212</td>
      <td>-0.719212</td>
      <td>1.407264</td>
      <td>-1.097933</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>-0.567003</td>
      <td>0.719212</td>
      <td>1.189355</td>
      <td>-0.582023</td>
    </tr>
    <tr>
      <th>973</th>
      <td>0.807446</td>
      <td>-0.832549</td>
      <td>-0.252468</td>
      <td>0.181069</td>
      <td>-1.073694</td>
      <td>1.931227</td>
      <td>1.259310</td>
      <td>1.045635</td>
      <td>-0.484095</td>
      <td>-0.834015</td>
      <td>...</td>
      <td>-0.429212</td>
      <td>-0.719212</td>
      <td>-0.703632</td>
      <td>-1.097933</td>
      <td>-2.225080</td>
      <td>-0.345968</td>
      <td>1.746369</td>
      <td>-1.376778</td>
      <td>1.189355</td>
      <td>-0.582023</td>
    </tr>
    <tr>
      <th>387</th>
      <td>-0.035256</td>
      <td>-0.832549</td>
      <td>-0.252468</td>
      <td>0.865110</td>
      <td>0.611180</td>
      <td>1.260068</td>
      <td>-0.021056</td>
      <td>0.052077</td>
      <td>0.523613</td>
      <td>1.099383</td>
      <td>...</td>
      <td>2.307012</td>
      <td>-0.719212</td>
      <td>1.407264</td>
      <td>-1.097933</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>-0.567003</td>
      <td>0.719212</td>
      <td>-0.832549</td>
      <td>-0.582023</td>
    </tr>
    <tr>
      <th>816</th>
      <td>-0.360655</td>
      <td>-0.832549</td>
      <td>2.361918</td>
      <td>-1.187011</td>
      <td>0.611180</td>
      <td>0.253330</td>
      <td>-0.929703</td>
      <td>-0.032645</td>
      <td>0.523613</td>
      <td>-0.834015</td>
      <td>...</td>
      <td>-0.429212</td>
      <td>-0.719212</td>
      <td>-0.703632</td>
      <td>0.901873</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>1.746369</td>
      <td>-1.376778</td>
      <td>1.189355</td>
      <td>-0.582023</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>1.369247</td>
      <td>-0.832549</td>
      <td>-0.252468</td>
      <td>0.865110</td>
      <td>0.611180</td>
      <td>-1.760147</td>
      <td>1.300612</td>
      <td>1.543698</td>
      <td>-0.484095</td>
      <td>1.099383</td>
      <td>...</td>
      <td>-0.429212</td>
      <td>1.376778</td>
      <td>-0.703632</td>
      <td>-1.097933</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>-0.567003</td>
      <td>0.719212</td>
      <td>1.189355</td>
      <td>-0.582023</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
print(X_train_z.shape)
```

    (102, 51)
    

## Intro to LassoCV
LassoCV in scikit-learn performs cross-validation to find the best alpha value (lambdas in traditional LASSO equation) from the specified list of alphas. It does this by fitting a separate LASSO regression for each alpha value on each fold of the cross-validation. The alphas parameter determines the values of alpha to be tested.

The LassoCV model doesn't store a reference to all individual models tested during cross-validation. Instead, it stores the coefficients, alpha values, and other relevant information for the best-performing model. By default, the LassoCV model returns the best model that was determined based on cross-validation performance. This best model's coefficients can be accessed using the .coef_ attribute, and the optimal alpha can be accessed using the .alpha_ attribute.

* **max_iter**: This is the maximum number of iterations for which we want the model to run if it doesn’t converge before. The default value is 1000
* **cv**: The number of folds to use during cross-validation
* **alphas**: The alphas you want to evaluate during cross-validation


```python
from sklearn.linear_model import LassoCV
# help(LassoCV)
```

### Specify range of alphas
Specify a range of alpha values. Typically, small alphas work well. However, you don't want to be so close to zero that you get no benefits from regularization (i.e., none of the coefs shrink to zero).


```python
import numpy as np
alphas = np.logspace(-4, 1, 300)
print(alphas[0:10])
max_iter = 100000
cv = 5
```

    [0.0001     0.00010393 0.00010801 0.00011225 0.00011665 0.00012123
     0.00012599 0.00013094 0.00013608 0.00014142]
    

### Call LassoCV



```python
reg = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter, random_state=0)
reg = reg.fit(X_train_z, y_train)
```

### Randomness in LassoCV
LassoCV uses coordinate descent, which is a convex optimization algorithm meaning that it solves for a global optimum (one possible optimal error).

However, during coordinate descent, when multiple features are highly correlated, the algorithm can choose any one of them to update at each iteration. This can lead to some randomness in the selection of features and the order in which they are updated. While coordinate descent itself is deterministic, the order in which the correlated features are selected can introduce variability.

The random state argument in LassoCV allows you to control this randomness by setting a specific random seed. This can be helpful for reproducibility when working with models that involve correlated features. By specifying a random seed, you ensure that the same features will be chosen in the same order across different runs of the algorithm, making the results more predictable and reproducible.

In summary, while coordinate descent is a convex algorithm, the random state argument in LassoCV helps manage the potential randomness introduced by the selection of correlated features during the optimization process.

## Use fit_eval_model() to quickly compare models


```python
from regression_predict_sklearn import fit_eval_model

# Full-dim model
trained_model, error_df = fit_eval_model(y=y, baseline_pred=y.mean(),
               X_train=X_train_z, y_train=y_train,
               X_test=X_test_z, y_test=y_test, 
               predictors=X_train_z.columns,
               metric='RMSE',
               y_log_scaled=True,
               model_type='unregularized',
               include_plots=True, plot_raw=True, verbose=True)

# LASSO
import numpy as np
trained_model, error_df = fit_eval_model(y=y, baseline_pred=y.mean(),
                                         X_train=X_train_z, y_train=y_train,
                                         X_test=X_test_z, y_test=y_test, 
                                         predictors=X_train_z.columns,
                                         metric='RMSE',
                                         y_log_scaled=True,
                                         model_type='LassoCV', alphas=alphas, cv=5, max_iter=100000,
                                         include_plots=True, plot_raw=True, verbose=True)
```

    # of predictor vars = 51
    # of train observations = 102
    # of test observations = 1358
    Baseline RMSE = 79415.29188606751
    Train RMSE = 14835.039689206125
    Holdout RMSE = 112820.25536497564
    (Holdout-Train)/Train: 660%
    


    

    


    
    # of predictor vars = 51
    # of train observations = 102
    # of test observations = 1358
    Baseline RMSE = 79415.29188606751
    Train RMSE = 18937.000079349345
    Holdout RMSE = 56985.86676300235
    (Holdout-Train)/Train: 201%
    


    

    


    
    

## Investigating sparsity of best LASSO model (returned from LassoCV)


```python
# Get coefficient matrix
coef_matrix = trained_model.coef_
coef_matrix
```




    array([ 0.        ,  0.        ,  0.01883461, -0.0113799 ,  0.02434467,
           -0.        ,  0.04836801,  0.06445211,  0.04569132,  0.00376452,
            0.03991102, -0.        ,  0.01658349,  0.        ,  0.        ,
            0.03003712,  0.        ,  0.00224332,  0.        ,  0.03145369,
            0.09581944,  0.05103676,  0.        , -0.        ,  0.        ,
           -0.        , -0.        , -0.01382748, -0.00251348,  0.        ,
            0.        ,  0.        , -0.        ,  0.        ,  0.        ,
            0.02047155, -0.        ,  0.        , -0.        ,  0.        ,
           -0.        ,  0.        ,  0.03235523,  0.        , -0.        ,
            0.        , -0.03051745,  0.        , -0.        ,  0.        ,
            0.        ])




```python
from interpret_model import coef_plot
help(coef_plot)
```

    Help on function coef_plot in module interpret_model:
    
    coef_plot(coefs: pandas.core.series.Series, plot_const: bool = False, index: bool = None) -> matplotlib.figure.Figure
        Plot coefficient values and feature importance based on sorted feature importance.
        
        Args:
            coefs (pd.Series or np.ndarray): Coefficient values.
            plot_const (bool, optional): Whether or not to plot the y-intercept coef value. Default is False.
            index (list or pd.Index, optional): Index labels for the coefficients. Default is None.
        
        Returns:
            plt.Figure: The figure containing the coefficient plots.
    
    


```python
fig = coef_plot(coefs=coef_matrix, plot_const=False, index=X_train_z.columns) 
```


    

    


### Investivating the alpha hyperparameter


```python
import matplotlib.pyplot as plt
# Calculate the corresponding training scores for each alpha and fold
from sklearn.linear_model import Lasso
train_scores = []

preds = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_train_z, y_train)
    pred = lasso.predict(X_train_z)
    preds.append(pred)
    train_scores.append(np.mean((pred - y_train) ** 2))

# Retrieve the validation scores at each alpha and fold
val_scores = trained_model.mse_path_

# Plot the training and validation scores
plt.figure(figsize=(10, 6))
plt.plot(np.log10(trained_model.alphas_), train_scores, label='Train', marker='o')
plt.plot(np.log10(trained_model.alphas_), val_scores.mean(axis=1), label='Validation (CV)', marker='o')
plt.xlabel('log(alpha)')
plt.ylabel('Mean Squared Error')
plt.title('LassoCV Train/Validation Scores')
plt.legend()
plt.grid(True)
plt.savefig('..//fig//regression//regularize//alpha_cv_results.png', bbox_inches='tight', dpi=300, facecolor='white');

plt.show()

```


    

    


<img src="../fig/regression/regularize/alpha_cv_results.png"  align="center" width="60%" height="60%">

## Hypothesis testing with LASSO models


```python
# # Calculate p-values for LASSO coefficients
# X_train_with_constant = sm.add_constant(X_train)
# lasso_model = sm.OLS(y_train, X_train_with_constant)
# lasso_results = lasso_model.fit_regularized(alpha=trained_model.alpha_, L1_wt=1.0)

# # Print the summary of LASSO results
# print(lasso_results.summary())

```
