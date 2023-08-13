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
    Columns removed: ['ScreenPorch', 'LowQualFinSF', 'MasVnrArea', 'GarageYrBlt', '3SsnPorch', 'BsmtHalfBath', 'LotFrontage', 'PoolArea', 'KitchenAbvGr', 'BsmtFinSF2', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'RoofStyle_Flat', 'RoofStyle_Gambrel', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'Foundation_BrkTil', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Alley_Grvl', 'Alley_Pave', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Partial', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'Utilities_AllPub', 'Utilities_NoSeWa', 'MasVnrType_BrkCmn', 'MasVnrType_Stone', "Exterior2nd_'Brk Cmn'", "Exterior2nd_'Wd Sdng'", "Exterior2nd_'Wd Shng'", 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_ImStucc', 'Exterior2nd_Other', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'MSSubClass_30', 'MSSubClass_40', 'MSSubClass_45', 'MSSubClass_50', 'MSSubClass_70', 'MSSubClass_75', 'MSSubClass_80', 'MSSubClass_85', 'MSSubClass_90', 'MSSubClass_120', 'MSSubClass_160', 'MSSubClass_180', 'MSSubClass_190', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'MiscFeature_Gar2', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', "MSZoning_'C (all)'", 'MSZoning_FV', 'MSZoning_RH', 'GarageType_2Types', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_ImStucc', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_WdShing', 'Street', 'CentralAir']
    




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
      <th>OverallQual</th>
      <th>MoSold</th>
      <th>1stFlrSF</th>
      <th>GrLivArea</th>
      <th>Fireplaces</th>
      <th>WoodDeckSF</th>
      <th>BedroomAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>OverallCond</th>
      <th>YearRemodAdd</th>
      <th>...</th>
      <th>HouseStyle_1Story</th>
      <th>HouseStyle_2Story</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>GarageType_Attchd</th>
      <th>GarageType_Detchd</th>
      <th>Exterior1st_'Wd Sdng'</th>
      <th>Exterior1st_HdBoard</th>
      <th>Exterior1st_MetalSd</th>
      <th>Exterior1st_VinylSd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>368</th>
      <td>-0.913011</td>
      <td>-1.088988</td>
      <td>0.362455</td>
      <td>-0.445068</td>
      <td>0.611180</td>
      <td>-0.709012</td>
      <td>-1.044188</td>
      <td>0.453853</td>
      <td>0.523613</td>
      <td>-1.626427</td>
      <td>...</td>
      <td>0.956799</td>
      <td>-0.73492</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>0.627189</td>
      <td>-0.413187</td>
      <td>-0.38031</td>
      <td>2.396484</td>
      <td>-0.429212</td>
      <td>-0.799272</td>
    </tr>
    <tr>
      <th>973</th>
      <td>0.877896</td>
      <td>1.931227</td>
      <td>0.807446</td>
      <td>-0.083375</td>
      <td>-1.073694</td>
      <td>-0.709012</td>
      <td>0.223755</td>
      <td>-0.198161</td>
      <td>-0.484095</td>
      <td>1.184480</td>
      <td>...</td>
      <td>0.956799</td>
      <td>-0.73492</td>
      <td>-2.225080</td>
      <td>-0.345968</td>
      <td>0.627189</td>
      <td>-0.413187</td>
      <td>-0.38031</td>
      <td>-0.413187</td>
      <td>-0.429212</td>
      <td>-0.799272</td>
    </tr>
    <tr>
      <th>387</th>
      <td>-0.017558</td>
      <td>1.260068</td>
      <td>-0.035256</td>
      <td>-0.768331</td>
      <td>0.611180</td>
      <td>1.398549</td>
      <td>0.223755</td>
      <td>-0.198161</td>
      <td>0.523613</td>
      <td>-0.481243</td>
      <td>...</td>
      <td>-1.034905</td>
      <td>-0.73492</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>-1.578785</td>
      <td>2.396484</td>
      <td>-0.38031</td>
      <td>-0.413187</td>
      <td>2.307012</td>
      <td>-0.799272</td>
    </tr>
    <tr>
      <th>816</th>
      <td>-0.913011</td>
      <td>0.253330</td>
      <td>-0.360655</td>
      <td>-1.032819</td>
      <td>0.611180</td>
      <td>-0.709012</td>
      <td>-1.044188</td>
      <td>-1.502190</td>
      <td>0.523613</td>
      <td>-1.626427</td>
      <td>...</td>
      <td>0.956799</td>
      <td>-0.73492</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>0.627189</td>
      <td>-0.413187</td>
      <td>-0.38031</td>
      <td>-0.413187</td>
      <td>-0.429212</td>
      <td>-0.799272</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>1.773349</td>
      <td>-1.760147</td>
      <td>1.369247</td>
      <td>0.373261</td>
      <td>0.611180</td>
      <td>0.515652</td>
      <td>0.223755</td>
      <td>1.105868</td>
      <td>-0.484095</td>
      <td>1.184480</td>
      <td>...</td>
      <td>0.956799</td>
      <td>-0.73492</td>
      <td>0.445016</td>
      <td>-0.345968</td>
      <td>0.627189</td>
      <td>-0.413187</td>
      <td>-0.38031</td>
      <td>-0.413187</td>
      <td>-0.429212</td>
      <td>1.238872</td>
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
help(LassoCV)
```

    Help on class LassoCV in module sklearn.linear_model._coordinate_descent:
    
    class LassoCV(sklearn.base.RegressorMixin, LinearModelCV)
     |  LassoCV(*, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=None, positive=False, random_state=None, selection='cyclic')
     |  
     |  Lasso linear model with iterative fitting along a regularization path.
     |  
     |  See glossary entry for :term:`cross-validation estimator`.
     |  
     |  The best model is selected by cross-validation.
     |  
     |  The optimization objective for Lasso is::
     |  
     |      (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
     |  
     |  Read more in the :ref:`User Guide <lasso>`.
     |  
     |  Parameters
     |  ----------
     |  eps : float, default=1e-3
     |      Length of the path. ``eps=1e-3`` means that
     |      ``alpha_min / alpha_max = 1e-3``.
     |  
     |  n_alphas : int, default=100
     |      Number of alphas along the regularization path.
     |  
     |  alphas : array-like, default=None
     |      List of alphas where to compute the models.
     |      If ``None`` alphas are set automatically.
     |  
     |  fit_intercept : bool, default=True
     |      Whether to calculate the intercept for this model. If set
     |      to false, no intercept will be used in calculations
     |      (i.e. data is expected to be centered).
     |  
     |  precompute : 'auto', bool or array-like of shape             (n_features, n_features), default='auto'
     |      Whether to use a precomputed Gram matrix to speed up
     |      calculations. If set to ``'auto'`` let us decide. The Gram
     |      matrix can also be passed as argument.
     |  
     |  max_iter : int, default=1000
     |      The maximum number of iterations.
     |  
     |  tol : float, default=1e-4
     |      The tolerance for the optimization: if the updates are
     |      smaller than ``tol``, the optimization code checks the
     |      dual gap for optimality and continues until it is smaller
     |      than ``tol``.
     |  
     |  copy_X : bool, default=True
     |      If ``True``, X will be copied; else, it may be overwritten.
     |  
     |  cv : int, cross-validation generator or iterable, default=None
     |      Determines the cross-validation splitting strategy.
     |      Possible inputs for cv are:
     |  
     |      - None, to use the default 5-fold cross-validation,
     |      - int, to specify the number of folds.
     |      - :term:`CV splitter`,
     |      - An iterable yielding (train, test) splits as arrays of indices.
     |  
     |      For int/None inputs, :class:`KFold` is used.
     |  
     |      Refer :ref:`User Guide <cross_validation>` for the various
     |      cross-validation strategies that can be used here.
     |  
     |      .. versionchanged:: 0.22
     |          ``cv`` default value if None changed from 3-fold to 5-fold.
     |  
     |  verbose : bool or int, default=False
     |      Amount of verbosity.
     |  
     |  n_jobs : int, default=None
     |      Number of CPUs to use during the cross validation.
     |      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
     |      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
     |      for more details.
     |  
     |  positive : bool, default=False
     |      If positive, restrict regression coefficients to be positive.
     |  
     |  random_state : int, RandomState instance, default=None
     |      The seed of the pseudo random number generator that selects a random
     |      feature to update. Used when ``selection`` == 'random'.
     |      Pass an int for reproducible output across multiple function calls.
     |      See :term:`Glossary <random_state>`.
     |  
     |  selection : {'cyclic', 'random'}, default='cyclic'
     |      If set to 'random', a random coefficient is updated every iteration
     |      rather than looping over features sequentially by default. This
     |      (setting to 'random') often leads to significantly faster convergence
     |      especially when tol is higher than 1e-4.
     |  
     |  Attributes
     |  ----------
     |  alpha_ : float
     |      The amount of penalization chosen by cross validation.
     |  
     |  coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
     |      Parameter vector (w in the cost function formula).
     |  
     |  intercept_ : float or ndarray of shape (n_targets,)
     |      Independent term in decision function.
     |  
     |  mse_path_ : ndarray of shape (n_alphas, n_folds)
     |      Mean square error for the test set on each fold, varying alpha.
     |  
     |  alphas_ : ndarray of shape (n_alphas,)
     |      The grid of alphas used for fitting.
     |  
     |  dual_gap_ : float or ndarray of shape (n_targets,)
     |      The dual gap at the end of the optimization for the optimal alpha
     |      (``alpha_``).
     |  
     |  n_iter_ : int
     |      Number of iterations run by the coordinate descent solver to reach
     |      the specified tolerance for the optimal alpha.
     |  
     |  n_features_in_ : int
     |      Number of features seen during :term:`fit`.
     |  
     |      .. versionadded:: 0.24
     |  
     |  feature_names_in_ : ndarray of shape (`n_features_in_`,)
     |      Names of features seen during :term:`fit`. Defined only when `X`
     |      has feature names that are all strings.
     |  
     |      .. versionadded:: 1.0
     |  
     |  See Also
     |  --------
     |  lars_path : Compute Least Angle Regression or Lasso path using LARS
     |      algorithm.
     |  lasso_path : Compute Lasso path with coordinate descent.
     |  Lasso : The Lasso is a linear model that estimates sparse coefficients.
     |  LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
     |  LassoCV : Lasso linear model with iterative fitting along a regularization
     |      path.
     |  LassoLarsCV : Cross-validated Lasso using the LARS algorithm.
     |  
     |  Notes
     |  -----
     |  In `fit`, once the best parameter `alpha` is found through
     |  cross-validation, the model is fit again using the entire training set.
     |  
     |  To avoid unnecessary memory duplication the `X` argument of the `fit`
     |  method should be directly passed as a Fortran-contiguous numpy array.
     |  
     |   For an example, see
     |   :ref:`examples/linear_model/plot_lasso_model_selection.py
     |   <sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`.
     |  
     |  :class:`LassoCV` leads to different results than a hyperparameter
     |  search using :class:`~sklearn.model_selection.GridSearchCV` with a
     |  :class:`Lasso` model. In :class:`LassoCV`, a model for a given
     |  penalty `alpha` is warm started using the coefficients of the
     |  closest model (trained at the previous iteration) on the
     |  regularization path. It tends to speed up the hyperparameter
     |  search.
     |  
     |  Examples
     |  --------
     |  >>> from sklearn.linear_model import LassoCV
     |  >>> from sklearn.datasets import make_regression
     |  >>> X, y = make_regression(noise=4, random_state=0)
     |  >>> reg = LassoCV(cv=5, random_state=0).fit(X, y)
     |  >>> reg.score(X, y)
     |  0.9993...
     |  >>> reg.predict(X[:1,])
     |  array([-78.4951...])
     |  
     |  Method resolution order:
     |      LassoCV
     |      sklearn.base.RegressorMixin
     |      LinearModelCV
     |      sklearn.base.MultiOutputMixin
     |      sklearn.linear_model._base.LinearModel
     |      sklearn.base.BaseEstimator
     |      sklearn.utils._metadata_requests._MetadataRequester
     |      abc.ABC
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, *, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=None, positive=False, random_state=None, selection='cyclic')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  set_fit_request(self: sklearn.linear_model._coordinate_descent.LassoCV, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.linear_model._coordinate_descent.LassoCV
     |      Request metadata passed to the ``fit`` method.
     |      
     |      Note that this method is only relevant if
     |      ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
     |      Please see :ref:`User Guide <metadata_routing>` on how the routing
     |      mechanism works.
     |      
     |      The options for each parameter are:
     |      
     |      - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
     |      
     |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
     |      
     |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
     |      
     |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
     |      
     |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
     |      existing request. This allows you to change the request for some
     |      parameters and not others.
     |      
     |      .. versionadded:: 1.3
     |      
     |      .. note::
     |          This method is only relevant if this estimator is used as a
     |          sub-estimator of a meta-estimator, e.g. used inside a
     |          :class:`pipeline.Pipeline`. Otherwise it has no effect.
     |      
     |      Parameters
     |      ----------
     |      sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
     |          Metadata routing for ``sample_weight`` parameter in ``fit``.
     |      
     |      Returns
     |      -------
     |      self : object
     |          The updated object.
     |  
     |  set_score_request(self: sklearn.linear_model._coordinate_descent.LassoCV, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.linear_model._coordinate_descent.LassoCV
     |      Request metadata passed to the ``score`` method.
     |      
     |      Note that this method is only relevant if
     |      ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
     |      Please see :ref:`User Guide <metadata_routing>` on how the routing
     |      mechanism works.
     |      
     |      The options for each parameter are:
     |      
     |      - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
     |      
     |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
     |      
     |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
     |      
     |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
     |      
     |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
     |      existing request. This allows you to change the request for some
     |      parameters and not others.
     |      
     |      .. versionadded:: 1.3
     |      
     |      .. note::
     |          This method is only relevant if this estimator is used as a
     |          sub-estimator of a meta-estimator, e.g. used inside a
     |          :class:`pipeline.Pipeline`. Otherwise it has no effect.
     |      
     |      Parameters
     |      ----------
     |      sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
     |          Metadata routing for ``sample_weight`` parameter in ``score``.
     |      
     |      Returns
     |      -------
     |      self : object
     |          The updated object.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  path = lasso_path(X, y, *, eps=0.001, n_alphas=100, alphas=None, precompute='auto', Xy=None, copy_X=True, coef_init=None, verbose=False, return_n_iter=False, positive=False, **params)
     |      Compute Lasso path with coordinate descent.
     |      
     |      The Lasso optimization function varies for mono and multi-outputs.
     |      
     |      For mono-output tasks it is::
     |      
     |          (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
     |      
     |      For multi-output tasks it is::
     |      
     |          (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21
     |      
     |      Where::
     |      
     |          ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}
     |      
     |      i.e. the sum of norm of each row.
     |      
     |      Read more in the :ref:`User Guide <lasso>`.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
     |          Training data. Pass directly as Fortran-contiguous data to avoid
     |          unnecessary memory duplication. If ``y`` is mono-output then ``X``
     |          can be sparse.
     |      
     |      y : {array-like, sparse matrix} of shape (n_samples,) or         (n_samples, n_targets)
     |          Target values.
     |      
     |      eps : float, default=1e-3
     |          Length of the path. ``eps=1e-3`` means that
     |          ``alpha_min / alpha_max = 1e-3``.
     |      
     |      n_alphas : int, default=100
     |          Number of alphas along the regularization path.
     |      
     |      alphas : ndarray, default=None
     |          List of alphas where to compute the models.
     |          If ``None`` alphas are set automatically.
     |      
     |      precompute : 'auto', bool or array-like of shape             (n_features, n_features), default='auto'
     |          Whether to use a precomputed Gram matrix to speed up
     |          calculations. If set to ``'auto'`` let us decide. The Gram
     |          matrix can also be passed as argument.
     |      
     |      Xy : array-like of shape (n_features,) or (n_features, n_targets),         default=None
     |          Xy = np.dot(X.T, y) that can be precomputed. It is useful
     |          only when the Gram matrix is precomputed.
     |      
     |      copy_X : bool, default=True
     |          If ``True``, X will be copied; else, it may be overwritten.
     |      
     |      coef_init : ndarray of shape (n_features, ), default=None
     |          The initial values of the coefficients.
     |      
     |      verbose : bool or int, default=False
     |          Amount of verbosity.
     |      
     |      return_n_iter : bool, default=False
     |          Whether to return the number of iterations or not.
     |      
     |      positive : bool, default=False
     |          If set to True, forces coefficients to be positive.
     |          (Only allowed when ``y.ndim == 1``).
     |      
     |      **params : kwargs
     |          Keyword arguments passed to the coordinate descent solver.
     |      
     |      Returns
     |      -------
     |      alphas : ndarray of shape (n_alphas,)
     |          The alphas along the path where models are computed.
     |      
     |      coefs : ndarray of shape (n_features, n_alphas) or             (n_targets, n_features, n_alphas)
     |          Coefficients along the path.
     |      
     |      dual_gaps : ndarray of shape (n_alphas,)
     |          The dual gaps at the end of the optimization for each alpha.
     |      
     |      n_iters : list of int
     |          The number of iterations taken by the coordinate descent optimizer to
     |          reach the specified tolerance for each alpha.
     |      
     |      See Also
     |      --------
     |      lars_path : Compute Least Angle Regression or Lasso path using LARS
     |          algorithm.
     |      Lasso : The Lasso is a linear model that estimates sparse coefficients.
     |      LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
     |      LassoCV : Lasso linear model with iterative fitting along a regularization
     |          path.
     |      LassoLarsCV : Cross-validated Lasso using the LARS algorithm.
     |      sklearn.decomposition.sparse_encode : Estimator that can be used to
     |          transform signals into sparse linear combination of atoms from a fixed.
     |      
     |      Notes
     |      -----
     |      For an example, see
     |      :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
     |      <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.
     |      
     |      To avoid unnecessary memory duplication the X argument of the fit method
     |      should be directly passed as a Fortran-contiguous numpy array.
     |      
     |      Note that in certain cases, the Lars solver may be significantly
     |      faster to implement this functionality. In particular, linear
     |      interpolation can be used to retrieve model coefficients between the
     |      values output by lars_path
     |      
     |      Examples
     |      --------
     |      
     |      Comparing lasso_path and lars_path with interpolation:
     |      
     |      >>> import numpy as np
     |      >>> from sklearn.linear_model import lasso_path
     |      >>> X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
     |      >>> y = np.array([1, 2, 3.1])
     |      >>> # Use lasso_path to compute a coefficient path
     |      >>> _, coef_path, _ = lasso_path(X, y, alphas=[5., 1., .5])
     |      >>> print(coef_path)
     |      [[0.         0.         0.46874778]
     |       [0.2159048  0.4425765  0.23689075]]
     |      
     |      >>> # Now use lars_path and 1D linear interpolation to compute the
     |      >>> # same path
     |      >>> from sklearn.linear_model import lars_path
     |      >>> alphas, active, coef_path_lars = lars_path(X, y, method='lasso')
     |      >>> from scipy import interpolate
     |      >>> coef_path_continuous = interpolate.interp1d(alphas[::-1],
     |      ...                                             coef_path_lars[:, ::-1])
     |      >>> print(coef_path_continuous([5., 1., .5]))
     |      [[0.         0.         0.46915237]
     |       [0.2159048  0.4425765  0.23668876]]
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  __annotations__ = {}
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.RegressorMixin:
     |  
     |  score(self, X, y, sample_weight=None)
     |      Return the coefficient of determination of the prediction.
     |      
     |      The coefficient of determination :math:`R^2` is defined as
     |      :math:`(1 - \frac{u}{v})`, where :math:`u` is the residual
     |      sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
     |      is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
     |      The best possible score is 1.0 and it can be negative (because the
     |      model can be arbitrarily worse). A constant model that always predicts
     |      the expected value of `y`, disregarding the input features, would get
     |      a :math:`R^2` score of 0.0.
     |      
     |      Parameters
     |      ----------
     |      X : array-like of shape (n_samples, n_features)
     |          Test samples. For some estimators this may be a precomputed
     |          kernel matrix or a list of generic objects instead with shape
     |          ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
     |          is the number of samples used in the fitting for the estimator.
     |      
     |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
     |          True values for `X`.
     |      
     |      sample_weight : array-like of shape (n_samples,), default=None
     |          Sample weights.
     |      
     |      Returns
     |      -------
     |      score : float
     |          :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.
     |      
     |      Notes
     |      -----
     |      The :math:`R^2` score used when calling ``score`` on a regressor uses
     |      ``multioutput='uniform_average'`` from version 0.23 to keep consistent
     |      with default value of :func:`~sklearn.metrics.r2_score`.
     |      This influences the ``score`` method of all the multioutput
     |      regressors (except for
     |      :class:`~sklearn.multioutput.MultiOutputRegressor`).
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.base.RegressorMixin:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from LinearModelCV:
     |  
     |  fit(self, X, y, sample_weight=None)
     |      Fit linear model with coordinate descent.
     |      
     |      Fit is on grid of alphas and best alpha estimated by cross-validation.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
     |          Training data. Pass directly as Fortran-contiguous data
     |          to avoid unnecessary memory duplication. If y is mono-output,
     |          X can be sparse.
     |      
     |      y : array-like of shape (n_samples,) or (n_samples, n_targets)
     |          Target values.
     |      
     |      sample_weight : float or array-like of shape (n_samples,),                 default=None
     |          Sample weights used for fitting and evaluation of the weighted
     |          mean squared error of each cv-fold. Note that the cross validated
     |          MSE that is finally used to find the best model is the unweighted
     |          mean over the (weighted) MSEs of each test fold.
     |      
     |      Returns
     |      -------
     |      self : object
     |          Returns an instance of fitted model.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.linear_model._base.LinearModel:
     |  
     |  predict(self, X)
     |      Predict using the linear model.
     |      
     |      Parameters
     |      ----------
     |      X : array-like or sparse matrix, shape (n_samples, n_features)
     |          Samples.
     |      
     |      Returns
     |      -------
     |      C : array, shape (n_samples,)
     |          Returns predicted values.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.BaseEstimator:
     |  
     |  __getstate__(self)
     |      Helper for pickle.
     |  
     |  __repr__(self, N_CHAR_MAX=700)
     |      Return repr(self).
     |  
     |  __setstate__(self, state)
     |  
     |  __sklearn_clone__(self)
     |  
     |  get_params(self, deep=True)
     |      Get parameters for this estimator.
     |      
     |      Parameters
     |      ----------
     |      deep : bool, default=True
     |          If True, will return the parameters for this estimator and
     |          contained subobjects that are estimators.
     |      
     |      Returns
     |      -------
     |      params : dict
     |          Parameter names mapped to their values.
     |  
     |  set_params(self, **params)
     |      Set the parameters of this estimator.
     |      
     |      The method works on simple estimators as well as on nested objects
     |      (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
     |      parameters of the form ``<component>__<parameter>`` so that it's
     |      possible to update each component of a nested object.
     |      
     |      Parameters
     |      ----------
     |      **params : dict
     |          Estimator parameters.
     |      
     |      Returns
     |      -------
     |      self : estimator instance
     |          Estimator instance.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.utils._metadata_requests._MetadataRequester:
     |  
     |  get_metadata_routing(self)
     |      Get metadata routing of this object.
     |      
     |      Please check :ref:`User Guide <metadata_routing>` on how the routing
     |      mechanism works.
     |      
     |      Returns
     |      -------
     |      routing : MetadataRequest
     |          A :class:`~utils.metadata_routing.MetadataRequest` encapsulating
     |          routing information.
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from sklearn.utils._metadata_requests._MetadataRequester:
     |  
     |  __init_subclass__(**kwargs) from abc.ABCMeta
     |      Set the ``set_{method}_request`` methods.
     |      
     |      This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It
     |      looks for the information available in the set default values which are
     |      set using ``__metadata_request__*`` class attributes, or inferred
     |      from method signatures.
     |      
     |      The ``__metadata_request__*`` class attributes are used when a method
     |      does not explicitly accept a metadata through its arguments or if the
     |      developer would like to specify a request value for those metadata
     |      which are different from the default ``None``.
     |      
     |      References
     |      ----------
     |      .. [1] https://www.python.org/dev/peps/pep-0487
    
    

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
    Train RMSE = 14835.039689206124
    Holdout RMSE = 112820.25536497678
    (Holdout-Train)/Train: 660%
    


    

    


    
    # of predictor vars = 51
    # of train observations = 102
    # of test observations = 1358
    Baseline RMSE = 79415.29188606751
    Train RMSE = 18937.05271249303
    Holdout RMSE = 56987.78539659686
    (Holdout-Train)/Train: 201%
    


    

    


    
    

## Investigating sparsity of best LASSO model (returned from LassoCV)


```python
# Get coefficient matrix
coef_matrix = trained_model.coef_
coef_matrix
```




    array([ 0.05099734, -0.        ,  0.        ,  0.09581748,  0.02434971,
            0.03003432,  0.        ,  0.        ,  0.04569418,  0.        ,
           -0.        ,  0.        , -0.        ,  0.        ,  0.        ,
            0.01884117,  0.06444948,  0.03995289,  0.04840338,  0.00376607,
            0.0165892 , -0.01137917,  0.00223687,  0.03142482,  0.        ,
           -0.        ,  0.        ,  0.02047176,  0.        ,  0.        ,
            0.        , -0.        , -0.        ,  0.        ,  0.03235407,
            0.        ,  0.        ,  0.        , -0.        , -0.        ,
            0.        , -0.        ,  0.        ,  0.        , -0.03051134,
           -0.        , -0.        , -0.01382   , -0.0025191 ,  0.        ,
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
