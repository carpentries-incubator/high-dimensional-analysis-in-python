```python
---
title: Exploring additional datasets
teaching: 45
exercises: 2
keypoints:
- ""
objectives:
- ""
questions:
- "How can we use everything we have learned to analyze a new dataset?"
---
```

## Preprocessing

**Note**: Adapt get_feat_types() and encode_predictors_housing_data() for your data. Use new functions with slightly different names.


```python
# get geat types - you'll need to create a similar function for your data that stores the type of each predictor
from preprocessing import get_feat_types
predictor_type_dict = get_feat_types()
continuous_fields = predictor_type_dict['continuous_fields']
```


```python
# encode predictors (one-hot encoding for categorical data) - note you may have to create a new function starting from a copy of this one
from preprocessing import encode_predictors_housing_data
X_encoded = encode_predictors_housing_data(X)
```


```python
# remove columns with nans or containing > 95% constant values (typically 0's)
from preprocessing import remove_bad_cols
X_good = remove_bad_cols(X, 95)
```


```python
# train/test splits
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_log, 
                                                    test_size=0.33, 
                                                    random_state=0)

print(x_train.shape)
print(x_test.shape)
```


```python
# zscore
from preprocessing import zscore
# get means and stds
train_means = X_train.mean()
train_stds = X_train.std()
X_train_z = zscore(df=X_train, train_means=train_means, train_stds=train_stds)
X_test_z = zscore(df=X_test, train_means=train_means, train_stds=train_stds)
X_train_z.head()
```


```python
# get random predictor permutations...
from preprocessing import get_predictor_combos
sampled_combinations = get_predictor_combos(X_train=X_train, K=K, n=25)
```

## Feature selection


```python
from feature_selection import get_best_uni_predictors

top_features = get_best_uni_predictors(N_keep=5, y=y, baseline_pred=y.mean(), 
                                       X_train=X_train, y_train=y_train,
                                       X_val=X_val, y_val=y_val,
                                       metric='RMSE', y_log_scaled=True)

top_features
```

## Fit/eval model (sklearn version)


```python
from regression_predict_sklearn import fit_eval_model
fit_eval_model(y=y, baseline_pred=y.mean(),
               X_train=X_train_z, y_train=y_train,
               X_test=X_test_z, y_test=y_test, 
               predictors=X_train_z.columns,
               metric='RMSE',
               y_log_scaled=True,
               model_type='unregularized',
               include_plots=True, plot_raw=True, verbose=True)
```

## Model eval


```python
# plot (1) true vs predicted vals for train/test sets and (2) best line of fit (only applies for univariate models)
from regression_predict_sklearn import plot_train_test_predictions 
(fig1, fig2) = plot_train_test_predictions(predictors=[predictor],
                                           X_train=x_train, X_test=x_test,
                                           y_train=y_train, y_test=y_test,
                                           y_pred_train=y_pred_train, y_pred_test=y_pred_test,
                                           y_log_scaled=True, plot_raw=True);
```


```python
# report baseline, train, and test errors
from regression_predict_sklearn import measure_model_err
error_df = measure_model_err(y=y, baseline_pred=baseline_predict,
                             y_train=y_train, y_pred_train=y_pred_train, 
                             y_test=y_test, y_pred_test=y_pred_test, 
                             metric='MAPE', y_log_scaled=False) 

error_df.head()
```

## Comparing models...


```python
df_model_err = compare_models(y=y, baseline_pred=baseline_predict,
                              X_train=X_train, y_train=y_train, 
                              X_val=X_val, y_val=y_val,
                              predictors_list=X_train.columns, 
                              metric='RMSE', y_log_scaled=True, 
                              model_type='unregularized', 
                              include_plots=False, plot_raw=False, verbose=False)
```
