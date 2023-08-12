import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm 

from typing import Optional, Tuple, List, Union

def regression_assumptions_e1_explore_altPredictors():
    
    # 1. First, extract the data you'll be using to fit the model.
    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True, parser='auto') #
    y = housing['target']
    y_log = np.log(y) 
    predictors = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageArea', 'GarageCars', 'Neighborhood'] 
    X=housing['data'][predictors]
    X.head()
    
    # 2. Next, preprocess the data using `encode_predictors_housing_data(X)` and `remove_bad_cols(X, 95)` from preprocessing.py.
    from preprocessing import encode_predictors_housing_data
    X_enc = encode_predictors_housing_data(X)
    X_enc.head()

    from preprocessing import remove_bad_cols
    X_good = remove_bad_cols(X_enc, 95) 
    
    # 3. Use `multicollinearity_test()` from check_assumptions.py to check for multicollinearity and remove any problematic predictors. Repeat the test until multicollinearity is removed (VIF scores < 10).
    from check_assumptions import multicollinearity_test 
    multicollinearity_test(X_good);
    X_better = X_good.drop(['GarageCars','YearBuilt'],axis = 1)
    multicollinearity_test(X_better);
    
    # 4. Perform a train/test split leaving 33% of the data out for the test set. Use `random_state=0` to match the same results as everyone else.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_better, y_log, test_size=0.33, random_state=0)
    
    # 5. Use the `zscore()` helper from preprocessing.py to zscore the train and test sets. This will allow us to compare coefficient sizes later.
    from preprocessing import zscore
    X_train_z = zscore(df=X_train, train_means=X_train.mean(), train_stds=X_train.std())
    X_test_z = zscore(df=X_test, train_means=X_train.mean(), train_stds=X_train.std())
    X_train_z.head()

    # 6. Train the model using the statsmodels package. Don't forget to add the constant to both train/test sets (so you can do prediciton on both)

    # Add a constant column to the predictor variables dataframe
    X_train_z = sm.add_constant(X_train_z)
    # Add the constant to the test set as well so we can use the model to form predictions on the test set later
    X_test_z = sm.add_constant(X_test_z)
    # Fit the multivariate regression model
    model = sm.OLS(y_train, X_train_z)
    trained_model = model.fit()

    # 7. Check for evidence of extreme underfitting or overfitting using `measure_model_err()` from regression_predict_sklearn.py
    from regression_predict_sklearn import measure_model_err
    # to calculate residuals and R-squared for the test set, we'll need to get the model predictions first
    y_pred_train = trained_model.predict(X_train_z)
    y_pred_test = trained_model.predict(X_test_z)
    errors_df = measure_model_err(y, np.mean(y),
                          y_train, y_pred_train,
                          y_test, y_pred_test,
                          'RMSE', y_log_scaled=True) 

    errors_df.head()

    # 8. Check all model assumptions
    from check_assumptions import eval_regression_assumptions
    eval_regression_assumptions(trained_model=trained_model, X=X_train_z, y=y_train, 
                                y_pred=y_pred_train, y_log_scaled=True, plot_raw=False, threshold_p_value=.05);
    
    
    return trained_model

