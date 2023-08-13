def get_best_uni_predictors(N_keep, y, baseline_pred,
                           X_train, y_train,
                           X_val, y_val,
                           metric,
                           y_log_scaled):
    
    from regression_predict_sklearn import compare_models
    df_model_err = compare_models(y=y, baseline_pred=y.mean(),
                                  X_train=X_train, y_train=y_train, 
                                  X_val=X_val, y_val=y_val,
                                  predictors_list=X_train.columns, 
                                  metric='RMSE', y_log_scaled=y_log_scaled, 
                                  model_type='unregularized', 
                                  include_plots=False, plot_raw=False, verbose=False)

    top_features = df_model_err.sort_values(by='Validation Error').head(N_keep)['Predictors'].tolist()
    return top_features
