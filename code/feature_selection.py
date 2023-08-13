import numpy as np
import pandas as pd

from typing import Optional, Tuple, List, Union


def get_best_uni_predictors(N_keep: int,
                            y: Union[np.ndarray, pd.Series],
                            baseline_pred: Union[np.ndarray, pd.Series],
                            X_train: pd.DataFrame, y_train: Union[np.ndarray, pd.Series],
                            X_val: pd.DataFrame, y_val: Union[np.ndarray, pd.Series],
                            metric: str,
                            y_log_scaled: bool) -> List[str]:
    """
    Get the top N_keep univariate predictors based on validation error.

    Args:
        N_keep (int): Number of top predictors to select.
        y (Union[np.ndarray, pd.Series]): Target variable.
        baseline_pred (Union[np.ndarray, pd.Series]): Baseline predictions.
        X_train (pd.DataFrame): Training feature data.
        y_train (Union[np.ndarray, pd.Series]): Training target values.
        X_val (pd.DataFrame): Validation feature data.
        y_val (Union[np.ndarray, pd.Series]): Validation target values.
        metric (str): Error metric to use for evaluation.
        y_log_scaled (bool): Whether the target values are log-scaled.

    Returns:
        List[str]: List of top N_keep univariate predictor names.
    """
    from regression_predict_sklearn import compare_models
    df_model_err = compare_models(y=y, baseline_pred=baseline_pred,
                                  X_train=X_train, y_train=y_train, 
                                  X_val=X_val, y_val=y_val,
                                  predictors_list=X_train.columns, 
                                  metric=metric, y_log_scaled=y_log_scaled, 
                                  model_type='unregularized', 
                                  include_plots=False, plot_raw=False, verbose=False)

    top_features = df_model_err.sort_values(by='Validation Error').head(N_keep)['Predictors'].tolist()
    return top_features

def hypothesis_driven_predictors() -> None:
    """
    Provide guidance on narrowing down predictors based on literature.

    Returns:
        None
    """
    print("Consult the literature in your field to narrow down your high-dimensional dataset "
          "to a list of predictors that have been shown to have some influence over the target "
          "variable. This approach, by itself, can be somewhat limiting as a good scientist should "
          "take care to investigate unexplored predictors in addition to well documented predictors. "
          "It is often worthwhile to combine a priori knowledge with data-driven methods.")

    return None
