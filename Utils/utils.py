import numpy as np

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
        Computes fraction of variance that ypred explaines about y.
        Returns 1 - Var[y-ypred] / Var[y]

        Interpretation:
            ev=0 => might as well have predicted zero
            ev=1 => perfect prediction
            ev<0 => worse that just predicting zero
        
        :param y_pred: the prediction
        :param y_true: the expected value
        :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y