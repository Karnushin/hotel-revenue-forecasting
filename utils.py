from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from numpy.typing import ArrayLike


###for EDA###
def get_count_or_frac_duplicated_rows(
    df: pd.DataFrame,
    is_fraq: bool = False,
    is_cnt: bool = False
) -> float:
    """Getting count or fraction of duplicated rows
    
    Args:
        df: pandas df
        is_fraq: should return fraction of duplicated rows or not (if True, is_cnt must be False)
        is_cnt: should return count of duplicated rows or not (if True, is_fraq must be False)
        
    Returns:
        fraction or count of full duplicated rows
    """
    assert is_fraq + is_cnt == 1, 'is_fraq or is_cnt must be True'
    
    if is_fraq:
        return round((len(df) - len(df.drop_duplicates())) / df.shape[0], 3)
    if is_cnt:
        return len(df) - len(df.drop_duplicates())
    
    
def plot_barplot_cancelation_by_agg_cols(
    df: pd.DataFrame, 
    groupby_cols: List[str] = None,
    figsize: Tuple[int, int] = (7, 5),
    method: str = 'count',
    col_to_agg: str = 'hotel',
    is_return_df: pd.DataFrame = False
) -> None:
    """Plot barplot via seaborn.barplot
    
    Args:
        df: pd dataframe with 
            'hotel' column where name of hotel
            'is_canceled' column as binary with 0 and 1 describing type of rows in df
        groupby_cols: list of columns to be used in groupby
        figsize: figsize of matplotlib pyplot. Tuple of 2 integers
        method: method of pandas aggregation like 'sum', 'mean', 'count' etc
        col_to_agg: which column must be used for aggregation with param 'method'
        is_return_df: df of grouped values fot 'is_canceled' and hote creates during code and it can be returned
    
    Returns:
        return df if is_return_df=True (see the parameter)
    """
    plt.figure(figsize=figsize)
    
    assert 'is_canceled' in df.columns, 'is_canceled column must be in columns'
    assert 'hotel' in df.columns, 'hotel column must be in columns'
    
    df_noncan = df[df['is_canceled'] == 0]
    df_can = df[df['is_canceled'] == 1]
    bardf = pd.DataFrame({
        'Hotel': list(df_can.groupby(groupby_cols)['hotel'].count().index),
        'cancelled': df_can.groupby(groupby_cols).agg(val=(col_to_agg, method))['val'].values,
        'not cancelled': df_noncan.groupby(groupby_cols).agg(val=(col_to_agg, method))['val'].values 
    })
    bardf = bardf.melt(id_vars='Hotel').rename(columns=str.title)
    sns.barplot(x='Hotel', y='Value', hue='Variable', data=bardf)
    plt.show()
    
    if is_return_df:
        return bardf
    

def plot_seasonal_decomposition(ts: pd.Series, **kwargs) -> None:
    """plot seasonal decomposition
    
    Args:
        ts: like pd.Series
        **kwargs: parameters for statsmodels.api.tsa.seasonal_decompose
    """
    decomp = sm.tsa.seasonal_decompose(ts, period=kwargs.get('period'))
    plt.figure(figsize=(10,5))
    plt.plot(decomp.trend, label='trend')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)

    plt.figure(figsize=(10,5))
    plt.plot(decomp.seasonal, label='seasonal')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)

    plt.figure(figsize=(10,5))
    plt.plot(decomp.resid, label='residuals')
    plt.legend(loc='best')
    plt.tick_params(labelsize=7)

    plt.show()


###for modeling###
def prepare_ts_for_modeling(data: pd.DataFrame,
                           groupby_date_col: str,
                           target: str,
                           method_agg: str) -> pd.Series:
    """Data preparation as time series
    
    Args:
        data: pandas dataframe with groupby_date_col and target col
        groupby_date_col: name of date column for aggregation
        target: name of columns as target
        method_agg: aggregation method in pandas like 'count', 'sum', 'mean', etc
        
    Returns:
        prepared ts with column: y as numeric target after aggregation and ds as date
    """
    assert target in data.columns, f"{target} must be in a column"
    assert groupby_date_col in data.columns, f"{groupby_date_col} must be in a column"    
    
    ts = (
        data
        .groupby([groupby_date_col]).agg(y=(target, method_agg))
        .reset_index()
        .rename(columns={groupby_date_col: 'ds'})
    )
    
    return ts

def create_dataset(data: np.array, n_steps: int = 1) -> Tuple[np.array, np.array]:
    '''Preparation data for LSTM model
    
    Parameters:
        data - numpy array
        n_steps - number of previous time steps to use as input variables to predict the next time period 
        
    Return:
        prepared data in a view X array and Y array
    '''
    
    X, y = [], []
    
    for i in range(len(data)-n_steps):
        a = data[i:i+n_steps]
        X.append(a)
        y.append(data[i + n_steps])
        
    return np.array(X), np.array(y)


def get_prediction_inerval(vec: ArrayLike, alpha: int = 0.95) -> Tuple[np.array, np.array]:
    """Get confindence interval
    
    Args:
        vec: vector of numeric values
        alpha: alpha in conf int
    
    Returns:
        two vectors: vec_lower and vec_upper for given alpha
    """
    vec = np.array(vec)
    sem = np.std(vec, ddof=1)
    z = np.abs(scipy.stats.t.ppf((1 - alpha) / 2, len(vec) - 1))
    return vec - z * sem, vec + z * sem
    

def get_metrics(y_true: ArrayLike,
                y_pred: ArrayLike,
                metrics: List[str] = ['rmse', 'mae', 'mape']
               ) -> Dict[str, float]:
    """Compute given metrics
    
    Args:
        y_true: actual values
        y_pred: forecasted values
        metrics: list of metrics or one metrics in string format, deault=['rmse', 'mae', 'mape']
            allowed_metrics ['rmse', 'mae', 'mape', 'corr', 'mse']
        
    Returns:
        dict with calculated metrics
    
    """
    allowed_metrics = ['rmse', 'mae', 'mape', 'corr', 'mse']
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    d = {}
    
    if isinstance(metrics, str):
        metrics = metrics.split() #to remake in a list 
    
    for m in metrics:
        m = m.lower()
        
        if m not in allowed_metrics:
            raise ValueError(f'{m} is not found. Only {", ".join(allowed_metrics)} can be calculated')
            
        if m == 'mse':
            d[m.upper()] = np.mean((y_pred - y_true)**2)
        elif m == 'rmse':
            d[m.upper()] = np.mean((y_pred - y_true)**2)**0.5
        elif m == 'mae':
            d[m.upper()] = np.mean(np.abs(y_pred - y_true))
        elif m == 'mape':
            d[m.upper()] = np.mean(np.abs((y_pred - y_true) / y_true))
        elif m == 'corr':
            d[m.upper()] = np.corrcoef(y_pred, y_true)[0,1]
    
    return d
      