from statsmodels.tsa.stattools import adfuller  # type: ignore
from core_ml.filters import apply_kalman_filtering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore


def check_stationarity(df, p_value_threshold=0.05, critical_value="1%"):
    """
    Checks if columns in a dataframe are stationary using the Augmented Dickey-Fuller (ADF) test.

    Args:
        df (pd.DataFrame): Input dataframe containing numerical columns.
        p_value_threshold (float): Threshold below which null hypothesis is rejected (stationary).
        critical_value (str): The critical value key to check against ("1%", "5%", "10%").

    Returns:
        dict: A dictionary mapping column names to boolean values (True if stationary, False otherwise).

    Raises:
        Exception: If the dataframe contains too many missing values (>20%).
    """
    df = df.select_dtypes("number")

    if df.isna().sum().sum() > 0.2 * len(df):
        print("Too many empty values")
        raise Exception

    results = {}

    for col in df.columns:
        if len(df[col].value_counts()) != 1:
            adf_res = adfuller(df[col])
            # print(adf_res)
            p_value = adf_res[1]
            is_stat = bool(p_value <= p_value_threshold)
            if not is_stat:
                is_stat = abs(adf_res[4][critical_value]) < abs(adf_res[0])
            results[col] = is_stat

    return results


def to_sequence(series, seq_len=30):
    """
    Converts a time series into overlapping sequences (X) and target values (y).

    Used to prepare data for LSTM/Autoencoder training.

    Args:
        series (np.ndarray): The normalized time series data.
        seq_len (int): The length of each input sequence window.

    Returns:
        tuple: (X, y)
            - X: Array of shape (samples, seq_len, features)
            - y: Array of shape (samples, features) representing the next step (or reconstruction target).
    """
    ts_len = len(series)
    x = []
    y = []
    for i in range(ts_len - seq_len):
        x.append(series[i:i + seq_len])
        y.append(series[i + seq_len])

    return np.array(x), np.array(y)


def split_time_series(df, test_ratio):
    """
    Splits a time series dataframe into training and testing sets preserving temporal order.

    Args:
        df (pd.DataFrame): Input dataframe.
        test_ratio (float): Ratio of data to set aside for testing (e.g., 0.2).

    Returns:
        tuple: (train_df, test_df)
    """
    total_len = len(df)
    train_len = total_len - int(total_len * test_ratio)

    train = df.iloc[:train_len]
    test = df.iloc[train_len:]

    return train, test


def preprocessing_pipeline(
    df: pd.DataFrame,
    sampling_period: str = "500ms",
    sequence_len: int = None,
    norm: StandardScaler = None,
    diff_cols: list = None,
):
    """
    A comprehensive pipeline that prepares raw robot/sensor data for machine learning.

    Steps:
    1. Sets timestamp index.
    2. Resamples and fills missing data.
    3. Applies Kalman filtering for noise reduction.
    4. Checks stationarity and applies differencing if needed.
    5. Normalizes data using StandardScaler.
    6. (Optional) Converts to sequences for LSTM input.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        sampling_period (str): Pandas offset string for resampling frequency.
        sequence_len (int, optional): If provided, returns sequenced data (X, y).
        norm (StandardScaler, optional): Pre-fitted scaler (for inference). If None, fits a new one.
        diff_cols (list, optional): List of columns to difference (for inference). If None, calculates them.

    Returns:
        tuple:
            - If sequence_len is set: (X, y, diff_cols, norm)
            - If sequence_len is None: (df_processed, diff_cols, norm)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
        if 'timestamp' in df.columns:
            df.drop('timestamp', axis=1)

    df = df.resample(sampling_period).ffill().dropna()
    if df.shape[0] < sequence_len:
        raise ValueError(f'Sample too short: Expected length: {sequence_len} of {sampling_period} periods, but got {df.shape[0]}')

    df = df.apply(lambda col: apply_kalman_filtering(col).squeeze())

    if diff_cols is None:

        stationarity_dict = check_stationarity(df)

        diff_cols = []

        for col, stat in stationarity_dict.items():
            if not stat:
                df[col] = df[col].diff()
                df.dropna(inplace=True)
                diff_cols.append(col)

    else:
        for col in diff_cols:
            df[col] = df[col].diff()
            df.dropna(inplace=True)

    if norm is None:

        norm = StandardScaler()

        norm.fit(df)

    df_normalized = norm.transform(df)

    if sequence_len is not None:
        x, y = to_sequence(df_normalized, seq_len=sequence_len)
        return x, y, diff_cols, norm
    else:
        return df, diff_cols, norm


def inverse_transform_single_column(scaler, column_data, column_index):
    """
    Reverses the StandardScaler normalization for a specific column index.

    Useful for converting model predictions back to the original physical units.

    Args:
        scaler (StandardScaler): The fitted scaler instance.
        column_data (np.ndarray or float): The normalized value(s).
        column_index (int): The index of the column in the scaler.

    Returns:
        np.ndarray or float: The value(s) scaled back to original units.
    """
    mean = scaler.mean_[column_index]
    scale = scaler.scale_[column_index]

    original_values = (column_data * scale) + mean

    return original_values
