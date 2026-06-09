from pykalman import KalmanFilter  # type: ignore


def apply_kalman_filtering(series, kf=None):
    """
    Applies Kalman Filtering to a time series to reduce noise.

    If no filter is provided, initializes a standard 1D Kalman Filter.
    This method returns the filtered state means (causal, suitable for real-time).

    Args:
        series (pd.Series or np.ndarray): The noisy input time series.
        kf (KalmanFilter, optional): A pre-configured KalmanFilter instance.

    Returns:
        np.ndarray: The filtered state means.
    """
    if kf is None:
        kf = KalmanFilter(
            initial_state_mean=series.iloc[0],
            n_dim_obs=1,
            n_dim_state=1,
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=0.1,
            observation_covariance=1.0,
        )

    means, covariancees = kf.filter(series)

    return means


def apply_kalman_smoothing(series, kf=None):
    """
    Applies Kalman Smoothing to a time series.

    Unlike filtering, smoothing uses future data to refine past estimates
    (non-causal, better for historical analysis).

    Args:
        series (pd.Series or np.ndarray): The noisy input time series.
        kf (KalmanFilter, optional): A pre-configured KalmanFilter instance.

    Returns:
        np.ndarray: The smoothed state means.
    """
    if kf is None:
        kf = KalmanFilter(
            initial_state_mean=series.iloc[0],
            n_dim_obs=1,
            n_dim_state=1,
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=0.1,
            observation_covariance=1.0,
        )

    means, covariancees = kf.smooth(series)

    return means
