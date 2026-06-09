from sklearn.neighbors import LocalOutlierFactor  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore
from scipy.special import expit  # type: ignore


def apply_LOF(df, contamination=0.001):
    """
    Applies Local Outlier Factor (LOF) to the input dataframe to detect anomalies.

    Args:
        df (pd.DataFrame or np.ndarray): Input data.
        contamination (float): The amount of contamination of the data set, i.e.,
                               the proportion of outliers in the data set.

    Returns:
        tuple: (predictions, anomaly_scores)
            - predictions (np.ndarray): 1 for inliers, -1 for outliers.
            - anomaly_scores (np.ndarray): Negative outlier factor scores.
    """
    lof = LocalOutlierFactor(contamination=contamination)
    predictions = lof.fit_predict(df)
    anomaly_scores = lof.negative_outlier_factor_

    return predictions, anomaly_scores


def apply_iforest(df, contamination=0.001):
    """
    Applies Isolation Forest to the input dataframe to detect anomalies.

    Converts decision function scores into probability-like scores using the expit function.

    Args:
        df (pd.DataFrame or np.ndarray): Input data.
        contamination (float): The amount of contamination of the data set.

    Returns:
        tuple: (predictions, probabilities)
            - predictions (np.ndarray): 1 for inliers, -1 for outliers.
            - probabilities (np.ndarray): Anomaly probabilities (0 to 1).
    """
    iso = IsolationForest(contamination=contamination)
    iso.fit(df)

    predictions = iso.predict(df)
    anomaly_scores = iso.score_samples(df)

    probabilities = expit(anomaly_scores)

    return predictions, probabilities
