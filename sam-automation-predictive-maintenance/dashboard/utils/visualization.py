import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_anomalies_scatter(df, predictions):
    """
    Plots the time series data as a continuous line and highlights anomalies
    as scattered points.

    This function assumes anomalies are marked with negative values in the
    prediction array (e.g., -1 from Isolation Forest or LOF).

    Args:
        df (pd.DataFrame or np.ndarray): The univariate time series data.
        predictions (np.ndarray): Array of anomaly scores or labels, where
                                  values < 0 indicate anomalies.

    Returns:
        plt.Figure: The generated matplotlib figure with the scatter plot.
    """
    x = np.arange(0, len(df))
    y = df.values

    # Handle case where predictions is a dict or array
    if isinstance(predictions, dict):
        # If passed a dict (error_dict), we need a specific way to determine anomalies.
        # This function seems legacy, assuming 'predictions' is the raw values.
        # We'll assume the user passes a numpy array here.
        pass

    mask = predictions < 0  # This logic seems specific to your old detector?
    # If using reconstruction error, mask would be: error > threshold.

    # Assuming 'predictions' here is actually the anomaly boolean mask or similar
    # For safety, let's just plot what we are given

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y, color="black", zorder=1)

    if mask.any():
        plt.scatter(x[mask], y[mask], color="red", zorder=2, s=20)

    plt.close()  # Close internal plt reference
    return fig


def plot_anomalies_overlay(df, predictions):
    """
    Visualizes the time series using a LineCollection, coloring segments differently
    based on their anomaly status.

    Segments corresponding to anomalies are colored red, while normal segments
    are colored blue.

    Args:
        df (pd.DataFrame or np.ndarray): The univariate time series data.
        predictions (np.ndarray): A boolean array or binary labels indicating
                                  anomalies (True/1 for anomaly).

    Returns:
        plt.Figure: The generated matplotlib figure containing the colored line segments.
    """
    x = np.arange(0, len(df))
    y = df.values.flatten()  # Ensure 1D

    segments = np.column_stack([x, y])
    points = segments.reshape(-1, 1, 2)
    segments = np.hstack([points[:-1], points[1:]])

    # Legacy logic: assuming 'predictions' contains negative values for anomalies?
    # Or implies 'predictions' is the boolean mask?
    # Let's assume predictions is a boolean array for "Is Anomaly"
    anomalies = np.array(predictions, dtype=bool)

    # Map True->Red, False->Blue
    colors = np.where(anomalies[:-1], "red", "blue")

    lc = LineCollection(segments, colors=colors, linewidth=2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    plt.close()
    return fig
