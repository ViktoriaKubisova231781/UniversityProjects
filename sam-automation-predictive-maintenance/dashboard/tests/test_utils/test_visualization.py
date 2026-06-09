import pytest
import pandas as pd
import numpy as np
import matplotlib

# CRITICAL: Use 'Agg' backend to prevent GUI windows from popping up
# This must be done BEFORE importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.visualization import plot_anomalies_scatter, plot_anomalies_overlay


@pytest.fixture
def dummy_data():
    """Generates a simple sine wave for plotting."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    return pd.Series(y)


def test_plot_anomalies_scatter(dummy_data):
    """
    Test that the scatter plotter returns a valid Figure.
    The function expects 'predictions < 0' to indicate anomalies.
    """
    # 1. Create fake predictions
    # 1.0 = Normal, -1.0 = Anomaly (based on your legacy logic)
    preds = np.ones(len(dummy_data))
    preds[10:20] = -1.0

    # 2. Generate Plot
    fig = plot_anomalies_scatter(dummy_data, preds)

    # 3. Assertions
    assert isinstance(fig, plt.Figure)
    # Check that at least one subplot (Axes) was created
    assert len(fig.axes) > 0

    # Optional: Check if collections (scatter points) were added
    # The function adds a Line2D (main line) and a PathCollection (scatter)
    ax = fig.axes[0]
    # We expect 1 line (the main data)
    assert len(ax.lines) == 1
    # We expect 1 collection (the red dots for anomalies)
    assert len(ax.collections) == 1


def test_plot_anomalies_overlay(dummy_data):
    """
    Test that the overlay plotter returns a valid Figure.
    This function uses LineCollection to color segments.
    """
    # 1. Create fake boolean predictions (True = Anomaly)
    preds = np.zeros(len(dummy_data), dtype=bool)
    preds[50:60] = True

    # 2. Generate Plot
    fig = plot_anomalies_overlay(dummy_data, preds)

    # 3. Assertions
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

    # Check that the LineCollection was added
    ax = fig.axes[0]
    # This function uses collections, not standard lines
    assert len(ax.collections) == 1


def test_plotting_does_not_crash_on_flat_line(dummy_data):
    """Ensure math doesn't break on constant data."""
    flat_data = pd.Series(np.zeros(100))
    preds = np.zeros(100)

    fig = plot_anomalies_scatter(flat_data, preds)
    assert isinstance(fig, plt.Figure)
