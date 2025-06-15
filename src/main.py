from reconciliation import BottomUpReconciliation, TopDownReconciliation
from hierarchy import HierarchicalTree
import pandas as pd
from models.dlm_model import DLM  # Your DLM class
import numpy as np
import random

random.seed(42)

def sine_wave(eps=0.1):
    """
    Generate a sine wave with added noise.
    """
    x = np.arange(0, 2 * np.pi, 0.1)
    y = np.sin(x) + np.random.normal(0, eps, len(x))
    return y


def test_bottom_up(data, hierarchy):

    # Initialize the hierarchical tree
    tree = HierarchicalTree(hierarchy)

    # Fit DLM models for all leaf nodes and generate predictions
    leaf_nodes = tree.get_leaves()
    predictions = {}

    # Number of steps to forecast
    forecast_steps = 2

    # DLM parameters (you can customize these)
    F = np.array([[1, 1], [0, 1]])  # State transition matrix
    G = np.array([[1, 0]])          # Observation matrix
    V = np.array([[1, 0], [0, 1]])  # State noise covariance
    W = np.array([[1]])             # Observation noise covariance

    for leaf in leaf_nodes:
        # Initialize DLM
        dlm = DLM(F, G, V, W)
        initial_state_mean = np.array([[data_df[leaf.name].iloc[0]], [0]])  # Initial state mean
        initial_state_cov = np.array([[1, 0], [0, 1]])                      # Initial state covariance
        dlm.initialize(initial_state_mean, initial_state_cov)

        # Fit DLM to the leaf node's time series
        for value in data_df[leaf.name]:
            dlm.update(np.array([[value]]))

        # Generate predictions
        leaf_predictions = []
        for _ in range(forecast_steps):
            obs_mean, _ = dlm.predict()
            leaf_predictions.append(obs_mean[0][0])
            
            # Update the DLM with the predicted value
            dlm.update(obs_mean)


        predictions[leaf.name] = leaf_predictions

    print(predictions)

    pred_df = pd.DataFrame(predictions)

    # Reconcile the predictions
    reconciler = BottomUpReconciliation()
    reconciled_data = reconciler.aggregate(pred_df, tree)

    print(reconciled_data)


def test_top_down(data, hierarchy):
    
    # Initialize the hierarchical tree
    tree = HierarchicalTree(hierarchy)

    # Fit DLM models for all leaf nodes and generate predictions
    leaf_nodes = tree.get_leaves()
    predictions = {}

    # Number of steps to forecast
    forecast_steps = 2

    # DLM parameters (you can customize these)
    F = np.array([[1, 1], [0, 1]])  # State transition matrix
    G = np.array([[1, 0]])          # Observation matrix
    V = np.array([[1, 0], [0, 1]])  # State noise covariance
    W = np.array([[1]])             # Observation noise covariance

    # We will predict the root node
    root = tree.root

    # Initialize DLM
    dlm = DLM(F, G, V, W)

    # Initialize the state mean and covariance
    initial_state_mean = np.array([[data_df[root.name].iloc[0]], [0]])  # Initial state mean
    initial_state_cov = np.array([[1, 0], [0, 1]])                      # Initial state covariance
    dlm.initialize(initial_state_mean, initial_state_cov)

    # Fit DLM to the root node's time series
    for value in data_df[root.name]:
        dlm.update(np.array([[value]]))

    # Generate predictions
    root_predictions = []
    for _ in range(forecast_steps):
        obs_mean, _ = dlm.predict()
        root_predictions.append(obs_mean[0][0])
        
        # Update the DLM with the predicted value
        dlm.update(obs_mean)

    predictions[root.name] = root_predictions

    # Reconcile the predictions
    reconciler = TopDownReconciliation("average")   # Only average
    reconciled_data = reconciler.aggregate(pd.DataFrame(predictions), tree)

    print(reconciled_data)


if __name__ == "__main__":
    # Example data
    # Create different random sine waves with noise for 20 time steps
    data = {
        'City1': sine_wave(0.1),
        'City2': sine_wave(0.2),
        'City3': sine_wave(0.3),
        'Region1': sine_wave(0.4),
        'Region2': sine_wave(0.5),
        'Total': sine_wave(0.6)
    }

    # Convert data to a DataFrame
    data_df = pd.DataFrame(data)

    # Example hierarchy
    hierarchy = {
        'Total': ['Region1', 'Region2'],
        'Region1': ['City1'],
        'Region2': ['City2', 'City3']
    }

    # test_bottom_up(data, hierarchy)

    test_top_down(data, hierarchy)

    



    
