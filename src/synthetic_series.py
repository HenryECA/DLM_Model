# from src.hierarchy import HDLM
import numpy as np
import os

import json
# Seed
from utils import seed_all
seed_all(42)

def synthetic_series_experiment(length = 10000, experiment_path = "data/synthetic_series/", folder_name = "example",hierarchy=None, functions=None, weights=None):
    """
    Generate a synthetic hierarchical time series and save it to a specified path.
    Parameters:
        length (int): Length of the synthetic time series.
        experiment_path (str): Path to save the synthetic time series.
        folder_name (str): Name of the folder to save the series.
        hierarchy (dict): Hierarchy structure for the time series.
        functions (dict): Functions to apply on the time series.
        weights (dict): Weights for the hierarchy.
    Returns:
        None
    """

    # Check if the experiment path exists
    if not os.path.exists(os.path.join(experiment_path, folder_name)):
        os.makedirs(os.path.join(experiment_path, folder_name))

    if (hierarchy and weights and functions) == None:
        hierarchy = {
            "O": ["A", "B"], 
            "A": ["AA", "AB"],
            "B": ["BA", "BB"],
            "BB": ["BBA", "BBB", "BBC"],
        }

        weights = {
            "O": {
                "A": 1,
                "B": 1,
            },
            "A": {
                "AA": 1,
                "AB": 1,
            },
            "B": {
                "BA": 1,
                "BB": 1,
            },
            "BB": {
                "BBA": 1,
                "BBB": 1,
                "BBC": 1,
            },
        }

        functions = {
            "AA": lambda t: 5 * np.log1p(t),           # logarithmic
            "AB": lambda t: 50 * np.sin(t / 5),         # scaled sine
            "BA": lambda t: 0.01 * t - 50.,               # linear
            "BBA": lambda t: -0.005 * t + 33,        # linear + offset
            "BBB": lambda t: 50 * np.cos(t / 3),        # cosine
            "BBC": lambda t: 50 * np.exp(-t / 500),      # exponential decay
        }

    t = np.arange(length)

    # Generate series for leaf nodes
    series = {}
    for leaf, func in functions.items():
        series[leaf] = func(t) + np.random.normal(0, 5, length)  # Add noise

    depth = {node: 0 for node in hierarchy}
    def compute_depth(node):
        if node not in hierarchy:
            return 0
        return 1 + max(compute_depth(child) for child in hierarchy[node])
    for node in hierarchy:
        depth[node] = compute_depth(node)
    # Sort non-leaf nodes by increasing depth (children first)
    nodes_sorted = sorted(hierarchy.keys(), key=lambda x: depth[x])
    for parent in nodes_sorted:
        children = hierarchy[parent]
        # Only compute if all children series exist
        if all(child in series for child in children):
            parent_series = np.zeros(length)
            for child in children:
                w = weights[parent].get(child, 1)
                parent_series += w * series[child]
            series[parent] = parent_series

    # Save each series to file
    for name, vals in series.items():
        out_path = os.path.join(experiment_path, folder_name, f"{name}.npy")
        np.save(out_path, vals)
        print(f"Saved series for {name} -> {out_path}")

    # Save the hierarchy, weights and functions
    hierarchy_path = os.path.join(experiment_path, folder_name, "hierarchy.json")

    with open(hierarchy_path, 'w') as f:
        json.dump(hierarchy, f, indent=4)


def read_synthetic_series(experiment_path, folder_name):

    """
    Read the synthetic series, hierarchy, weights and functions from the specified folder.
    Parameters:
        experiment_path (str): Path to the experiment folder.
        folder_name (str): Name of the folder containing the synthetic series.
    Returns:
        hierarchy (dict): Hierarchy structure for the time series.
        series (dict): Generated synthetic time series.
    """

    series = {}
    hierarchy = {}

    # Read the hierarchy, weights and functions
    hierarchy_path = os.path.join(experiment_path, folder_name, "hierarchy.json")
    if os.path.exists(hierarchy_path):
        with open(hierarchy_path, 'r') as f:
            hierarchy = json.load(f)


    # Read the series
    series_folder = os.path.join(experiment_path, folder_name)
    for filename in os.listdir(series_folder):
        if filename.endswith(".npy"):
            name = filename[:-4]  # Remove .npy extension
            series[name] = np.load(os.path.join(series_folder, filename))


    return hierarchy, series

if __name__=="__main__":

    # Example usage
    synthetic_series_experiment(
        length=10000,
        experiment_path="data/synthetic_series/",
        folder_name="example",
        hierarchy=None,
        functions=None,
        weights=None
    )

    hierarchy, series = read_synthetic_series(
        experiment_path="data/synthetic_series/",
        folder_name="example"
    )

    print("Hierarchy:", hierarchy)
