from models.dlm_model import DLM
from models.arima import ARIMA_Model
from models.lstm import LSTM
from general_arch import HierarchicalTree
from utils import download_data
import numpy as np
import matplotlib.pyplot as plt

HORIZON = 4

def test_dlm():
    data, hierarchy = download_data()

    # Separate the data into training and prediction values

    data_train = {key: value[:-HORIZON] for key, value in data.items()}
    data_pred = {key: value[-HORIZON:] for key, value in data.items()}

    pred_cols = hierarchy['ACT'] + hierarchy['Victoria']
    tree = HierarchicalTree(hierarchy, data_train, pred_cols)

    # Make a dlm model per prediction node
    models = {}
    predictions = {}

    for node in pred_cols:
        F = np.array([[1, 1], [0, 1]])
        G = np.array([[1, 0]])
        V = np.array([[1, 0], [0, 1]])
        W = np.array([[1]])

        dlm = DLM(F, G, V, W)
        
        # Adjust initial state mean and covariance to the first data point
        state_mean = np.array([[data_train[node][0]], [0]])
        state_cov = np.array([[1, 0], [0, 1]])
        dlm.initialize(state_mean, state_cov)

        models[node] = dlm

        # Train the model
        for value in data_train[node]:
            dlm.update(np.array([[value]]))

        # Predict the next HORIZON steps
        predictions[node] = []
        for _ in range(HORIZON):
            pred, _ = dlm.predict()
            predictions[node].append(pred[0][0])
            # Update the model with the prediction
            dlm.update(pred)

    # Aggregate the predictions
    updated_pred = tree.aggregate(np.array([predictions[node] for node in pred_cols]))

    # Calculate the error
    error = np.mean(np.abs(updated_pred - np.array([data_pred[node] for node in data_pred.keys()])))
    print("DLM error:", error)

    # Plot the results -> make subplot for every node in the tree. Show train data in blue, prediction in red and real data in green

    fig, axs = plt.subplots(len(data_train.keys()), 1, figsize=(10, 10))

    for i, node in enumerate(data_train.keys()):
        axs[i].plot(data_train[node], label="Train data", color="blue")
        axs[i].plot(np.arange(len(data_train[node]), len(data_train[node]) + HORIZON), data_pred[node], label="Real data", color="green")
        axs[i].plot(np.arange(len(data_train[node]), len(data_train[node]) + HORIZON), updated_pred[i], label="Prediction", color="red")
        axs[i].set_title(node)
    
    plt.show()



def test_arima():
    data, hierarchy = download_data()

    # Separate the data into training and prediction values

    data_train = {key: value[:-HORIZON] for key, value in data.items()}
    data_pred = {key: value[-HORIZON:] for key, value in data.items()}

    pred_cols = hierarchy['ACT'] + hierarchy['Victoria']
    tree = HierarchicalTree(hierarchy, data_train, pred_cols)

    # Make a dlm model per prediction node
    models = {}
    predictions = {}

    for node in pred_cols:
        # Example ARIMA model
        arima = ARIMA_Model(list(data_train[node]))    
        
        model_fit = arima.fit()
        models[node] = model_fit
        predictions[node] = arima.predict(HORIZON)

    # Aggregate the predictions
    updated_pred = tree.aggregate(np.array([predictions[node] for node in pred_cols]))

    # Calculate the error
    error = np.mean(np.abs(updated_pred - np.array([data_pred[node] for node in data_pred.keys()])))
    print("ARIMA error:", error)

    # Plot the results -> make subplot for every node in the tree. Show train data in blue, prediction in red and real data in green

    fig, axs = plt.subplots(len(data_train.keys()), 1, figsize=(10, 10))

    for i, node in enumerate(data_train.keys()):
        axs[i].plot(data_train[node], label="Train data", color="blue")
        axs[i].plot(np.arange(len(data_train[node]), len(data_train[node]) + HORIZON), data_pred[node], label="Real data", color="green")
        axs[i].plot(np.arange(len(data_train[node]), len(data_train[node]) + HORIZON), updated_pred[i], label="Prediction", color="red")
        axs[i].set_title(node)
    
    plt.show()


if __name__ == "__main__":
    test_dlm()
    # 
    # test_arima()