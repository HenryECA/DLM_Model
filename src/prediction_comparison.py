from models.arima import ARIMA_Model
from models.lstm import LSTM
from models.conv_1D import Conv1D
from models.mlp import MLP
from utils import rmse, mae, crps


import numpy as np
import matplotlib.pyplot as plt
import torch
from pydlm import dlm, trend, seasonality, dynamic
import copy
import random

models_check = {
    "ARIMA": False, 
    "LSTM": False,
    "DLM": False,
    "Conv1D": False,
    "MLP": True
}

STD_DLM = 2

days = [1,3,7,14,30]

# Seed
random.seed(42)

def create_inout_sequences(data, tw = 10):
    inout_sq = []
    L = len(data)
    for i in range(L - tw):
        train_seq = data[i:i + tw]
        train_label = data[i + tw:i + tw + 1]
        inout_sq.append((train_seq, train_label))
    return inout_sq

def one_step_prediction(): 

    data = np.load("data/synthetic_series.npy")[:500]

    train = data[:400]
    test = data[400:]

    model_results = {model: {h: [] for h in days} for model in models_check.keys()}
    model_variances = {model: {h: {"lower": [], "upper": []} for h in days} for model in models_check.keys()}

    # We will compute metrics like RMSE, MAE for each model

    model_metrics = {model: {h: {"RMSE": None, "MAE": None, "CRPS": None} for h in days} for model in models_check.keys()}

    if models_check["ARIMA"]:
        print("ARIMA")
        arima = ARIMA_Model(train)
        arima.fit()
        base_model = arima.best_model
        variances = {h: [] for h in days}

        for h in model_results["ARIMA"].keys():
            arima.best_model = base_model
            # h will be the horizon, which will mark the step taken in the prediction
            # We will need to take "h" steps in the prediction
            prev_model = copy.deepcopy(arima)
            for i in range(0, len(test) - h, h):
                # Method 1 -> with horizon steps directly
                # # Now predict h steps
                # prediction = arima.predict(h)
                # model_results["ARIMA"][h].append(prediction[0])
                # variances[h].append(prediction[1])

                # # Now update the model with the h last observations
                # for v in test[i:i + h]:
                #     arima.append(v)

                # Method 2 -> with 1 step predictions over the horizon and update with predictions
                for j in range(h):
                    prediction = arima.predict(1)
                    model_results["ARIMA"][h].append(prediction[0])
                    variances[h].append(prediction[1])
                    arima.append(prediction[0])

                arima = copy.deepcopy(prev_model)
                # Now update the model with the h last observations
                for v in test[i:i + h]:
                    arima.append(v)
                
                prev_model = copy.deepcopy(arima)


        # Plot the results
        plt.plot(test, color="black", label="True")
        for h in model_results["ARIMA"].keys():
            # Each prediction is a tuple (list of h elements, list of 2 elements). Keep only the first list and conact
            model_results["ARIMA"][h] = np.concatenate([x for x in model_results["ARIMA"][h]])
            model_variances["ARIMA"][h]["lower"] = np.concatenate([x[:,0] for x in variances[h]])
            model_variances["ARIMA"][h]["upper"] = np.concatenate([x[:,1] for x in variances[h]])
            plt.plot(model_results["ARIMA"][h], label=f"ARIMA: {h}")
            plt.fill_between(range(len(model_results["ARIMA"][h])), model_variances["ARIMA"][h]["lower"], model_variances["ARIMA"][h]["upper"], alpha=0.3)
        plt.legend()
        plt.title("ARIMA")
        plt.show()

        print("ARIMA done")

    if models_check["LSTM"]:
        print("LSTM")
        inout_seq = create_inout_sequences(data, 10)
        train_seq, test_seq = inout_seq[:len(train)-10], inout_seq[len(train)-10:]
        lstm = LSTM(input_size=1, hidden_layer_size=150, output_size=1)
        lstm_results = []
        lstm.fit(train_seq, epochs=100)

        for h in model_results["LSTM"].keys():
            for i in range(0, len(test_seq) - h, h):
                last_sequence = test_seq[i][0]
                for j in range(h):
                    tensor_seq = torch.FloatTensor(last_sequence)
                    result = lstm(tensor_seq)
                    model_results["LSTM"][h].append(result.detach().numpy()[0])
                    last_sequence = np.concatenate((last_sequence[1:], result.detach().numpy()))
        
        print("LSTM done")

        # Plot the test data in red
        plt.plot(test, color="black", label="True")

        for h in model_results["LSTM"].keys():
            plt.plot(model_results["LSTM"][h], label=f"LSTM: {h}")
        plt.legend()
        plt.title("LSTM")
        plt.show()


    if models_check["DLM"]:
        print("DLM")

        my_dlm = dlm(train) + trend(degree=1, name="linear_trend", w=1.0) \
                        + seasonality(period=int(2*np.pi), name="sine_wave", w=1.0)
        my_dlm.fit()
        variances = {h: [] for h in days}

        for h in model_results["DLM"].keys():
            temp_dlm = copy.deepcopy(my_dlm)
            
            for i in range(0, len(test), h):  # Changed to handle edge cases
                chunk = test[i:i+h]
                if len(chunk) == 0:
                    continue
                    
                # Predict then update
                forecast = temp_dlm.predictN(h)
                model_results["DLM"][h].append(forecast[0])
                variances[h].append(forecast[1])
                
                # Update model with new data
                temp_dlm.append(chunk)
                temp_dlm.fit()  # Full refit is safer than forwardFilter for multiple steps

        # Plotting remains the same
        plt.plot(test, color="black", label="True")
        for h in model_results["DLM"].keys():
            # Flatten predictions
            model_results["DLM"][h] = [item for sublist in model_results["DLM"][h] for item in sublist]
            # We get the variances and multiplicate by the standard deviation and add the mean
            model_variances["DLM"][h]["lower"] = [item for sublist in variances[h] for item in sublist]
            model_variances["DLM"][h]["upper"] = [item for sublist in variances[h] for item in sublist]
            for i in range(len(model_variances["DLM"][h]["lower"])):
                model_variances["DLM"][h]["lower"][i] = model_results["DLM"][h][i] - STD_DLM * model_variances["DLM"][h]["lower"][i]
                model_variances["DLM"][h]["upper"][i] = model_results["DLM"][h][i] + STD_DLM * model_variances["DLM"][h]["upper"][i]
            plt.plot(model_results["DLM"][h][:len(test)], label=f"h={h}")  # Truncate to test length
            plt.fill_between(range(len(model_results["DLM"][h][:len(test)])), model_variances["DLM"][h]["lower"][:len(test)], model_variances["DLM"][h]["upper"][:len(test)], alpha=0.3)
        plt.legend()
        plt.title("DLM Multi-step Forecasts")
        plt.show()

    if models_check["Conv1D"]:
        print("Conv 1D")
        inout_seq = create_inout_sequences(data, 10)
        train_seq, test_seq = inout_seq[:len(train)-10], inout_seq[len(train)-10:]
        conv1d = Conv1D(input_size=1, output_size=64, kernel_size=5, stride=1, padding=0)
        conv1d_results = []
        conv1d.fit(train_seq, epochs=100)

        for h in model_results["Conv1D"].keys():
            for i in range(0, len(test_seq) - h, h):
                last_sequence = test_seq[i][0]
                for j in range(h):
                    tensor_seq = torch.FloatTensor(last_sequence)
                    # Add batch dimension for conv1d
                    tensor_seq = tensor_seq.unsqueeze(0)
                    result = conv1d(tensor_seq).squeeze()
                    model_results["Conv1D"][h].append(result.detach().numpy())
                    last_sequence = np.concatenate((last_sequence[1:], [result.detach().numpy()]))

        print(model_results["Conv1D"])
        print("Conv 1D done")

        # Plot the test data in red
        plt.plot(test, color="black", label="True")

        for h in model_results["Conv1D"].keys():
            plt.plot(model_results["Conv1D"][h], label=f"Conv1D: {h}")
        plt.legend()
        plt.title("Conv1D")
        plt.show()

    if models_check["MLP"]:
        print("MLP")
        inout_seq = create_inout_sequences(data, 10)
        train_seq, test_seq = inout_seq[:len(train)-10], inout_seq[len(train)-10:]
        mlp = MLP(input_size=len(train_seq[0][0]), hidden_sizes=[16, 64, 128, 256, 128, 32, 8], output_size=1)
        mlp_results = []
        mlp.fit(train_seq, epochs=50, loss_fn=torch.nn.MSELoss(), optimizer=torch.optim.Adam, lr=0.0001)
        print("MLP trained")
        for h in model_results["MLP"].keys():
            for i in range(0, len(test_seq) - h, h):
                last_sequence = test_seq[i][0]
                for j in range(h):
                    tensor_seq = torch.FloatTensor(last_sequence)
                    result = mlp(tensor_seq)
                    model_results["MLP"][h].append(result.detach().numpy()[0])
                    last_sequence = np.concatenate((last_sequence[1:], result.detach().numpy()))

        print("MLP done")

        # Plot the test data in red
        plt.plot(test, color="black", label="True")

        for h in model_results["MLP"].keys():
            plt.plot(model_results["MLP"][h], label=f"MLP: {h}")
        plt.legend()
        plt.title("MLP")
        plt.show()


    for model in model_results.keys():
        for h in model_results[model].keys():

            preds = model_results[model][h]
            lower = model_variances[model][h]["lower"]
            upper = model_variances[model][h]["upper"]

            min_len = min(len(test), len(preds), len(lower), len(upper))

            model_metrics[model][h]["RMSE"] = round(rmse(test[:min_len], preds[:min_len]), 3)
            model_metrics[model][h]["MAE"] = round(mae(test[:min_len], preds[:min_len]), 3)

            if model in ["ARIMA", "DLM"]:
                model_metrics[model][h]["CRPS"] = round(
                    crps(test[:min_len], preds[:min_len], lower[:min_len], upper[:min_len]), 3
                )

    # Lets make a plot per metric where in the x axis we have the horizon, on the y axis the value of the metric and each line is a model

    for metric in ["RMSE", "MAE"]:
        plt.figure()
        for model in model_metrics.keys():
            values = [model_metrics[model][h][metric] for h in model_metrics[model].keys()]
            plt.plot(list(model_metrics[model].keys()), values, label=model)
        plt.legend()
        plt.title(metric)
        plt.show()


    
        

if __name__ == "__main__":
    one_step_prediction()