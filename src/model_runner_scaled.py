import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time

import warnings
warnings.filterwarnings("ignore")

from models.arima import ARIMA_Model
from models.lstm import LSTM
from models.conv_1D import Conv1D
from models.mlp import MLP
from models.sma import SMA
from models.ema import EMA
from models.garch import GARCH_Model
from models.dlm_model import DLM
from utils import rmse, mae, crps, nll, save_data_point, seed_all, expected_calibration_error
from plotter import MetricsTablePlotter


PCT_95 = 1.645
PCT_80 = 1.282  # For 80% confidence interval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_all(42)

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

class ModelRunner:
    def __init__(self, data: np.ndarray, train_ratio: float = 0.8, max_horizon=10, window_size=10, get_train=False):
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a NumPy array.")

        self.alpha = 0.8
        self.data = data
        self.train_ratio = train_ratio
        self.h = max_horizon
        self.window_size = window_size
        self.get_train = get_train

        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

        if train_ratio > 1:
            # It is a number of samples
            train_quantity = len(data) - train_ratio
            self.train = data[:train_quantity]
            self.test = data[train_quantity:]

        else:
            split_idx = int(len(data) * train_ratio)
            self.train = data[:split_idx]
            self.test = data[split_idx:]

        # scale train and test sets
        self.train = self.scaler.transform(self.train.reshape(-1, 1)).flatten()
        self.test = self.scaler.transform(self.test.reshape(-1, 1)).flatten()

        self.models = {
            "ARIMA": self.run_arima,
            "LSTM": self.run_lstm,
            "DLM": self.run_dlm,
            "Conv1D": self.run_conv1d,
            "MLP": self.run_mlp, 
            "SMA": self.run_sma,
            "EMA": self.run_ema, 
            "GARCH": self.run_garch
        }

        self.times = {
            model: {
                "training": [],
                "forecast": [],
                "update": []
            } for model in self.models.keys()
        }

        self.params = {
            model : {
                "num_params": 0,
                "params": {}
                } for model in self.models.keys()
        }

        self.predictions = {
            "train": {
                "ARIMA": {hor: [] for hor in range(self.h)},
                "LSTM": {hor: [] for hor in range(self.h)},
                "DLM": {hor: [] for hor in range(self.h)},
                "Conv1D": {hor: [] for hor in range(self.h)},
                "MLP": {hor: [] for hor in range(self.h)},
                "SMA": {hor: [] for hor in range(self.h)},
                "EMA": {hor: [] for hor in range(self.h)}, 
                "GARCH": {hor: [] for hor in range(self.h)}},
            "test": {
                "ARIMA": {hor: [] for hor in range(self.h)},
                "LSTM": {hor: [] for hor in range(self.h)},
                "DLM": {hor: [] for hor in range(self.h)},
                "Conv1D": {hor: [] for hor in range(self.h)},
                "MLP": {hor: [] for hor in range(self.h)},
                "SMA": {hor: [] for hor in range(self.h)},
                "EMA": {hor: [] for hor in range(self.h)}, 
                "GARCH": {hor: [] for hor in range(self.h)}}
        }

        self.variances = {
            "train": {
                "ARIMA": {hor: [] for hor in range(self.h)},
                "LSTM": {hor: [] for hor in range(self.h)},
                "DLM": {hor: [] for hor in range(self.h)},
                "Conv1D": {hor: [] for hor in range(self.h)},
                "MLP": {hor: [] for hor in range(self.h)},
                "SMA": {hor: [] for hor in range(self.h)},
                "EMA": {hor: [] for hor in range(self.h)}, 
                "GARCH": {hor: [] for hor in range(self.h)}},
            "test": {
                "ARIMA": {hor: [] for hor in range(self.h)},
                "LSTM": {hor: [] for hor in range(self.h)},
                "DLM": {hor: [] for hor in range(self.h)},
                "Conv1D": {hor: [] for hor in range(self.h)},
                "MLP": {hor: [] for hor in range(self.h)},
                "SMA": {hor: [] for hor in range(self.h)},
                "EMA": {hor: [] for hor in range(self.h)}, 
                "GARCH": {hor: [] for hor in range(self.h)}}
        }

        self.std = {
            "train": {
                "ARIMA": {hor: [] for hor in range(self.h)},
                "LSTM": {hor: [] for hor in range(self.h)},
                "DLM": {hor: [] for hor in range(self.h)},
                "Conv1D": {hor: [] for hor in range(self.h)},
                "MLP": {hor: [] for hor in range(self.h)},
                "SMA": {hor: [] for hor in range(self.h)},
                "EMA": {hor: [] for hor in range(self.h)}, 
                "GARCH": {hor: [] for hor in range(self.h)}},
            "test": {
                "ARIMA": {hor: [] for hor in range(self.h)},
                "LSTM": {hor: [] for hor in range(self.h)},
                "DLM": {hor: [] for hor in range(self.h)},
                "Conv1D": {hor: [] for hor in range(self.h)},
                "MLP": {hor: [] for hor in range(self.h)},
                "SMA": {hor: [] for hor in range(self.h)},
                "EMA": {hor: [] for hor in range(self.h)}, 
                "GARCH": {hor: [] for hor in range(self.h)}}
        }
   
    def create_inout_sequences(self, data, window_size=10, output_size=1):
        """
        Args:
        data         : sequence (list or 1D array) of floats
        window_size  : length of each input sequence (seq_len)
        output_size  : length of each output sequence (horizon)
        Returns:
        List of (input_seq, output_seq) tuples where
            input_seq  is a list of length window_size
            output_seq is a list of length output_size
        """
        # ensure floats
        data = [float(d) for d in data]
        seqs = []
        last_start = len(data) - window_size - output_size + 1
        for i in range(last_start):
            inp  = data[i : i + window_size]
            outp = data[i + window_size : i + window_size + output_size]
            seqs.append((inp, outp))
        return seqs



    def run_arima(self):
        t_0 = time.time()
        arima = ARIMA_Model()
        arima.fit(self.train)
        self.times["ARIMA"]["training"] = time.time() - t_0

        # The model will be fitted with every new data point in test set after saving the prediction
        t_forecast = 0
        t_update = 0
        for x in self.test:
            t0_f = time.time()
            pred, std = arima.predict(self.h)
            t_forecast += time.time() - t0_f

            # Unscale predictions and standard deviations
            pred_scaled_arr = np.array(pred).reshape(-1, 1)
            pred = self.scaler.inverse_transform(pred_scaled_arr).flatten().tolist()
            std_scaled_arr = np.array(std).reshape(-1, 1)
            std = (std_scaled_arr * self.scaler.scale_[0]).flatten().tolist()  # Assuming StandardScaler
            self.predictions["test"]["ARIMA"] = save_data_point(self.predictions["test"]["ARIMA"], pred, self.h)
            self.std["test"]["ARIMA"] = save_data_point(self.std["test"]["ARIMA"], std, self.h)    
            t0_u = time.time()
            arima.append(x)
            t_update += time.time() - t0_u
        self.times["ARIMA"]["forecast"] = t_forecast/len(self.test)
        self.times["ARIMA"]["update"] = t_update/len(self.test)
                
        self.params["ARIMA"]["num_params"] = arima.get_num_params()
        self.params["ARIMA"]["params"] = arima.get_params()
        return self.predictions["test"]["ARIMA"], self.std["test"]["ARIMA"]

    def run_dlm(self):
        # 1. Build a spec that always has level & trend,
        #    and tells DLM.from_spec to pick up the strongest seasonal frequencies.
        spec = {
            'level': True,
            'trend': True,
            'seasonal': {
                # pick the top 1 frequency by power;
                # you can raise n_components to pick more
                'n_components': 4,
                # optionally, only keep freqs above a minimum power threshold
                'importance_th': 0.01  
            },
            # you could also auto-infer an AR block by checking PACF/ACF
            # 'ar': <some positive integer>
        }

        # 2. Fit the model once on the training set
        t_0 = time.time()
        model = DLM.from_spec(spec, data=self.train, ci=0.95)
        self.times["DLM"]["training"] = time.time() - t_0

        # 3. Rolling‐forecast + update
        t_forecast = 0
        t_update = 0
        for x in self.test:
            # multi-step forecast
            t0_f = time.time()
            preds, lowers, uppers, std = model.predict(horizon=self.h)
            t_forecast += time.time() - t0_f

            # Unscale preds and std
            # preds_scaled_arr = np.array(preds).reshape(-1, 1)
            # preds = self.scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
            # lowers_scaled_arr = np.array(lowers).reshape(-1, 1)
            # lowers = self.scaler.inverse_transform(lowers_scaled_arr).flatten().tolist()
            # uppers_scaled_arr = np.array(uppers).reshape(-1, 1)
            # uppers = self.scaler.inverse_transform(uppers_scaled_arr).flatten().tolist()
            # std_scaled_arr = np.array(std).reshape(-1, 1)
            # std = (std_scaled_arr * self.scaler.scale_[0]).flatten().tolist()  # Assuming StandardScaler

            # store point forecasts
            self.predictions["test"]["DLM"] = save_data_point(
                self.predictions["test"]["DLM"],
                np.array(preds),
                self.h
            )

            # # store CIs as (lower, upper) pairs
            # ci = np.vstack([lowers, uppers]).T
            # self.variances["test"]["DLM"] = save_data_point(
            #     self.variances["test"]["DLM"],
            #     ci,
            #     self.h
            # )

            # store standard deviations
            self.std["test"]["DLM"] = save_data_point(
                self.std["test"]["DLM"],
                np.array(std),
                self.h
            )

            # feed the new observation into the filter
            t0_u = time.time()
            model.update(np.array([[x]]))
            t_update += time.time() - t0_u
        self.times["DLM"]["forecast"] = t_forecast / len(self.test)
        self.times["DLM"]["update"] = t_update / len(self.test)

        self.params["DLM"]["num_params"] = model.get_num_params()
        self.params["DLM"]["params"] = model.get_params()

        return self.predictions["test"]["DLM"], self.variances["test"]["DLM"], self.std["test"]["DLM"]

    def run_lstm(self):
        inout_seq = self.create_inout_sequences(self.scaled_data, self.window_size)
        train_seq = inout_seq[:len(self.train) - self.window_size]
        test_seq = inout_seq[len(self.train) - self.window_size:]

        t0 = time.time()
        model = LSTM(input_size=self.window_size, hidden_size=32, num_layers=2, device=device, dropout=0.2)
        model.to(device)
        model.fit(train_seq, epochs=5, lr=1e-3)
        self.times["LSTM"]["training"] = time.time() - t0


        t_forecast = 0
        t_update = 0
        for x in test_seq:
            t0_f = time.time()
            pred = model.predict(x[0], self.h)
            # pred = model.predict(x[0])

            preds_scaled_arr = np.array(pred).reshape(-1, 1)
            preds_unscaled = self.scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
            t_forecast += time.time() - t0_f
            self.predictions["test"]["LSTM"] = save_data_point(self.predictions["test"]["LSTM"], preds_unscaled, self.h)

            # pred, lower, upper = model.predict(self.h, x[0])
            # lower = np.array(lower).reshape(-1, 1)
            # upper = np.array(upper).reshape(-1, 1)
            # lower_scaled = self.scaler.inverse_transform(lower).flatten().tolist()
            # upper_scaled = self.scaler.inverse_transform(upper).flatten().tolist()
            # preds_scaled_arr = np.array(pred).reshape(-1, 1)
            # preds_unscaled = self.scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
            # self.predictions["test"]["LSTM"] = save_data_point(self.predictions["test"]["LSTM"], preds_unscaled, self.h)
            # self.variances["test"]["LSTM"] = save_data_point(self.variances["test"]["LSTM"], [(l, u) for l, u in zip(lower_scaled, upper_scaled)], self.h)
            t0_u = time.time()
            model.fit([x], epochs=1, lr=1e-5)
            t_update += time.time() - t0_u

        self.times["LSTM"]["forecast"] = t_forecast / len(test_seq)
        self.times["LSTM"]["update"] = t_update / len(test_seq)

        self.params["LSTM"]["num_params"] = model.get_num_params()
        self.params["LSTM"]["params"] = model.get_params()

        return self.predictions["test"]["LSTM"], None

    def run_conv1d(self):
        inout_seq = self.create_inout_sequences(self.scaled_data, self.window_size, output_size=1)
        train_seq = inout_seq[:len(self.train) - self.window_size]
        test_seq = inout_seq[len(self.train) - self.window_size:]

        t0 = time.time()
        model = Conv1D(
            input_length=self.window_size,
            input_size=1,  # Assuming univariate time series 
            output_seq_len=1,  # Assuming we want a single output for each input sequence
            device=device,
            conv_channels=[32, 64, 32],  # Example channel sizes
            kernel_sizes=[3, 1, 3],  # Example kernel sizes
            strides=[1, 1, 1],  # Example strides
            paddings=[1, 1, 1],  # Example paddings
            dropout=0.2  # Example dropout rate
        )
        model.to(device)
        model.fit(train_seq, epochs=5, lr=1e-3)
        self.times["Conv1D"]["training"] = time.time() - t0

        t_forecast = 0
        t_update = 0
        for x in test_seq:    
            t0_f = time.time()  
            pred = model.predict(x[0], self.h)
            preds_scaled_arr = np.array(pred).reshape(-1, 1)
            preds_unscaled = self.scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
            t_forecast += time.time() - t0_f
            self.predictions["test"]["Conv1D"] = save_data_point(self.predictions["test"]["Conv1D"], preds_unscaled, self.h)

            # Online update with the new data point
            t0_u = time.time()
            model.fit([x], epochs=5)
            t_update += time.time() - t0_u
        
        self.times["Conv1D"]["forecast"] = t_forecast / len(test_seq)
        self.times["Conv1D"]["update"] = t_update / len(test_seq)

        self.params["Conv1D"]["num_params"] = model.get_num_params()
        self.params["Conv1D"]["params"] = model.get_params()

        return self.predictions["test"]["Conv1D"], None

    def run_mlp(self):
        # 1) Build sequences
        inout_seq = self.create_inout_sequences(self.scaled_data, self.window_size)
        train_seq = inout_seq[: len(self.train) - self.window_size]
        test_seq  = inout_seq[len(self.train) - self.window_size :]

        t_0 = time.time()
        # 2) instantiate the Pyro-based BayesianMLP
        model = MLP(
            input_size=self.window_size,
            hidden_sizes=[32, 64, 32],
            device=device,
        )

        # 3) initial training on train set
        model.fit(train_seq, epochs=5, lr=1e-3)
        self.times["MLP"]["training"].append(time.time() - t_0)

        t_forecast = 0
        t_update = 0
        # 4) rolling-forecast on test set
        for seq, true_y in test_seq:
            # a) get predictive means & stds for horizon self.h
            t0_f = time.time()
            means = model.predict(
                horizon=self.h,
                last_input=seq,
            )

            # b) un-scale means and stds
            means_arr = np.array(means).reshape(-1, 1)
            means_unscaled = self.scaler.inverse_transform(means_arr).flatten().tolist()
            t_forecast += time.time() - t0_f

            # # For std, multiply by the scaler's scale_ (StdScaler) or by (max-min) for MinMaxScaler
            # # Adjust the line below if you're using a MinMaxScaler instead
            # stds_arr = np.array(stds).reshape(-1, 1)
            # stds_unscaled = (stds_arr * self.scaler.scale_).flatten().tolist()

            # c) save point forecasts and stds
            self.predictions["test"]["MLP"] = save_data_point(
                self.predictions["test"]["MLP"],
                means_unscaled,
                self.h,
            )
            # self.std["test"]["MLP"] = save_data_point(
            #     self.std["test"]["MLP"],
            #     stds_unscaled,
            #     self.h,
            # )

            # d) online update on the most recent example
            t0_u = time.time()
            model.fit(
                train_seq=[(seq, true_y)],
                epochs=5,
                lr=1e-3,
            )
            t_update += time.time() - t0_u

        self.times["MLP"]["forecast"] = t_forecast / len(test_seq)
        self.times["MLP"]["update"] = t_update / len(test_seq)

        # 5) store model parameters
        self.params["MLP"]["num_params"] = model.get_num_params()
        self.params["MLP"]["params"] = model.get_params()

        # 5) return your stored results
        return self.predictions["test"]["MLP"], None

    def run_sma(self):
        inout_seq = self.create_inout_sequences(self.scaled_data, self.window_size)
        train_seq = inout_seq[:len(self.train) - self.window_size]
        test_seq = inout_seq[len(self.train) - self.window_size:]

        # We only need to do the testing part

        model = SMA(window_size=self.window_size)
        self.times["SMA"]["training"] = 0  # No training time for SMA

        t_forecast = 0
        for x in test_seq:  
            t0_f = time.time()
            pred = model.predict(self.h, x[0])
            # unscale predictions
            preds_scaled_arr = np.array(pred).reshape(-1, 1)
            pred = self.scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
            t_forecast += time.time() - t0_f
            self.predictions["test"]["SMA"] = save_data_point(self.predictions["test"]["SMA"], pred, self.h)

        self.times["SMA"]["forecast"] = t_forecast / len(test_seq)
        self.times["SMA"]["update"] = 0  # No update time for SMA

        self.params["SMA"]["num_params"] = model.get_num_params()
        self.params["SMA"]["params"] = model.get_params()

        return self.predictions["test"]["SMA"], self.variances["test"]["SMA"]

    def run_ema(self):
        inout_seq = self.create_inout_sequences(self.scaled_data, 1)
        train_seq = inout_seq[:len(self.train) - 1]
        test_seq = inout_seq[len(self.train) - 1:]

        t0 = time.time()
        model = EMA(alpha=0.1)
        model.fit(train_seq)
        self.times["EMA"]["training"] = time.time() - t0

        t_forecast = 0

        for x in test_seq:
            t0_f = time.time()
            pred = model.predict(self.h, x[0])
            # Unscale predictions
            preds_scaled_arr = np.array(pred).reshape(-1, 1)
            pred = self.scaler.inverse_transform(preds_scaled_arr).flatten().tolist()
            t_forecast += time.time() - t0_f
            self.predictions["test"]["EMA"] = save_data_point(self.predictions["test"]["EMA"], pred, self.h)

            # Fine-tune with the new data point
            # model.fit([x])

        self.times["EMA"]["forecast"] = t_forecast / len(test_seq)
        self.times["EMA"]["update"] = 0

        self.params["EMA"]["num_params"] = model.get_num_params()
        self.params["EMA"]["params"] = model.get_params()

        return self.predictions["test"]["EMA"], self.variances["test"]["EMA"]
    
    def run_garch(self):
        t0 = time.time()
        garch = GARCH_Model()
        garch.fit(self.train)
        self.times["GARCH"]["training"] = time.time() - t0

        # The model will be fitted with every new data point in test set after saving the prediction
        t_forecast = 0
        t_update = 0
        for x in self.test:
            t0_f = time.time()
            pred, var = garch.predict(self.h)
            # Unscale predictions
            pred_scaled_arr = np.array(pred).reshape(-1, 1)
            pred = self.scaler.inverse_transform(pred_scaled_arr).flatten().tolist()
            # Unscale variances
            var_scaled_arr = np.array(var).reshape(-1, 1)
            var = self.scaler.inverse_transform(var_scaled_arr).flatten().tolist()
            t_forecast += time.time() - t0_f
            self.predictions["test"]["GARCH"] = save_data_point(self.predictions["test"]["GARCH"], pred, self.h)
            lower = np.array(pred) - PCT_95 * np.sqrt(var)
            upper = np.array(pred) + PCT_95 * np.sqrt(var)
            self.variances["test"]["GARCH"] = save_data_point(self.variances["test"]["GARCH"], [(l, u) for l, u in zip(lower, upper)], self.h)
            t0_u = time.time()
            garch.append(x)
            t_update += time.time() - t0_u
        
        self.times["GARCH"]["forecast"] = t_forecast / len(self.test)
        self.times["GARCH"]["update"] = t_update / len(self.test)

        self.params["GARCH"]["num_params"] = garch.get_num_params()
        self.params["GARCH"]["params"] = garch.get_params()
                
        return self.predictions["test"]["GARCH"], self.variances["test"]["GARCH"]

    def evaluate(self, results, std):
        metrics = {m: [] for m in ["RMSE", "MAE", "CRPS", "NLL", "ECE"]}
        metrics_se = {m: [] for m in ["RMSE", "MAE", "CRPS", "NLL", "ECE"]}
        test_unscaled = self.scaler.inverse_transform(self.test.reshape(-1, 1)).flatten()
        for h in range(self.h):
            x_values = test_unscaled[h:][:150]  # Get the first 150 values for each horizon
            y_values = results[h][:x_values.shape[0]]
            if std is not None:
                std_h = std[h][:x_values.shape[0]] if std is not None else None

            rmse_value = rmse(x_values, y_values)
            mae_value = mae(x_values, y_values)
            metrics["RMSE"].append(rmse_value[0])
            metrics["MAE"].append(mae_value[0])
            metrics_se["RMSE"].append(rmse_value[1])
            metrics_se["MAE"].append(mae_value[1])
            if std is not None and len(std_h) > 0:
                # Only compute CRPS, NLL, and Calibration if variances are provided
                crps_value = crps(x_values, y_values, std_h, alpha = self.alpha)
                nll_value = nll(x_values, y_values, std_h, alpha = self.alpha)
                calibration_value = expected_calibration_error(x_values, y_values, std_h)

                metrics["CRPS"].append(crps_value[0])
                metrics_se["CRPS"].append(crps_value[1])
                metrics["NLL"].append(nll_value[0])
                metrics_se["NLL"].append(nll_value[1])
                metrics["ECE"].append(calibration_value[0])
                metrics_se["ECE"].append(calibration_value[1])

        return metrics, metrics_se


    def plot_horizon_series(self, active_models, results, out_folder):
        """
        For each horizon h:
        • Plot the first 200 true values vs. each model’s first-200 forecasts.
        • Save as out_folder/horizon_1.png, …, horizon_h.png
        """
        os.makedirs(out_folder, exist_ok=True)
        sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4})
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        })
        palette = sns.color_palette("colorblind", n_colors=len(active_models) + 1)

        for h in range(self.h):
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle(rf"Horizon {h+1}", fontsize=16, y=0.95)

            # true series
            y_true = np.array(self.test[h:][:150])
            x = np.arange(len(y_true))
            ax.plot(x, y_true, label="True", color=palette[0], linewidth=2.0)

            # each model
            for i, model in enumerate(active_models, start=1):
                y_pred = np.array(results[model][h][:len(x)])
                ax.plot(x, y_pred, label=model, linestyle="--", color=palette[i])

            ax.set_xlabel("Time Index", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.tick_params(labelsize=10)
            ax.legend(frameon=False, fontsize=10)
            sns.despine(ax=ax, trim=False)

            fname = os.path.join(out_folder, f"horizon_{h+1}.png")
            fig.tight_layout(rect=[0,0,1,0.93])
            fig.savefig(fname, dpi=300)
            plt.close(fig)

    def plot_dlm_arima_first_forecasts(self, results, std, out_folder):
        """
        Single plot comparing DLM vs ARIMA:
        • x-axis = horizon 1…h
        • y = the first forecast at each horizon
        • errorbars = 80% CI from std
        • Save as out_folder/dlm_arima_first_forecast.png
        """
        # Existing plot code
        horizons = np.arange(1, self.h + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, color_idx in zip(("DLM", "ARIMA"), (0, 1)):
            preds = [results[model][h][0] for h in range(self.h)]
            errs_up = [results[model][h][0] + PCT_80 * std[model][h][0] for h in range(self.h)]
            errs_low = [results[model][h][0] - PCT_80 * std[model][h][0] for h in range(self.h)]
            ax.plot(horizons, preds, label=model, color=sns.color_palette("colorblind")[color_idx], linewidth=2.0)
            ax.fill_between(horizons, errs_low, errs_up, alpha=0.3, color=sns.color_palette("colorblind")[color_idx])
        # ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.8)
        ax.plot(horizons, self.test[:self.h], label="True", color='gray', linestyle='-', linewidth=1.5)
        ax.set_title("First-Step Forecasts with 80% CI", fontsize=14)
        ax.set_xlabel("Horizon", fontsize=12)
        ax.set_ylabel("Forecast Value", fontsize=12)
        ax.set_xticks(horizons)
        ax.tick_params(labelsize=10)
        ax.legend(frameon=False, fontsize=10)
        sns.despine(ax=ax, trim=False)

        fname = os.path.join(out_folder, "dlm_arima_first_forecast.png")
        fig.tight_layout(rect=[0,0,1,0.93])
        fig.savefig(fname, dpi=300)
        plt.close(fig)

        # New two-subplot visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        models              = ["DLM", "ARIMA"]
        n_plot              = 150
        forecast_horizons   = [1, 5, self.h//2, self.h]
        colors              = sns.color_palette("colorblind", len(forecast_horizons))

        for idx, model in enumerate(models):
            ax = axes[idx]

            # 1) Plot the first n_plot true values, aligned 0…n_plot-1
            true_vals = self.test[:n_plot]
            x_true    = np.arange(n_plot)
            ax.plot(x_true,
                    true_vals,
                    label="True (first 150)",
                    color="gray",
                    linewidth=1.5)
            
            # 2) For each horizon, take the first (n_plot - h + 1) forecasts,
            #    and plot them starting at x = (horizon-1)…(horizon-1 + len(fc)—1)
            for i, h_step in enumerate(forecast_horizons):
                if h_step > self.h:
                    continue

                # raw forecasts and std for this horizon
                fc_all = results[model][h_step-1]
                if std is not None:
                    sd_all = std[model][h_step-1]
                else:
                    sd_all = None

                # we only need the first (n_plot - h_step + 1) of them
                n_fc    = n_plot - h_step + 1
                fc_vals = fc_all[:n_fc]
                x_fc    = np.arange(h_step-1, h_step-1 + n_fc)

                # plot the lines
                ax.plot(x_fc,
                        fc_vals,
                        label=f"h={h_step}",
                        color=colors[i],
                        linewidth=1.2)

                # optional 80% CI
                if sd_all is not None:
                    sd_vals = np.array(sd_all[:n_fc])
                    up = fc_vals + PCT_80 * sd_vals
                    lo = fc_vals - PCT_80 * sd_vals
                    ax.fill_between(x_fc, lo, up, alpha=0.3, color=colors[i])

            ax.set_title(f"{model} Forecasts at Multiple Horizons", fontsize=13)
            ax.legend(frameon=False, fontsize=10)
            ax.set_ylabel("Value", fontsize=11)
            ax.tick_params(labelsize=9)
            sns.despine(ax=ax, trim=False)

        axes[-1].set_xlabel("Time index", fontsize=12)
        fig.suptitle("Multi-Horizon Forecasts: DLM vs ARIMA", fontsize=15)
        fig.tight_layout(rect=[0,0,1,0.96])

        # Save new plot
        fname2 = os.path.join(out_folder, "dlm_arima_multihorizon_forecasts.png")
        fig.savefig(fname2, dpi=300)
        plt.close(fig)




    def run_all(self, plot=True, active_models=["ARIMA",  "DLM", "LSTM", "Conv1D", "MLP", "SMA", "EMA", "GARCH"], save_path=None):
        self.all_metrics = {
            "err": {name: {"RMSE": [], "MAE": [], "CRPS": [], "NLL": [], "ECE": []} for name in active_models},
            "err_se": {name: {"RMSE": [], "MAE": [], "CRPS": [], "NLL": [], "ECE": []} for name in active_models}
        }
        for name in active_models:
            self.models[name]()
            metrics = self.evaluate(self.predictions["test"][name], self.std["test"][name])
            self.all_metrics["err"][name] = metrics[0]  # Store only the first element (metrics without SE)
            self.all_metrics["err_se"][name] = metrics[1]  # Store only the second element (metrics with SE)
            print(f"Metrics for {name}:")

        self.test_unscaled = self.scaler.inverse_transform(self.test.reshape(-1, 1)).flatten()

        
        if save_path is not None:
            # Save self.predictions["test"] and self.std["test"] to json files
            import json
            with open(os.path.join(save_path, "predictions_test_forecasting.json"), "w") as f:
                json.dump(self.predictions["test"], f)
            with open(os.path.join(save_path, "std_test_forecasting.json"), "w") as f:
                json.dump(self.std["test"], f)

            # Unscale test and save
            np.save(os.path.join(save_path, "test_set_forecasting.npy"), self.test_unscaled)

            # Also save the metrics
            with open(os.path.join(save_path, "all_metrics_forecasting.json"), "w") as f:
                json.dump(self.all_metrics, f)

            with open(os.path.join(save_path, "times_forecasting.json"), "w") as f:
                json.dump(self.times, f)
            
            clean_params = to_python(self.params)
            with open(os.path.join(save_path, "params_forecasting.json"), "w") as f:
                json.dump(clean_params, f)

            print(f"Results saved to {save_path}")

            # # Save the test set
            # np.save(os.path.join(save_path, "test_set_forecasting.npy"), self.test)
            # self.plot_horizon_series(active_models, self.predictions["test"], os.path.join(save_path, "horizon_series/"))
            # self.plot_dlm_arima_first_forecasts(self.predictions["test"], self.std["test"], os.path.join(save_path, "dlm_arima_first_forecast/"))
            # self.plot_metrics(self.all_metrics)
            # df1, df2, df1_se, df2_se = MetricsTablePlotter.create_dataframes(self.all_metrics, group=5)

            # # Save the dataframes to csv files
            # df1.to_csv(os.path.join(save_path, "metrics_rmse_mae.csv"), index=False)
            # df2.to_csv(os.path.join(save_path, "metrics_crps_nll_calibration.csv"), index=False)
            # df1_se.to_csv(os.path.join(save_path, "metrics_rmse_mae_se.csv"), index=False)
            # df2_se.to_csv(os.path.join(save_path, "metrics_crps_nll_calibration_se.csv"), index=False)


            # MetricsTablePlotter.plot_table_as_image(df1, "Forecast Metrics: RMSE & MAE", "results/metrics_synth_rmse_mae.png")
            # MetricsTablePlotter.plot_table_as_image(df2, "Forecast Metrics: CRPS, NLL & Calibration", "results/metrics_synth_crps_nll_calibration.png")


    def plot_metrics(self, all_metrics):
        # 1. Base style via Seaborn
        sns.set_style("whitegrid", {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
        })
        plt.rcParams.update({
            "text.usetex": False,              # use mathtext instead of LaTeX
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        })

        # 2. Line‐styles (we’ll only use the relevant one per subplot)
        line_styles = {
            "RMSE": "-",
            "MAE": "--",
            "CRPS": "-.",
            "NLL": ":",
            "ECE": (0, (3, 1, 1, 1))
        }

        models = list(all_metrics["err"].keys())
        palette = sns.color_palette("colorblind", n_colors=len(models))
        horizons = None  # placeholder for x‐axis ticks

        # --- Figure 1: RMSE & MAE ---
        fig1, axs1 = plt.subplots(1, 2, figsize=(12, 4), sharex=False, sharey=False)
        fig1.suptitle("Forecast Metrics: RMSE & MAE", fontsize=16, y=0.95)

        for ax, metric in zip(axs1, ["RMSE", "MAE"]):
            for idx, model in enumerate(models):
                values = all_metrics["err"][model][metric]
                values_se = all_metrics["err_se"][model][metric]
                horizons = list(range(1, len(values) + 1))
                ax.plot(
                    horizons,
                    values,
                    label=model,
                    linewidth=1.8,
                    linestyle=line_styles[metric],
                    color=palette[idx]
                )
                # Add error bars
                ax.errorbar(
                    horizons,
                    values,
                    yerr=values_se,
                    fmt='none',
                    ecolor=palette[idx],
                    elinewidth=1,
                    capsize=3,
                    alpha=0.6
                )

            ax.set_title(metric, fontsize=14)
            ax.set_xlabel("Forecast Horizon", fontsize=12)
            ax.set_xticks(horizons)
            ax.tick_params(labelsize=10)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.4)
            sns.despine(ax=ax, trim=False)
            ax.legend(frameon=False, fontsize=9, loc="best")
            

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

        # --- Figure 2: CRPS, NLL & Calibration ---
        fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
        fig2.suptitle("Forecast Metrics: CRPS, NLL & ECE", fontsize=16, y=0.95)

        for ax, metric in zip(axs2, ["CRPS", "NLL", "ECE"]):
            for idx, model in enumerate(models):
                values = all_metrics["err"][model][metric]
                values_se = all_metrics["err_se"][model][metric]
                horizons = list(range(1, len(values) + 1))
                ax.plot(
                    horizons,
                    values,
                    label=model,
                    linewidth=1.8,
                    linestyle=line_styles[metric],
                    color=palette[idx]
                )
                # Add error bars
                ax.errorbar(
                    horizons,
                    values,
                    yerr=values_se,
                    fmt='none',
                    ecolor=palette[idx],
                    elinewidth=1,
                    capsize=3,
                    alpha=0.6
                )

            ax.set_title(metric, fontsize=14)
            ax.set_xlabel("Forecast Horizon", fontsize=12)
            ax.set_xticks(horizons)
            ax.tick_params(labelsize=10)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.4)
            sns.despine(ax=ax, trim=False)
            ax.legend(frameon=False, fontsize=9, loc="best")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
                    

if __name__ == "__main__":
    # t = np.arange(0, 1000, 0.1)
    # data = np.sin(t) + np.random.normal(0, 0.1, len(t))  # Example data
    # runner = ModelRunner(data=data, train_ratio=0.8, max_horizon=15)
    # # runner.run_all(active_models=["ARIMA", "DLM", "LSTM", "Conv1D", "MLP", "SMA", "EMA"], plot=True)
    # runner.run_all(active_models=["SMA"], plot=True)

    data_SnP500 = pd.read_csv("D:\Documentos\ICAI\TFG\Code\DLM Model\data\snp500_data\yfinance\GSPC_history.csv", parse_dates=True, index_col="Date")
    data_SnP500 = data_SnP500["Close"].values
    data_SnP500 = data_SnP500[~np.isnan(data_SnP500)]

    runner_SnP500 = ModelRunner(data=data_SnP500, train_ratio=0.7, max_horizon=30)
    runner_SnP500.run_all(active_models=["DLM", "SMA", "ARIMA", "LSTM", "Conv1D", "MLP", "EMA"], plot=True, save_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\snp500_results\forecasting")

    data_synth = np.load("D:\Documentos\ICAI\TFG\Code\DLM Model\data\synthetic_series\example\O.npy")
    data_synth = data_synth[~np.isnan(data_synth)]

    runner_synth = ModelRunner(data=data_synth, train_ratio=0.8, max_horizon=30)
    runner_synth.run_all(active_models=["DLM", "SMA", "ARIMA", "LSTM", "Conv1D", "MLP", "EMA"], plot=True, save_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\synth_results\forecasting")
