from HDLM import HDLM
import numpy as np
import matplotlib.pyplot as plt
from dlm_model import DLM
# Seed
np.random.seed(42)

def sine_wave(eps=0.1):
    x = np.arange(0, 2*np.pi, 0.1)
    y = np.sin(x) + np.random.normal(0, eps, len(x))

    test_data_x = np.arange(2*np.pi, 4*np.pi, 0.1)
    test_data_y = np.sin(test_data_x) + np.random.normal(0, eps, len(test_data_x))
    return y, test_data_y

def main():
    eps = 1
    series = [sine_wave(eps) for _ in range(4)]

    F = np.array([[1, 1], [0, 1]])  # State transition matrix
    G = np.array([[1, 0]])          # Observation matrix
    V = np.array([[eps**2, 0], [0, eps**2]])  # State noise covariance
    W = np.array([[eps**2]])        # Observation noise covariance

    # Si el prior es abierto, el modelo se ajusta más rápido a los datos
    # Hacer un analisis de los datos para ver como inicializar. Sino, un prior abierto y no pierdes informacion 

    models = [DLM(F, G, V, W) for _ in range(4)]

    for i, model in enumerate(models):
        model.initialize(np.array([[0], [0]]), np.array([[1, 0], [0, 1]]))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for i, (train_data, test_data) in enumerate(series):
        for value in train_data:
            models[i].update(np.array([[value]]))

        results = []
        for value in test_data:
            pred_mean, pred_cov = models[i].predict()
            results.append((pred_mean[0, 0], pred_cov[0, 0]))
            models[i].update(np.array([[value]]))

        predictions = [r[0] for r in results]
        prediction_vars = [r[1] for r in results]

        # Align x-axis properly
        train_x = np.arange(len(train_data))
        test_x = np.arange(len(train_data), len(train_data) + len(test_data))

        axs[i//2, i%2].plot(train_x, train_data, label="Training Data", color="blue")
        axs[i//2, i%2].plot(test_x, test_data, label="True Test Data", color="green", linestyle="dashed")
        axs[i//2, i%2].plot(test_x, predictions, label="Predictions", color="red")

        lower_bound = [pred - 1.96 * np.sqrt(var) for pred, var in zip(predictions, prediction_vars)]
        upper_bound = [pred + 1.96 * np.sqrt(var) for pred, var in zip(predictions, prediction_vars)]

        axs[i//2, i%2].fill_between(test_x, lower_bound, upper_bound, color="red", alpha=0.3)

        axs[i//2, i%2].legend()
        axs[i//2, i%2].set_title(f"Series {i+1}")

    plt.tight_layout()
    plt.show()
    


if __name__=="__main__":

    main()