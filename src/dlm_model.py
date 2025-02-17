import numpy
import matplotlib.pyplot as plt


class DLM(object):

    def __init__(self, F, G, V, W):
        self.F = F  # Dims: (n, n)
        self.G = G  # Dims: (m, n)
        self.V = V  # Dims: (n, n)
        self.W = W  # Dims: (m, m)

        self.state_mean = None  # Dims: (n, 1)
        self.state_cov = None   # Dims: (n, n)

    def initialize(self, state_mean, state_cov):
        self.state_mean = state_mean    # Dims: (n, 1)
        self.state_cov = state_cov      # Dims: (n, n)

    def update(self, y):
        # Prediction
        state_mean_pred = self.F @ self.state_mean
        state_cov_pred = self.F @ self.state_cov @ self.F.T + self.V

        # Update
        y_pred = self.G @ state_mean_pred
        y_cov = self.G @ state_cov_pred @ self.G.T + self.W
        K = state_cov_pred @ self.G.T @ numpy.linalg.inv(y_cov)
        self.state_mean = state_mean_pred + K @ (y - y_pred)
        self.state_cov = state_cov_pred - K @ y_cov @ K.T

    def predict(self):
        # Predict the observation (not the state)
        obs_mean = self.G @ (self.F @ self.state_mean)
        obs_cov = self.G @ (self.F @ self.state_cov @ self.F.T + self.V) @ self.G.T + self.W
        return obs_mean, obs_cov


if __name__ == "__main__":

    # Test stock data
    stock = [60, 62, 65, 63, 59, 57, 55, 58, 59]
    F = numpy.array([[1, 1], [0, 1]])  # State transition matrix
    G = numpy.array([[1, 0]])          # Observation matrix
    V = numpy.array([[1, 0], [0, 1]])  # State noise covariance
    W = numpy.array([[1]])             # Observation noise covariance

    dlm = DLM(F, G, V, W)
    dlm.initialize(numpy.array([[60], [2]]), numpy.array([[1, 0], [0, 1]]))

    results = []

    for value in stock:
        results.append(dlm.predict())
        dlm.update(numpy.array([[value]]))

    # Plot the results
    predictions = [result[0][0][0] for result in results]
    prediction_vars = [result[1][0][0] for result in results]

    plt.plot(stock, label="True values")
    plt.plot(predictions, label="Predictions")
    plt.fill_between(
        range(len(stock)),
        [p - numpy.sqrt(v) for p, v in zip(predictions, prediction_vars)],
        [p + numpy.sqrt(v) for p, v in zip(predictions, prediction_vars)],
        color='gray', alpha=0.5, label="Uncertainty"
    )
    plt.legend()
    plt.show()