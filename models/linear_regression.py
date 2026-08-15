import numpy as np


class LinearRegression:

    # Initialize model hyperparameters and tracking variables
    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        tolerance=0.001
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance

        self.weights = None
        self.bias = None
        self.losses = []

    # Generate predictions using the current model parameters
    def predict(self, x):

        predictions = np.dot(x, self.weights) + self.bias

        return predictions

    # Compute Mean Squared Error loss
    def compute_cost(self, y_true, y_pred):

        Error = y_true - y_pred

        cost = (Error) ** 2

        return sum(cost) / len(y_true)

    # Update model parameters using Gradient Descent
    def gradient_decent(self, x, y_true, y_pred):

        n_samples = x.shape[0]

        error = y_pred - y_true

        dw = np.dot(x.T, error) / n_samples
        db = np.mean(error)

        self.weights = (
            self.weights
            - self.learning_rate * dw
        )

        self.bias = (
            self.bias
            - self.learning_rate * db
        )

    # Train the model using batch Gradient Descent
    def fit(self, x, y):

        # Initialize model parameters
        n_features = x.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            # Generate predictions
            y_pred = self.predict(x)

            # Calculate training loss
            cost = self.compute_cost(y, y_pred)

            # Store loss history for analysis and visualization
            self.losses.append(cost)

            # Stop training when improvement becomes very small
            if len(self.losses) > 1:

                difference = abs(
                    self.losses[-1]
                    - self.losses[-2]
                )

                relative_change = (
                    difference
                    / self.losses[-2]
                )

                if relative_change < self.tolerance:
                    print(
                        f"Early Stopping At Epoch: {epoch}")
                    break

            # Update model parameters
            self.gradient_decent( x,y,y_pred)