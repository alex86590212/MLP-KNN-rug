import numpy as np
from models.base_model import Model


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model.

    This class implements a multiple linear regression model using the
    ordinary least squares method.

    Methods:
        fit: Fits the model to the provided data.
        predict: Predicts the target values for the provided data.
    """

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the multiple linear regression model to the provided observations
        and ground truth.
        """
        X = np.hstack(
            (
                np.ones((observations.shape[0], 1)),
                np.array(observations, dtype=np.float64),
            )
        )
        y = np.array(ground_truth, dtype=np.float64)

        X_transpose = np.transpose(X)

        self._parameters["coefficients"] = (
            np.linalg.pinv(X_transpose @ X) @ X_transpose @ y
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the provided observations using the
        fitted multiple linear regression model.

        Returns:
            np.ndarray: The predicted target values.
        """
        X = np.hstack(
            (
                np.ones((observations.shape[0], 1)),
                np.array(observations, dtype=np.float64),
            )
        )
        coefficients = self._parameters["coefficients"]
        prediction = np.dot(X, coefficients)

        return prediction
