import copy
from collections import Counter

import numpy as np
from models.base_model import Model
from pydantic import Field, field_validator


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors (KNN) model for classification.

    Attributes:
        k (int): Number of neighbors to consider for classification.
    """

    k: int = Field(title="Number of neighbors", default=3)

    @field_validator("k")
    def k_greater_than_zero(cls, value):
        """
        Validator to ensure k is greater than 0.
        """
        if value <= 0:
            raise ValueError("k must be greater than 0")
        return value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the KNN model to the provided observations and ground truth.
        """
        self._parameters = {
            "observations": observations, "ground_truth": ground_truth
            }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the provided observations.
        """
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with arguments")

        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> any:
        """
        Predict the target value for a single observation.

        Returns:
            any: The predicted target value.
        """
        distances = np.linalg.norm(
            observation - self._parameters["observations"], axis=1
        )
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[: self.k]
        k_near_label = [self._parameters["ground_truth"][i] for i in k_indices]
        most_common = Counter(k_near_label).most_common(1)
        return most_common[0][0]

    @property
    def parameters(self):
        """
        Get the model parameters.

        Returns:
            dict: A deep copy of the model parameters.
        """
        return copy.deepcopy(self._parameters)
