from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, PrivateAttr


class Model(BaseModel, ABC):
    """
    Abstract base class for machine learning models.

    Attributes:
        _parameters (dict): A dictionary to store model parameters.
    """

    _parameters: dict = PrivateAttr(default_factory=dict)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the provided observations.
        """
        pass
