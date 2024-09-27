import numpy as np
from models.base_model import Model, PrivateAttr
from sklearn.linear_model import Lasso as SkLasso


class Lasso(Model):
    """
    Lasso Regression model.

    This class wraps the scikit-learn Lasso regression model to fit the
    interface defined by the Model base class.
    """

    _lasso_model: SkLasso = PrivateAttr()

    def __init__(self) -> None:
        """
        Initialize the Lasso model.
        """
        super().__init__()
        self._lasso_model = SkLasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.
        """
        self._lasso_model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the provided observations using the
        fitted Lasso regression model.

        Returns:
            np.ndarray: The predicted target values.
        """
        return self._lasso_model.predict(observations)
