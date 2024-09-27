import pandas as pd
from models.k_nearest_neighbors import KNearestNeighbors
from models.multiple_linear_regression import MultipleLinearRegression
from models.sklearn_wrap import Lasso


def main():
    dp = pd.read_csv(r"oop-24-25-assignment-1-group-8\data\Real estate.csv")
    dk = pd.read_csv(
        r"oop-24-25-assignment-1-group-8\data\iphone_purchase_records.csv"
        )

    observations = dp.iloc[:, :-1].to_numpy()
    ground_truth = dp.iloc[:, -1].to_numpy()

    observations_k = dk.iloc[:, :-1].to_numpy()
    ground_truth_k = dk.iloc[:, -1].to_numpy()

    print("=== Multiple Linear Regression ===")
    mlr = MultipleLinearRegression()
    mlr.fit(observations, ground_truth)
    mlr_predictions = mlr.predict(observations)
    print("MLR Predictions:", mlr_predictions)

    print("=== Lasso Regression ===")
    lasso = Lasso()
    lasso.fit(observations, ground_truth)
    lasso_predictions = lasso.predict(observations)
    print("Lasso Predictions:", lasso_predictions)

    print("=== K-Nearest Neighbors (KNN) ===")
    knn = KNearestNeighbors()
    knn.fit(observations_k, ground_truth_k)
    knn_predictions = knn.predict(observations_k)
    print("KNN Predictions:", knn_predictions)


if __name__ == "__main__":
    main()
