# -*- coding: utf-8 -*-
"""Main script for Project 1 submission."""

from helpers import load_csv_data, create_csv_submission
from implementations import *  # import your trained models

# import preprocessing functions if you have them in a separate file


def main():
    # 1. Load data
    data_path = "data/dataset"  # or adjust path depending on your folder layout
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    # 2. Preprocessing & Feature Engineering
    # TO DO: apply your best feature transformations here

    # 3. Train your best model
    # TO DO: fit your model using your implemented weights/parameters
    # e.g., weights, loss = least_squares(y_train, x_train)

    # 4. Make predictions on test data
    # y_pred = predict(x_test, weights) # implement a helper or use your model's formula

    # 5. Output submission file
    # OUTPUT_PATH = "sub_prediction.csv"
    # create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print("Submission file generated successfully!")


if __name__ == "__main__":
    main()
