# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':

    # Data Loading
    df = pd.read_csv("Resources/delaney_solubility_with_descriptors.csv")

    # Data Preparation
    y = df["logS"]
    x = df.drop("logS", axis=1)

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # applying the model to make a prediction

    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)

    # Evaluate model performance

    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    # Data Printing
    lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ["Method", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]

    print(lr_results)

