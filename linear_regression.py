# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def linear_regression(x_train, y_train, x_test, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # applying the model to make a prediction

    y_lr_train_pred = lr.predict(x_train)
    y_lr_test_pred = lr.predict(x_test)

    # Evaluate model performance

    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    plt.figure(figsize=(5, 5))
    plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)

    z = np.polyfit(y_train, y_lr_train_pred, 1)
    p = np.poly1d(z)

    plt.plot(y_train, p(y_train), '#F8766D')
    plt.ylabel("Predict logS")
    plt.xlabel("Experimental logS")
    plt.show()

    return pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
