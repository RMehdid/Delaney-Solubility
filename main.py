import pandas as pd
from sklearn.model_selection import train_test_split

from linear_regression import linear_regression
from random_forest import random_forest

if __name__ == "__main__":
    # Data Loading
    df = pd.read_csv("Resources/delaney_solubility_with_descriptors.csv")

    # Data Preparation
    y = df["logS"]
    x = df.drop("logS", axis=1)

    # Data Splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    file = open("Resources/Results.csv", "w+")
    file.close()

    cols = ["Methode", "Training MSE", "Training R2", "Testing MSE", "Testing R2"]

    lr_df = linear_regression(x_train, y_train, x_test, y_test)
    rf_df = random_forest(x_train, y_train, x_test, y_test)

    df_models = pd.concat([lr_df, rf_df], axis=0)
    df_models.to_csv("Resources/Results.csv", mode='w', index=False, header=cols)

    results = pd.read_csv("Resources/Results.csv")



    print(results)
