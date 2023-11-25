import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def random_forest(x_train, y_train, x_test, y_test):
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)

    # applying the model to make a prediction

    y_rf_train_pred = rf.predict(x_train)
    y_rf_test_pred = rf.predict(x_test)

    # Evaluate model performance

    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)

    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    return pd.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
