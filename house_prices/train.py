import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def build_model(data: pd.DataFrame) -> dict[str, float]:
    """
    Builds, trains, evaluates, and saves a Random Forest model
    for house price prediction.

    Args:
        data (pd.DataFrame): Training dataset containing features
            and the target column 'SalePrice'.

    Returns:
        dict[str, float]: Dictionary containing model performance metric.
    """

    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_train.median(numeric_only=True))
    for col in X_train.select_dtypes(include="object"):
        X_train[col].fillna("Unknown", inplace=True)
        X_test[col].fillna("Unknown", inplace=True)


    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = X_train.select_dtypes(include="object")
    test_cat = X_test.select_dtypes(include="object")

    encoder.fit(train_cat)
    encoded_train = pd.DataFrame(
        encoder.transform(train_cat),
        columns=encoder.get_feature_names_out(),
        index=X_train.index,
    )
    encoded_test = pd.DataFrame(
        encoder.transform(test_cat),
        columns=encoder.get_feature_names_out(),
        index=X_test.index,
    )

 
    scaler = StandardScaler()
    train_num = X_train.select_dtypes(exclude="object")
    test_num = X_test.select_dtypes(exclude="object")

    scaler.fit(train_num)
    scaled_train = pd.DataFrame(
        scaler.transform(train_num),
        columns=train_num.columns,
        index=train_num.index,
    )
    scaled_test = pd.DataFrame(
        scaler.transform(test_num),
        columns=test_num.columns,
        index=test_num.index,
    )

   
    X_train_final = pd.concat([scaled_train, encoded_train], axis=1)
    X_test_final = pd.concat([scaled_test, encoded_test], axis=1)


    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_final, y_train)


    preds = model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"📊 RMSE: {rmse:.2f}")

   
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/model.joblib")
    joblib.dump(encoder, "../models/encoder.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")

    print("Model, encoder, and scaler saved in 'models/' folder.")
    return {"rmse": float(rmse)}
