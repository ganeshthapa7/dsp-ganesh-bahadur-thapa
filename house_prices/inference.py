import os
import joblib
import pandas as pd
import numpy as np
from house_prices.preprocess import (
    handle_missing_values,
    encode_categorical_features,
    scale_numeric_features,
)


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make house price predictions using a trained model and
    saved preprocessing objects.

    Args:
        input_data (pd.DataFrame): The unseen dataset (e.g., test.csv)
            containing all the feature columns required by the model.

    Returns:
        np.ndarray: Array of predicted house prices.
    """

    base_path = os.path.join("..", "models")
    model = joblib.load(os.path.join(base_path, "model.joblib"))
    encoder = joblib.load(os.path.join(base_path, "encoder.joblib"))
    scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))

  
    cleaned_data = handle_missing_values(input_data)

   
    encoded_data = encode_categorical_features(
        encoder,
        cleaned_data,
        fit=False,
    )
    scaled_data = scale_numeric_features(
        scaler,
        cleaned_data,
        fit=False,
    )

    
    processed_data = pd.concat(
        [scaled_data, encoded_data],
        axis=1,
    )


    predictions = model.predict(processed_data)
    print("Predictions completed successfully.")
    return predictions
