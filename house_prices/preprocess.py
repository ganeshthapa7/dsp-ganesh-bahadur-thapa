import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with median and categorical with 'Unknown'.
    """
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(include="object"):
        df[col].fillna("Unknown", inplace=True)
    return df


def encode_categorical_features(
    encoder: OneHotEncoder, df: pd.DataFrame, fit: bool = False
) -> pd.DataFrame:
    """
    Encode categorical columns using OneHotEncoder.

    Args:
        encoder: Fitted or unfitted OneHotEncoder instance.
        df (pd.DataFrame): Data containing categorical columns.
        fit (bool): If True, fit encoder before transforming.

    Returns:
        pd.DataFrame: Encoded categorical features.
    """
    categorical_data = df.select_dtypes(include="object")
    if fit:
        encoder.fit(categorical_data)
    encoded = pd.DataFrame(
        encoder.transform(categorical_data),
        columns=encoder.get_feature_names_out(categorical_data.columns),
        index=df.index,
    )
    return encoded


def scale_numeric_features(
    scaler: StandardScaler, df: pd.DataFrame, fit: bool = False
) -> pd.DataFrame:
    """
    Scale numeric columns using StandardScaler.

    Args:
        scaler: Fitted or unfitted StandardScaler instance.
        df (pd.DataFrame): Data containing numeric columns.
        fit (bool): If True, fit scaler before transforming.

    Returns:
        pd.DataFrame: Scaled numeric features.
    """
    numeric_data = df.select_dtypes(exclude="object")
    if fit:
        scaler.fit(numeric_data)
    scaled = pd.DataFrame(
        scaler.transform(numeric_data),
        columns=numeric_data.columns,
        index=df.index,
    )
    return scaled
