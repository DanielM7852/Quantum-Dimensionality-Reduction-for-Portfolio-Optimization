import numpy as np
import pandas as pd


def prepare_targets(
    df: pd.DataFrame, target_days: int = 5
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    raw = df.copy()
    if isinstance(raw.index, pd.DatetimeIndex):
        date_values = raw.index.to_list()
    elif "Date" in raw.columns:
        date_values = raw["Date"].tolist()
    else:
        date_values = list(range(len(raw)))

    df = raw.reset_index(drop=True)
    future_prices = df["Close"].shift(-target_days)
    current_prices = df["Close"]
    future_returns = (future_prices / current_prices) - 1
    valid_mask = (~future_returns.isna()).to_numpy()
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    targets_valid = np.nan_to_num((future_returns > 0)[valid_mask], nan=0).astype(int)
    returns_valid = np.nan_to_num(future_returns[valid_mask], nan=0.0)
    dates = pd.Series([date_values[i] for i in range(len(valid_mask)) if valid_mask[i]])

    return df_valid, targets_valid, returns_valid, dates
