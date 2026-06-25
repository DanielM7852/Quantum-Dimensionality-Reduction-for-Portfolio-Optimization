import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns_5"] = df["Close"].pct_change(5).fillna(0)
    df["returns_10"] = df["Close"].pct_change(10).fillna(0)
    df["volatility"] = df["Close"].pct_change().rolling(20).std().fillna(0)
    for period in [10, 20, 50]:
        ma = df["Close"].rolling(period).mean()
        df[f"ma_ratio_{period}"] = (df["Close"] / (ma + 1e-10)).fillna(1.0)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    vol_ma = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = (df["Volume"] / (vol_ma + 1)).fillna(1.0)
    df = df.replace([np.inf, -np.inf], 0)
    return df.fillna(0)


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    exclude = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    features = [c for c in df.columns if c not in exclude]
    x = df[features].values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, features
