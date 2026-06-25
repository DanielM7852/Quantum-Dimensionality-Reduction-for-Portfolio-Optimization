import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance MultiIndex columns for single-ticker downloads."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' in range {start} to {end}")

    data = _flatten_columns(data)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Downloaded data missing columns: {sorted(missing)}")

    return data
