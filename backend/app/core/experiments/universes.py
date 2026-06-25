"""Ticker universes for batch experiments."""

SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "KO", "PEP", "COST", "AVGO", "MCD", "TMO", "CSCO", "ACN", "ABT",
]

TECH_10 = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD", "AVGO", "CRM"]

UNIVERSES: dict[str, list[str]] = {
    "sp500_sample": SP500_SAMPLE,
    "tech_10": TECH_10,
    "single": ["NVDA"],
}
