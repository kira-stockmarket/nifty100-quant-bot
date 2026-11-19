import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Paths
UNIVERSE_CSV = Path("data/ind_nifty100list.csv")
RAW_DATA_DIR = Path("data/raw")
BATCH_SIZE = 10
BATCH_SLEEP_SEC = 2.0
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_nifty100_symbols(csv_path: Path) -> list[str]:
    """Read NSE symbols from the Nifty 100 universe file."""
    df = pd.read_csv(csv_path)
    symbols = df["Symbol"].dropna().astype(str).str.strip().unique().tolist()
    logger.info("Loaded %d symbols from %s", len(symbols), csv_path)
    return symbols


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize OHLCV columns, index, and dtypes for ML feature engineering."""
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    available_cols = [c for c in required_cols if c in df.columns]
    if len(available_cols) < len(required_cols):
        return pd.DataFrame()

    df = df[required_cols].copy()
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame()

    # Ensure clean, timezone-naive DatetimeIndex sorted ascending
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Enforce consistent numeric types
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype("float64")
    df["Volume"] = df["Volume"].fillna(0).astype("int64")

    return df


def save_parquet(symbol: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Save a single stock's OHLCV DataFrame to data/raw/{SYMBOL}.parquet."""
    file_path = output_dir / f"{symbol}.parquet"
    df.to_parquet(file_path, engine="pyarrow", compression="snappy")


def download_single_with_retry(symbol: str) -> pd.DataFrame:
    """Fallback single-ticker downloader with exponential backoff."""
    ticker = f"{symbol}.NS"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                tickers=ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            cleaned = clean_ohlcv(df)
            if not cleaned.empty:
                return cleaned
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, ticker, exc
            )
        time.sleep(2 * attempt)
    return pd.DataFrame()


def fetch_all_stocks() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_nifty100_symbols(UNIVERSE_CSV)

    succeeded: list[str] = []
    failed: list[str] = []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        yf_tickers = [f"{sym}.NS" for sym in batch]
        logger.info(
            "Downloading batch %d/%d: %s",
            (i // BATCH_SIZE) + 1,
            (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE,
            ", ".join(batch),
        )

        try:
            batch_data = yf.download(
                tickers=yf_tickers,
                period="max",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                multi_level_index=True,
            )
        except Exception as exc:
            logger.error("Batch download error: %s", exc)
            batch_data = pd.DataFrame()

        for sym in batch:
            ticker = f"{sym}.NS"
            stock_df = pd.DataFrame()

            if not batch_data.empty and ticker in batch_data.columns.get_level_values(0):
                stock_df = clean_ohlcv(batch_data[ticker])

            # Retry individually if missing from batch response
            if stock_df.empty:
                logger.info("Retrying %s individually...", ticker)
                stock_df = download_single_with_retry(sym)

            if not stock_df.empty:
                save_parquet(sym, stock_df, RAW_DATA_DIR)
                succeeded.append(sym)
            else:
                logger.error("Failed to fetch data for %s", sym)
                failed.append(sym)

        time.sleep(BATCH_SLEEP_SEC)

    logger.info(
        "Download complete | Succeeded: %d/%d | Failed: %d",
        len(succeeded),
        len(symbols),
        len(failed),
    )
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))

    # Fail workflow only if more than 10% of the universe fails
    if len(failed) > max(1, int(0.10 * len(symbols))):
        raise RuntimeError(f"Too many tickers failed ({len(failed)}/{len(symbols)}).")


if __name__ == "__main__":
    fetch_all_stocks()
