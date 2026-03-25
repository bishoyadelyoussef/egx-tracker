"""
EGX Financial Analyzer

Production-ready script with:
- Reliable data fetching (retries + timeout)
- Rich console table output
- Integration-ready JSON summary
- Safe JSON exports to the script's local directory
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# .env placeholders (optional for future integration):
# TELEGRAM_BOT_TOKEN=...
# TELEGRAM_CHAT_ID=...
# DATABASE_URL=...
# ---------------------------------------------------------------------------


logger = logging.getLogger("egx_financial_analyzer")


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the logging module for consistent output.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_last_scalar(value: Any) -> float:
    """
    Return the latest scalar numeric value from a Series/DataFrame/scalar-like input.
    """
    if isinstance(value, pd.DataFrame):
        return float(value.iloc[-1, -1])
    if isinstance(value, pd.Series):
        return float(value.iloc[-1])
    return float(value)


def safe_round(value: Any, digits: int = 2) -> Any:
    """
    Round numeric values safely; pass through non-numeric values.
    """
    try:
        if pd.isna(value):
            # JSON-safe "missing value" instead of emitting NaN.
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a closing price series.
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def sma_rolling(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute a rolling simple moving average (SMA).
    """
    return close.rolling(window).mean()


def strategy_from_rsi(rsi_value: Any) -> str:
    """
    Map RSI thresholds to an actionable strategy.

    Rules:
    - RSI < 35  => Buy Zone (Start Accumulating)
    - RSI > 65  => Profit Zone (Consider Selling)
    - Otherwise => Observation (Wait for setup)
    """
    if rsi_value is None or pd.isna(rsi_value):
        return "Observation (Wait for setup)"

    rsi = float(rsi_value)
    if rsi < 35:
        return "Buy Zone (Start Accumulating)"
    if rsi > 65:
        return "Profit Zone (Consider Selling)"
    return "Observation (Wait for setup)"


def detect_sma_cross_alert(close: pd.Series, sma_20_series: pd.Series) -> str:
    """
    Detect whether price crossed the 20-day SMA in the most recent session.

    Alert Trigger:
    - Price crossed above SMA: prev_price <= prev_sma AND last_price > last_sma
    - Price crossed below SMA: prev_price >= prev_sma AND last_price < last_sma
    """
    prev_price = close.iloc[-2]
    last_price = close.iloc[-1]
    prev_sma = sma_20_series.iloc[-2]
    last_sma = sma_20_series.iloc[-1]

    # If SMA isn't fully formed yet, default to stable.
    if any(pd.isna(x) for x in (prev_price, last_price, prev_sma, last_sma)):
        return "Stable"

    crossed_up = prev_price <= prev_sma and last_price > last_sma
    crossed_down = prev_price >= prev_sma and last_price < last_sma
    return "Price Cross Alert" if (crossed_up or crossed_down) else "Stable"


def fetch_history_with_retry(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    retries: int = 3,
    retry_delay_sec: float = 2.0,
    timeout_sec: int = 20,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with retries to mitigate transient timeouts.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            hist = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=timeout_sec,
            )
            if hist.empty:
                raise ValueError(f"No data returned for {ticker}.")
            return hist
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                # Simple backoff to help with transient network issues.
                time.sleep(retry_delay_sec * attempt)
            else:
                raise RuntimeError(f"Failed to fetch data for {ticker}: {last_error}") from last_error

    raise RuntimeError(f"Failed to fetch data for {ticker}: {last_error}")


def fetch_and_analyze_ticker(ticker: str) -> Dict[str, Any]:
    """
    Fetch data for one ticker and compute RSI, SMA(20), Strategy, and Alert.
    """
    try:
        hist = fetch_history_with_retry(ticker=ticker)
        if len(hist) < 50:
            raise ValueError(f"Not enough data for {ticker}.")
        if "Close" not in hist.columns:
            raise ValueError(f"Missing 'Close' column for {ticker}.")

        close_raw = hist["Close"]
        # yfinance returns MultiIndex columns, so hist["Close"] may be a DataFrame even for one ticker.
        if isinstance(close_raw, pd.DataFrame):
            if close_raw.shape[1] == 1:
                close = close_raw.iloc[:, 0]
            elif ticker in close_raw.columns:
                close = close_raw[ticker]
            else:
                close = close_raw.iloc[:, 0]
        else:
            close = close_raw

        rsi_series = calculate_rsi(close)
        sma_20_series = sma_rolling(close, window=20)

        last_price = safe_round(get_last_scalar(close), 2)
        last_rsi = safe_round(get_last_scalar(rsi_series), 2)
        last_sma_20 = safe_round(get_last_scalar(sma_20_series), 2)

        strategy = strategy_from_rsi(last_rsi)
        alert = detect_sma_cross_alert(close, sma_20_series)
        if alert == "Price Cross Alert":
            logger.info("Price Cross Alert detected for %s", ticker)

        return {
            "Ticker": ticker,
            "Price": last_price,
            "RSI": last_rsi,
            "SMA_20": last_sma_20,
            "Strategy": strategy,
            "Alert": alert,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ticker analysis failed for %s", ticker)
        return {
            "Ticker": ticker,
            "Price": "N/A",
            "RSI": "N/A",
            "SMA_20": "N/A",
            "Strategy": "Observation (Wait for setup)",
            "Alert": "Stable",
            "Error": str(exc),
        }


def display_results_table(results: List[Dict[str, Any]]) -> None:
    """
    Display analysis results in a Rich-formatted table.
    """
    table = Table(title="EGX Financial Analyzer", show_lines=True)
    headers = ["Ticker", "Price", "RSI", "SMA (20)", "Strategy", "Alert"]
    for col in headers:
        table.add_column(col, justify="center", no_wrap=True)

    for item in results:
        table.add_row(
            str(item.get("Ticker")),
            str(item.get("Price")),
            str(item.get("RSI")),
            str(item.get("SMA_20")),
            str(item.get("Strategy")),
            str(item.get("Alert")),
        )

    Console().print(table)


def build_json_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build integration-ready JSON payload for APIs, Flowwise, or bots.

    Numeric values are rounded to 2 decimals for readability.
    """
    stocks: List[Dict[str, Any]] = []
    for item in results:
        stocks.append(
            {
                "Ticker": item.get("Ticker"),
                "Price": safe_round(item.get("Price"), 2),
                "RSI": safe_round(item.get("RSI"), 2),
                "SMA": safe_round(item.get("SMA_20"), 2),
                # Backward-compatible alias for earlier integration schemas.
                "Action Strategy": item.get("Strategy"),
                "Strategy": item.get("Strategy"),
                "Alert": item.get("Alert"),
            }
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stocks": stocks,
    }


def export_results(data: Dict[str, Any]) -> List[Path]:
    """
    Safely save the analysis JSON to the script's local directory.

    Creates:
    - `daily_report.json` (latest overwrite)
    - `report_YYYY-MM-DD_HHMMSS.json` (timestamped snapshot)
    """
    script_dir = Path(__file__).resolve().parent
    daily_path = script_dir / "daily_report.json"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    snapshot_path = script_dir / f"report_{ts}.json"

    paths_saved: List[Path] = []
    for path in (daily_path, snapshot_path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            paths_saved.append(path)
        except Exception:
            logger.exception("Failed to export JSON to %s", path)

    return paths_saved


def run_analysis(tickers: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run analysis over tickers and return both table rows and JSON payload.
    """
    results = [fetch_and_analyze_ticker(ticker) for ticker in tickers]
    payload = build_json_summary(results)
    return results, payload


def main() -> None:
    """
    Script entry point.
    """
    setup_logging()

    tickers = ["TMGH.CA", "COMI.CA", "EFIH.CA"]
    results, payload = run_analysis(tickers)

    # Visual: keep Rich table output for immediate human viewing.
    display_results_table(results)

    # Data: save the integration-ready JSON payload.
    saved_paths = export_results(payload)
    if saved_paths:
        logger.info("Exported JSON reports: %s", ", ".join(str(p) for p in saved_paths))
    else:
        logger.warning("No JSON exports were saved (check permissions).")

    logger.info("JSON payload ready for external integrations.")
    logger.debug("Payload: %s", json.dumps(payload, indent=2, ensure_ascii=False))

    # Integration readiness: emit pure JSON to stdout as the final output.
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
