import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fetch_price_change(ticker: str, call_date: pd.Timestamp,
                        window_days: int = 3) -> Optional[float]:
    try:
        import yfinance as yf
        start = call_date - timedelta(days=7)
        end = call_date + timedelta(days=window_days + 7)
        hist = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"),
                           progress=False, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None

        dates_before = closes.index[closes.index < call_date]
        dates_after = closes.index[closes.index >= call_date]

        if len(dates_before) == 0 or len(dates_after) < window_days:
            return None

        price_before = float(closes[dates_before[-1]])
        price_after = float(closes[dates_after[min(window_days - 1, len(dates_after) - 1)]])
        if price_before == 0:
            return None
        return (price_after - price_before) / price_before
    except Exception as e:
        logger.debug("yfinance error for %s on %s: %s", ticker, call_date, e)
        return None


def _fetch_eps_surprise(ticker: str, call_date: pd.Timestamp,
                        window_days: int = 10, limit: int = 60,
                        max_retries: int = 3,
                        retry_delay: float = 30.0) -> dict:
    import time
    import yfinance as yf
    empty = {"reported_eps": None, "consensus_eps": None, "label_eps_surprise": None}

    for attempt in range(max_retries):
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        except Exception as e:
            logger.debug("yfinance EPS error for %s on %s (attempt %d): %s",
                         ticker, call_date, attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return empty

        if ed is None or ed.empty:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return empty

        ed_full = ed.dropna(subset=["Reported EPS", "EPS Estimate"])
        if ed_full.empty:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return empty

        idx = pd.DatetimeIndex(ed_full.index).tz_convert("UTC")
        call_ts = pd.Timestamp(call_date).tz_localize("UTC") if call_date.tzinfo is None \
                  else pd.Timestamp(call_date).tz_convert("UTC")
        diffs = pd.Series(idx - call_ts).abs()
        min_pos = diffs.argmin()
        if diffs[min_pos] > pd.Timedelta(days=window_days):
            return empty
        row = ed_full.iloc[min_pos]
        actual = float(row["Reported EPS"])
        estimate = float(row["EPS Estimate"])
        return {
            "reported_eps": actual,
            "consensus_eps": estimate,
            "label_eps_surprise": int(actual >= estimate),
        }
    return empty


def _resolve_column(columns: list, candidates: list) -> Optional[str]:
    for cand in candidates:
        for col in columns:
            if col.lower() == cand or col.lower() == cand.replace("_", ""):
                return col
    for cand in candidates:
        for col in columns:
            if cand in col.lower():
                return col
    return None


def _load_sp500_transcripts(max_samples: int) -> pd.DataFrame:
    from datasets import load_dataset
    logger.info("Loading Bose345/sp500_earnings_transcripts ...")
    ds = load_dataset("Bose345/sp500_earnings_transcripts", split="train")
    df = ds.to_pandas()
    logger.info("  Raw rows: %d, columns: %s", len(df), list(df.columns))

    cols = list(df.columns)
    ticker_col = _resolve_column(cols, ["ticker", "symbol"])
    date_col = _resolve_column(cols, ["date", "earnings_date", "call_date"])
    text_col = _resolve_column(cols, ["transcript", "content", "text", "body"])

    if not ticker_col or not date_col or not text_col:
        raise ValueError(
            f"Cannot resolve required columns. Found: {cols}. "
            f"Resolved: ticker={ticker_col}, date={date_col}, text={text_col}"
        )

    logger.info("  Resolved columns -> ticker: %s | date: %s | text: %s",
                ticker_col, date_col, text_col)
    df = df.rename(columns={ticker_col: "ticker", date_col: "date", text_col: "text"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 100]

    if max_samples and len(df) > max_samples:
        df = df.sort_values("date").tail(max_samples)

    return df[["ticker", "date", "text"]].reset_index(drop=True)


def _load_lamini_transcripts(max_samples: int) -> pd.DataFrame:
    from datasets import load_dataset
    logger.info("Loading lamini/earnings-calls-qa ...")
    ds = load_dataset("lamini/earnings-calls-qa", split="train")
    df = ds.to_pandas()
    logger.info("  Raw rows: %d, columns: %s", len(df), list(df.columns))

    cols = list(df.columns)
    ticker_col = _resolve_column(cols, ["ticker", "symbol"])
    date_col = _resolve_column(cols, ["date", "earnings_date"])
    text_col = _resolve_column(cols, ["transcript", "text", "answer", "body"])

    if not date_col:
        raise ValueError(f"Cannot find date column. Available: {cols}")
    if not text_col:
        raise ValueError(f"Cannot find text column. Available: {cols}")
    if not ticker_col:
        logger.warning("No ticker column found; using 'UNKNOWN'")

    logger.info("  Resolved columns -> ticker: %s | date: %s | text: %s",
                ticker_col, date_col, text_col)

    rename_map = {date_col: "date_raw", text_col: "text"}
    if ticker_col:
        rename_map[ticker_col] = "ticker"
    df = df.rename(columns=rename_map)
    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"

    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["date", "ticker", "text"])
    df = df.groupby(["ticker", "date"])["text"].first().reset_index()

    if max_samples and len(df) > max_samples:
        df = df.sort_values("date").tail(max_samples)

    return df[["ticker", "date", "text"]].reset_index(drop=True)


def _make_synthetic_data(n: int = 200) -> pd.DataFrame:
    logger.warning("Using SYNTHETIC data - for smoke testing only!")
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    dates = pd.date_range("2018-01-01", "2023-12-31", periods=n)

    positive_phrases = [
        "strong revenue growth exceeded expectations",
        "record-breaking quarterly earnings",
        "robust demand across all segments",
        "confident in our full-year guidance",
        "exceptional performance driven by innovation",
    ]
    negative_phrases = [
        "headwinds from macro-economic uncertainty",
        "disappointing margin contraction this quarter",
        "challenging competitive environment remains",
        "supply chain disruptions impacted results",
        "cautious outlook for the coming quarters",
    ]

    rows = []
    for i, date in enumerate(dates):
        ticker = tickers[i % len(tickers)]

        prep_sent = np.random.choice(["pos", "neg"])
        prep_phrases = positive_phrases if prep_sent == "pos" else negative_phrases
        remarks = " ".join(np.random.choice(prep_phrases, size=3, replace=True))

        qa_sent = np.random.choice(["pos", "neg"])
        qa_phrases = positive_phrases if qa_sent == "pos" else negative_phrases
        qa_text = " ".join(np.random.choice(qa_phrases, size=3, replace=True))

        text = (f"Good afternoon everyone. {remarks} "
                f"Operator: Question-and-Answer Session. "
                f"Q: Could you elaborate on guidance? A: {qa_text}")

        base_label = 1 if prep_sent == "pos" else 0
        if np.random.random() < 0.35:
            base_label = 1 - base_label
        label = base_label

        eps_beat = 1 if qa_sent == "pos" else 0
        if np.random.random() < 0.30:
            eps_beat = 1 - eps_beat
        reported_eps = round(np.random.uniform(0.5, 3.0), 2)
        consensus_eps = round(reported_eps + (0.05 if eps_beat else -0.05)
                              + np.random.uniform(-0.02, 0.02), 2)

        rows.append({
            "ticker": ticker,
            "date": date,
            "text": text,
            "label": label,
            "label_eps_surprise": eps_beat,
            "reported_eps": reported_eps,
            "consensus_eps": consensus_eps,
            "_synthetic": True,
        })

    return pd.DataFrame(rows)


_FINAL_COLS = [
    "ticker", "date", "text",
    "label_direction", "price_change_pct",
    "label_eps_surprise", "reported_eps", "consensus_eps",
]


def load_dataset(max_samples: int = 500,
                 price_window: int = 3,
                 use_synthetic_fallback: bool = True,
                 task: str = "both",
                 min_date: str = None,
                 newest_first: bool = False,
                 delay_sec: float = 0.0) -> pd.DataFrame:
    import time
    df = None
    synthetic = False

    for loader, name in [(_load_sp500_transcripts, "sp500_transcripts"),
                         (_load_lamini_transcripts, "lamini_qa")]:
        try:
            df = loader(max_samples)
            logger.info("Loaded %d records from %s", len(df), name)
            break
        except Exception as e:
            logger.warning("Could not load %s: %s", name, e)

    if df is None:
        if not use_synthetic_fallback:
            raise RuntimeError("All data sources failed and synthetic fallback is disabled.")
        df = _make_synthetic_data(max_samples or 200)
        synthetic = True

    if synthetic and "_synthetic" in df.columns:
        df = df.rename(columns={"label": "label_direction"})
        df["price_change_pct"] = np.where(df["label_direction"] == 1, 0.02, -0.02)
        return df[_FINAL_COLS].reset_index(drop=True)

    if min_date:
        before = len(df)
        df = df[df["date"] >= pd.Timestamp(min_date)].reset_index(drop=True)
        logger.info("Filtered by min_date=%s: %d -> %d transcripts",
                    min_date, before, len(df))

    if newest_first:
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    if task in ("direction", "both"):
        logger.info("Fetching stock-price labels via yfinance ...")
        price_changes = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            pc = _fetch_price_change(row["ticker"], row["date"], price_window)
            price_changes.append(pc)
            if delay_sec > 0:
                time.sleep(delay_sec)
            if i % 200 == 0:
                ok = sum(1 for x in price_changes if x is not None)
                logger.info("  Price progress: %d / %d  (%d ok)", i, len(df), ok)
        df["price_change_pct"] = price_changes
        df = df.dropna(subset=["price_change_pct"])
        df["label_direction"] = (df["price_change_pct"] > 0).astype(int)
    else:
        df["price_change_pct"] = None
        df["label_direction"] = None

    if task in ("surprise", "both"):
        logger.info("Fetching EPS surprise labels via yfinance ...")
        eps_rows = []
        for i, row in enumerate(df.itertuples(), 1):
            eps_rows.append(_fetch_eps_surprise(row.ticker, row.date))
            if delay_sec > 0:
                time.sleep(delay_sec)
            if i % 50 == 0:
                ok = sum(1 for r in eps_rows if r["label_eps_surprise"] is not None)
                logger.info("  EPS progress: %d / %d  (%d ok = %.1f%%)",
                            i, len(df), ok, 100 * ok / i)
        eps_df = pd.DataFrame(eps_rows, index=df.index)
        df = pd.concat([df, eps_df], axis=1)
    else:
        for col in ("label_eps_surprise", "reported_eps", "consensus_eps"):
            df[col] = None

    direction_counts = (df["label_direction"].dropna().astype(int).value_counts().to_dict()
                        if df["label_direction"].notna().any() else {})
    eps_counts = (df["label_eps_surprise"].dropna().astype(int).value_counts().to_dict()
                  if df["label_eps_surprise"].notna().any() else {})
    logger.info(
        "Final dataset: %d records | direction labels: %s | eps labels: %s",
        len(df), direction_counts, eps_counts,
    )
    return df[_FINAL_COLS].reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8):
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


