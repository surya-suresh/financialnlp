"""
Data loading: earnings call transcripts + stock price labels.
Primary source: Bose345/sp500_earnings_transcripts (HuggingFace)
Fallback:       lamini/earnings-calls-qa
Last resort:    synthetic data for smoke-testing
"""

import re
import logging
from datetime import timedelta, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stock price utilities
# ---------------------------------------------------------------------------

def _fetch_price_change(ticker: str, call_date: pd.Timestamp,
                        window_days: int = 3) -> Optional[float]:
    """
    Return % price change from close before the call to close `window_days`
    trading days after the call.  Returns None on failure.
    """
    try:
        import yfinance as yf
        start = call_date - timedelta(days=7)
        end   = call_date + timedelta(days=window_days + 7)
        hist  = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            progress=False, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return None

        # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None

        dates_before = closes.index[closes.index < call_date]
        dates_after  = closes.index[closes.index >= call_date]

        if len(dates_before) == 0 or len(dates_after) < window_days:
            return None

        price_before = float(closes[dates_before[-1]])
        price_after  = float(closes[dates_after[min(window_days - 1,
                                                    len(dates_after) - 1)]])
        if price_before == 0:
            return None
        return (price_after - price_before) / price_before
    except Exception as e:
        logger.debug("yfinance error for %s on %s: %s", ticker, call_date, e)
        return None


# ---------------------------------------------------------------------------
# EPS surprise utilities
# ---------------------------------------------------------------------------

def _fetch_eps_surprise(ticker: str, call_date: pd.Timestamp,
                        window_days: int = 10, limit: int = 60) -> dict:
    """
    Return a dict with keys reported_eps, consensus_eps, label_eps_surprise
    (1=beat, 0=miss) matched to the earnings call closest to call_date.

    Uses yfinance.Ticker.earnings_dates whose index IS the earnings call date,
    so a tight window_days=10 is sufficient.  Returns a dict of Nones on failure.
    """
    empty = {"reported_eps": None, "consensus_eps": None, "label_eps_surprise": None}
    try:
        import yfinance as yf
        ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)

        if ed is None or ed.empty:
            return empty

        # Drop future rows (Reported EPS is NaN until the call happens)
        ed = ed.dropna(subset=["Reported EPS", "EPS Estimate"])
        if ed.empty:
            return empty

        # Normalise index to UTC for comparison
        idx = pd.DatetimeIndex(ed.index).tz_convert("UTC")
        call_ts = pd.Timestamp(call_date).tz_localize("UTC") if call_date.tzinfo is None \
                  else pd.Timestamp(call_date).tz_convert("UTC")

        diffs = pd.Series(idx - call_ts).abs()
        min_pos = diffs.argmin()

        if diffs[min_pos] > pd.Timedelta(days=window_days):
            return empty

        row      = ed.iloc[min_pos]
        actual   = float(row["Reported EPS"])
        estimate = float(row["EPS Estimate"])

        return {
            "reported_eps":       actual,
            "consensus_eps":      estimate,
            "label_eps_surprise": int(actual >= estimate),
        }
    except Exception as e:
        logger.debug("yfinance EPS error for %s on %s: %s", ticker, call_date, e)
        return empty


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _resolve_column(columns: list, candidates: list) -> Optional[str]:
    """
    Return the first column whose lowercase name exactly matches or contains
    any candidate string.  First match wins — avoids mapping two source
    columns to the same target (e.g. "content" AND "structured_content"
    both containing "content").
    """
    for cand in candidates:
        for col in columns:
            if col.lower() == cand or col.lower() == cand.replace("_", ""):
                return col
    # Second pass: substring match (less precise, only if exact failed)
    for cand in candidates:
        for col in columns:
            if cand in col.lower():
                return col
    return None


def _load_sp500_transcripts(max_samples: int) -> pd.DataFrame:
    """Load Bose345/sp500_earnings_transcripts from HuggingFace."""
    from datasets import load_dataset
    logger.info("Loading Bose345/sp500_earnings_transcripts …")
    # trust_remote_code no longer supported in newer datasets versions
    ds = load_dataset("Bose345/sp500_earnings_transcripts", split="train")
    df = ds.to_pandas()
    logger.info("  Raw rows: %d, columns: %s", len(df), list(df.columns))

    # BUG FIX: use first-match-wins resolver so "content" and
    # "structured_content" don't both map to "text".
    cols = list(df.columns)
    ticker_col = _resolve_column(cols, ["ticker", "symbol"])
    date_col   = _resolve_column(cols, ["date", "earnings_date", "call_date"])
    # Prefer plain "content" or "transcript" over "structured_content"
    text_col   = _resolve_column(cols, ["transcript", "content", "text", "body"])

    if not ticker_col or not date_col or not text_col:
        raise ValueError(
            f"Cannot resolve required columns. Found: {cols}. "
            f"Resolved: ticker={ticker_col}, date={date_col}, text={text_col}"
        )

    logger.info("  Resolved columns → ticker: %s | date: %s | text: %s",
                ticker_col, date_col, text_col)
    df = df.rename(columns={ticker_col: "ticker",
                             date_col:   "date",
                             text_col:   "text"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 100]

    if max_samples and len(df) > max_samples:
        # Keep most-recent records to use full chronological range
        df = df.sort_values("date").tail(max_samples)

    return df[["ticker", "date", "text"]].reset_index(drop=True)


def _load_lamini_transcripts(max_samples: int) -> pd.DataFrame:
    """
    Fallback: lamini/earnings-calls-qa.
    Columns: question, answer, date, transcript, q, ticker, predictions
    We use `transcript` (full call text) rather than `answer` to avoid
    aggregating partial QA answers into a pseudo-transcript.
    """
    from datasets import load_dataset
    logger.info("Loading lamini/earnings-calls-qa …")
    # trust_remote_code no longer supported in newer datasets versions
    ds = load_dataset("lamini/earnings-calls-qa", split="train")
    df = ds.to_pandas()
    logger.info("  Raw rows: %d, columns: %s", len(df), list(df.columns))

    cols = list(df.columns)

    # BUG FIX: use first-match-wins resolver so "answer" and "transcript"
    # don't both collide on the "text" target.
    # Prefer "transcript" (full call) over "answer" (single QA answer).
    ticker_col = _resolve_column(cols, ["ticker", "symbol"])
    date_col   = _resolve_column(cols, ["date", "earnings_date"])
    text_col   = _resolve_column(cols, ["transcript", "text", "answer", "body"])

    if not date_col:
        raise ValueError(f"Cannot find date column. Available: {cols}")
    if not text_col:
        raise ValueError(f"Cannot find text column. Available: {cols}")
    if not ticker_col:
        logger.warning("No ticker column found; using 'UNKNOWN'")

    logger.info("  Resolved columns → ticker: %s | date: %s | text: %s",
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

    # Deduplicate: if transcript column is used, the same transcript appears
    # for every QA pair — keep one row per (ticker, date).
    df = (df.groupby(["ticker", "date"])["text"]
            .first()
            .reset_index())

    if max_samples and len(df) > max_samples:
        df = df.sort_values("date").tail(max_samples)

    return df[["ticker", "date", "text"]].reset_index(drop=True)


def _make_synthetic_data(n: int = 200) -> pd.DataFrame:
    """
    Smoke-test fallback: generates synthetic transcripts with random sentiment
    so the full pipeline can be validated end-to-end without internet access.
    """
    logger.warning("Using SYNTHETIC data — for smoke testing only!")
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    dates   = pd.date_range("2018-01-01", "2023-12-31", periods=n)

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

        # --- prepared remarks ---
        prep_sent = np.random.choice(["pos", "neg"])
        prep_phrases = positive_phrases if prep_sent == "pos" else negative_phrases
        remarks = " ".join(np.random.choice(prep_phrases, size=3, replace=True))

        # --- Q&A section (independent sentiment) ---
        qa_sent = np.random.choice(["pos", "neg"])
        qa_phrases = positive_phrases if qa_sent == "pos" else negative_phrases
        qa_text = " ".join(np.random.choice(qa_phrases, size=3, replace=True))

        text = (f"Good afternoon everyone. {remarks} "
                f"Operator: Question-and-Answer Session. "
                f"Q: Could you elaborate on guidance? A: {qa_text}")

        # BUG FIX: old code used  label = 1 if prep_sent == "pos" else 0
        # That made the label a PERFECT function of the prepared-remarks text,
        # so FinBERT recovered it with 1.0 AUC — meaningless.
        #
        # Real stock moves are noisy.  We simulate a weak signal:
        #   ~60% chance label matches prepared sentiment
        #   ~50% chance label matches Q&A sentiment (weaker / independent)
        #   ~30% pure noise flips to break determinism
        base_label = 1 if prep_sent == "pos" else 0
        if np.random.random() < 0.35:      # 35% label-flip noise
            base_label = 1 - base_label
        label = base_label

        # Synthetic EPS surprise: weakly correlated with Q&A sentiment,
        # independent of price direction to reflect real-world noise.
        eps_beat = 1 if qa_sent == "pos" else 0
        if np.random.random() < 0.30:   # 30% flip for realism
            eps_beat = 1 - eps_beat
        reported_eps  = round(np.random.uniform(0.5, 3.0), 2)
        consensus_eps = round(reported_eps + (0.05 if eps_beat else -0.05)
                              + np.random.uniform(-0.02, 0.02), 2)

        rows.append({
            "ticker": ticker,
            "date":   date,
            "text":   text,
            "label":              label,
            "label_eps_surprise": eps_beat,
            "reported_eps":       reported_eps,
            "consensus_eps":      consensus_eps,
            "_synthetic": True,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(max_samples: int = 500,
                 price_window: int = 3,
                 use_synthetic_fallback: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        ticker, date, text, label, price_change
    where label=1 means price went up after the earnings call.

    Loading order:
        1. Bose345/sp500_earnings_transcripts
        2. lamini/earnings-calls-qa
        3. Synthetic (smoke-test) data
    """
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
            raise RuntimeError("All data sources failed and synthetic "
                               "fallback is disabled.")
        df = _make_synthetic_data(max_samples or 200)
        synthetic = True

    _FINAL_COLS = [
        "ticker", "date", "text",
        "label_direction", "price_change_pct",
        "label_eps_surprise", "reported_eps", "consensus_eps",
    ]

    if synthetic and "_synthetic" in df.columns:
        # Labels already baked in; no network calls needed.
        df = df.rename(columns={"label": "label_direction"})
        df["price_change_pct"] = np.where(df["label_direction"] == 1, 0.02, -0.02)
        return df[_FINAL_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fetch real stock-price labels (yfinance)
    # ------------------------------------------------------------------
    logger.info("Fetching stock-price labels via yfinance …")
    price_changes = []
    for _, row in df.iterrows():
        pc = _fetch_price_change(row["ticker"], row["date"], price_window)
        price_changes.append(pc)

    df["price_change_pct"] = price_changes
    df = df.dropna(subset=["price_change_pct"])
    df["label_direction"] = (df["price_change_pct"] > 0).astype(int)

    # ------------------------------------------------------------------
    # Fetch EPS surprise labels (yahooquery)
    # ------------------------------------------------------------------
    logger.info("Fetching EPS surprise labels via yahooquery …")
    eps_rows = []
    for i, row in enumerate(df.itertuples(), 1):
        if i % 50 == 0:
            logger.info("  EPS progress: %d / %d", i, len(df))
        eps_rows.append(_fetch_eps_surprise(row.ticker, row.date))

    eps_df = pd.DataFrame(eps_rows, index=df.index)
    df = pd.concat([df, eps_df], axis=1)

    logger.info(
        "Final dataset: %d records | direction labels: %s | eps labels: %s",
        len(df),
        df["label_direction"].value_counts().to_dict(),
        df["label_eps_surprise"].dropna().astype(int).value_counts().to_dict(),
    )
    return df[_FINAL_COLS].reset_index(drop=True)


def chronological_split(df: pd.DataFrame,
                        train_ratio: float = 0.8):
    """Strict chronological train/test split — NO random shuffling."""
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ---------------------------------------------------------------------------
# Non-transcript context block (for prompt augmentation)
# ---------------------------------------------------------------------------

def fetch_context_blocks_batch(rows: list[dict]) -> dict:
    """
    Pre-fetch non-transcript contextual features for a list of JSONL pair rows.

    Returns a dict keyed by (ticker, date_str) → formatted context block string.

    Features (no label leakage — reported_eps is never included):
      - Prior 4 quarters EPS history (strictly before call_date)
      - Analyst EPS consensus for this quarter (from row["consensus_eps"] if present)
      - Company sector
      - Stock performance YTD (Jan 1 of call year → call_date)

    Network calls are minimized:
      - Sector and EPS history: one yfinance call per unique ticker
      - YTD price: one yfinance call per unique (ticker, year)
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available — context blocks will be empty")
        return {}

    # Collect unique tickers and (ticker, year) pairs
    tickers = {r["ticker"] for r in rows}
    ticker_years = {(r["ticker"], r["date"][:4]) for r in rows}

    # Cache: ticker → {"sector": str, "eps_history": list of (date, eps)}
    ticker_cache: dict[str, dict] = {}

    logger.info("Fetching context metadata for %d unique tickers …", len(tickers))
    for ticker in sorted(tickers):
        cache_entry: dict = {"sector": None, "eps_history": []}
        try:
            t = yf.Ticker(ticker)

            # Sector
            try:
                info = t.fast_info
                # fast_info doesn't have sector; fall back to full info
                sector = None
                try:
                    full_info = t.info
                    sector = full_info.get("sector") or full_info.get("industryDisp")
                except Exception:
                    pass
                cache_entry["sector"] = sector
            except Exception:
                pass

            # EPS history (all historical reported EPS dates)
            try:
                ed = t.get_earnings_dates(limit=20)
                if ed is not None and not ed.empty:
                    ed_clean = ed.dropna(subset=["Reported EPS"]).copy()
                    # Normalise to tz-naive UTC dates for comparisons
                    idx = pd.DatetimeIndex(ed_clean.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    ed_clean.index = idx
                    ed_clean = ed_clean.sort_index(ascending=False)  # newest first
                    cache_entry["eps_history"] = list(
                        zip(ed_clean.index.tolist(),
                            ed_clean["Reported EPS"].tolist())
                    )
            except Exception as exc:
                logger.debug("EPS history error for %s: %s", ticker, exc)

        except Exception as exc:
            logger.debug("Ticker fetch error for %s: %s", ticker, exc)

        ticker_cache[ticker] = cache_entry
        logger.debug("Cached %s: sector=%s, eps_quarters=%d",
                     ticker, cache_entry["sector"], len(cache_entry["eps_history"]))

    # Cache: (ticker, year_str) → ytd_pct (float or None)
    ytd_cache: dict[tuple, Optional[float]] = {}

    logger.info("Fetching YTD price data for %d ticker-year pairs …", len(ticker_years))
    for ticker, year_str in sorted(ticker_years):
        try:
            year_start = f"{year_str}-01-01"
            # Fetch through end of year + buffer so we get all trading days
            year_end   = f"{int(year_str) + 1}-01-15"
            hist = yf.download(ticker, start=year_start, end=year_end,
                               progress=False, auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                ytd_cache[(ticker, year_str)] = None
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            closes = hist["Close"].dropna()
            # Store the full price series so we can slice per call_date
            ytd_cache[(ticker, year_str)] = closes
        except Exception as exc:
            logger.debug("YTD fetch error for %s %s: %s", ticker, year_str, exc)
            ytd_cache[(ticker, year_str)] = None

    # Assemble one context block per row
    result: dict[tuple, str] = {}
    for row in rows:
        ticker   = row["ticker"]
        date_str = row["date"][:10]          # "YYYY-MM-DD"
        year_str = date_str[:4]
        call_ts  = pd.Timestamp(date_str)
        consensus_eps = row.get("consensus_eps")

        parts: list[str] = []
        cache = ticker_cache.get(ticker, {})

        # 1. Prior 4 quarters EPS (strictly before call_date)
        history = cache.get("eps_history", [])
        prior = [(dt, eps) for dt, eps in history
                 if pd.Timestamp(dt) < call_ts][:4]
        if prior:
            eps_strs = []
            for dt, eps in prior:
                ts = pd.Timestamp(dt)
                q  = (ts.month - 1) // 3 + 1
                eps_strs.append(f"Q{q} {ts.year}: ${float(eps):.2f}")
            parts.append(f"- Prior 4 quarters EPS: {', '.join(eps_strs)}")

        # 2. Analyst consensus (no leakage: this is the estimate, NOT reported)
        if consensus_eps is not None:
            try:
                parts.append(
                    f"- Analyst EPS consensus this quarter: ${float(consensus_eps):.2f}"
                )
            except (TypeError, ValueError):
                pass

        # 3. Sector
        sector = cache.get("sector")
        if sector:
            parts.append(f"- Sector: {sector}")

        # 4. YTD stock performance (Jan 1 → call_date)
        price_series = ytd_cache.get((ticker, year_str))
        if price_series is not None and hasattr(price_series, "__len__") and len(price_series) >= 2:
            # Slice up to (and including) call_date
            prior_prices = price_series[price_series.index <= call_ts]
            if len(prior_prices) >= 2:
                ytd_ret = (float(prior_prices.iloc[-1]) - float(prior_prices.iloc[0])) \
                          / float(prior_prices.iloc[0]) * 100
                sign = "+" if ytd_ret >= 0 else ""
                parts.append(f"- Stock performance YTD: {sign}{ytd_ret:.1f}%")

        if parts:
            result[(ticker, date_str)] = "Context:\n" + "\n".join(parts)
        else:
            result[(ticker, date_str)] = ""

    logger.info("Context blocks assembled: %d / %d non-empty",
                sum(1 for v in result.values() if v), len(result))
    return result
