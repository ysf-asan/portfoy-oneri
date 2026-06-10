import yfinance as yf
import pandas as pd
import numpy as np
import os
import io
import time
import contextlib
import warnings
import logging
import kagglehub
from config import HISTORY_PERIOD, HISTORY_INTERVAL, DATA_DIR, KAGGLE_DATASETS

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("peewee").setLevel(logging.WARNING)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 gün

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def cache_path(symbols, period):
    key = "_".join(sorted(symbols)) + f"_{period}"
    return os.path.join(CACHE_DIR, f"{hash(key)}.parquet")

def fetch_data(symbols, period=None, interval=None, use_cache=True):
    if period is None:
        period = HISTORY_PERIOD
    if interval is None:
        interval = HISTORY_INTERVAL
    if not symbols:
        return pd.DataFrame()
    symbols = list(symbols)

    if use_cache:
        ensure_cache_dir()
        cpath = cache_path(symbols, period)
        if os.path.exists(cpath):
            age = time.time() - os.path.getmtime(cpath)
            if age < CACHE_TTL_SECONDS:
                df = pd.read_parquet(cpath)
                if len(df.columns) > 0:
                    return df

    with contextlib.redirect_stderr(io.StringIO()):
        df = yf.download(
            tickers=symbols, period=period, interval=interval,
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(1):
            close_df = df.xs("Close", axis=1, level=1)
        else:
            close_df = df
    else:
        close_df = df

    close_df = close_df.dropna(axis=1, how="all")
    close_df = close_df.ffill().bfill()
    close_df = close_df.dropna(axis=1)

    if use_cache:
        ensure_cache_dir()
        close_df.to_parquet(cache_path(symbols, period))

    return close_df

def _get_bist100_kaggle_path():
    try:
        return kagglehub.dataset_download(KAGGLE_DATASETS["bist100_prices"])
    except Exception:
        return None

def _parse_bist_filename(fname):
    base = os.path.splitext(fname)[0]
    parts = base.split("-")
    ticker = parts[0].upper()
    return f"{ticker}.IS"

def _calc_price_features(prices):
    if len(prices) < 20:
        return None
    rets = prices.pct_change().dropna()
    if len(rets) < 10:
        return None
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = prices.cummax()
    dd = (prices - peak) / peak
    max_dd = dd.min()
    return {
        "ann_return": ann_ret,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }

def load_bist100_features():
    cache_file = os.path.join(DATA_DIR, "_bist_features.parquet")
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < CACHE_TTL_SECONDS:
            return pd.read_parquet(cache_file)

    # Try local data dir first, then kaggle cache
    path = DATA_DIR
    local_files = [f for f in os.listdir(path) if "-" in f and f.endswith(".csv") and f != "snp500_companies_description.csv"]
    if not local_files:
        path = _get_bist100_kaggle_path()
    if path is None or not os.path.isdir(path):
        return None

    rows = []
    for fname in sorted(os.listdir(path)):
        if "-" not in fname or not fname.endswith(".csv") or "snp500" in fname.lower():
            continue
        sym = _parse_bist_filename(fname)
        try:
            df = pd.read_csv(os.path.join(path, fname), parse_dates=["Date"])
            df = df.sort_values("Date")
            if "adjclose" in df.columns:
                col = "adjclose"
            elif "Adj Close" in df.columns:
                col = "Adj Close"
            elif "close" in df.columns:
                col = "close"
            else:
                cols_lower = [c.lower() for c in df.columns]
                close_candidates = ["adjclose", "adj close", "close", "adj_close", "adjusted close"]
                found = None
                for cand in close_candidates:
                    if cand in cols_lower:
                        found = df.columns[cols_lower.index(cand)]
                        break
                if found:
                    col = found
                else:
                    continue
            df = df.set_index("Date")[col].dropna()
            feat = _calc_price_features(df)
            if feat is None:
                continue
            rows.append({
                "symbol": sym,
                "ann_return": feat["ann_return"],
                "ann_volatility": feat["ann_volatility"],
                "sharpe": feat["sharpe"],
                "max_drawdown": feat["max_drawdown"],
            })
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    try:
        df.to_parquet(cache_file)
    except Exception:
        pass
    return df

def fetch_ticker_info(symbols):
    """Her sembol için sektör, şirket adı ve kısa açıklama döndür. Cache'li."""
    cache_file = os.path.join(CACHE_DIR, "ticker_info.parquet")
    existing = {}
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            existing = df.set_index("symbol").to_dict("index")
        except Exception:
            pass

    to_fetch = [s for s in symbols if s not in existing]
    new_rows = []
    for sym in to_fetch:
        try:
            info = yf.Ticker(sym).info
            new_rows.append({
                "symbol": sym,
                "name": info.get("longName") or info.get("shortName", sym),
                "sector": info.get("sector") or info.get("quoteType", "—"),
                "industry": info.get("industry", "—"),
                "summary": (info.get("longBusinessSummary") or "")[:300],
                "country": info.get("country", "—"),
                "market_cap": info.get("marketCap"),
            })
        except Exception:
            new_rows.append({
                "symbol": sym, "name": sym, "sector": "—",
                "industry": "—", "summary": "", "country": "—", "market_cap": None,
            })

    if new_rows:
        for r in new_rows:
            existing[r["symbol"]] = r
        try:
            pd.DataFrame(list(existing.values())).to_parquet(cache_file)
        except Exception:
            pass

    result = {}
    for sym in symbols:
        result[sym] = existing.get(sym, {"symbol": sym, "name": sym, "sector": "—", "industry": "—", "summary": "", "country": "—", "market_cap": None})
    return result


def get_returns(prices):
    return prices.pct_change().dropna()

def get_covariance(prices):
    returns = get_returns(prices)
    return returns.cov() * 252

def get_correlation(prices):
    returns = get_returns(prices)
    return returns.corr()
