import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import (
    SP500_FUNDAMENTALS_FILE, ML_FEATURES, INVERTED_FEATURES,
    PROFILE_ML_WEIGHTS, MIN_FEATURES_FOR_ML, STOCK_SELECTOR_CONFIG,
    STOCK_POOL_FACTOR, INCOME_TO_CARDINALITY, DATA_DIR,
)
from src.data_fetcher import load_bist100_features

def _get_canonical_features():
    return list(ML_FEATURES.keys())

def _auto_detect_columns(df):
    available = {}
    for canonical, aliases in ML_FEATURES.items():
        for alias in aliases:
            if alias in df.columns:
                available[canonical] = alias
                break
    return available

def _clean_value(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("%", "").replace(",", "")
    if s in ("", "-", "—", "N/A", "None", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def _load_and_clean_sp500():
    if not os.path.exists(SP500_FUNDAMENTALS_FILE):
        return None

    df = pd.read_csv(SP500_FUNDAMENTALS_FILE)
    col_map = _auto_detect_columns(df)

    available = [c for c in _get_canonical_features() if c in col_map]
    if len(available) < MIN_FEATURES_FOR_ML:
        return None

    symbol_col = None
    for col in ["Symbol", "symbol", "Ticker", "ticker", "Company", "company"]:
        if col in df.columns:
            symbol_col = col
            break
    if symbol_col is None:
        return None

    keep = [symbol_col] + [col_map[c] for c in available]
    sub = df[keep].copy()
    sub[symbol_col] = sub[symbol_col].astype(str).str.strip()

    for can in available:
        col = col_map[can]
        if "Volatility" in col:
            sub[col] = sub[col].astype(str).str.strip().str.split().str[0]
        sub[col] = sub[col].apply(_clean_value)

    sub = sub.dropna(subset=[col_map[c] for c in available]).copy()

    for can in available:
        col = col_map[can]
        q1 = sub[col].quantile(0.01)
        q99 = sub[col].quantile(0.99)
        sub = sub[(sub[col] >= q1) & (sub[col] <= q99)]

    if len(sub) < 20:
        return None

    sub.rename(columns={symbol_col: "_symbol"}, inplace=True)
    for can, alias in col_map.items():
        if can in available:
            sub.rename(columns={alias: can}, inplace=True)

    return sub, available

def _score_stocks(df, feature_cols, profile_weights):
    x = df[feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.fillna(0).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    for i, col in enumerate(feature_cols):
        if col in INVERTED_FEATURES:
            x_scaled[:, i] = -x_scaled[:, i]

    weights = np.array([profile_weights.get(c, 0) for c in feature_cols])
    weights = weights / weights.sum()

    df = df.copy()
    df["_score"] = x_scaled @ weights
    return df.sort_values("_score", ascending=False)

def _score_bist_pool(df, profile_weights):
    if df is None or len(df) < 5:
        return None

    bist_features = {
        "perf_1y": "ann_return",
        "volatility": "ann_volatility",
    }

    available = [k for k, v in bist_features.items() if v in df.columns]
    if not available:
        return None

    x = df[[bist_features[c] for c in available]].copy()
    for col in list(x.columns):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.fillna(0).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    for i, col in enumerate(available):
        if col in INVERTED_FEATURES:
            x_scaled[:, i] *= -1.0

    w = np.array([profile_weights[c] for c in available])
    w = w / w.sum()

    df = df.copy()
    df["_score"] = x_scaled @ w
    return df.sort_values("_score", ascending=False)

def run_selection(profile_name, target_count):
    profile_weights = PROFILE_ML_WEIGHTS.get(profile_name, PROFILE_ML_WEIGHTS["moderate"])
    pool_size = max(target_count * STOCK_POOL_FACTOR, 15)
    us_take = max(pool_size * 3 // 4, 8)
    bist_take = min(pool_size - us_take, 5)
    us_take = pool_size - bist_take

    us_ranked = []
    bist_ranked = []
    all_ranked = []

    # --- S&P 500 ---
    sp500_result = _load_and_clean_sp500()
    if sp500_result is not None:
        sp_df, sp_features = sp500_result
        sp_df = _score_stocks(sp_df, sp_features, profile_weights)
        max_score = sp_df["_score"].max() or 1
        for _, row in sp_df.iterrows():
            all_ranked.append({
                "_symbol": row["_symbol"],
                "_source": "US",
                "_score": row["_score"] / (max_score + 1e-10),
            })
            us_ranked.append((row["_symbol"], row["_score"]))

    # --- BIST100 ---
    bist_df = load_bist100_features()
    if bist_df is not None:
        bist_df = _score_bist_pool(bist_df, profile_weights)
        if bist_df is not None and len(bist_df) > 0:
            max_score = bist_df["_score"].max() or 1
            for _, row in bist_df.iterrows():
                all_ranked.append({
                    "_symbol": row["symbol"],
                    "_source": "BIST",
                    "_score": row["_score"] / (max_score + 1e-10),
                })
                bist_ranked.append((row["symbol"], row["_score"]))

    if not us_ranked and not bist_ranked:
        return None, None

    us_ranked.sort(key=lambda r: -r[1])
    bist_ranked.sort(key=lambda r: -r[1])

    us_take = min(us_take, len(us_ranked))
    bist_take = min(bist_take, len(bist_ranked))
    actual_us = us_take
    actual_bist = bist_take

    if actual_us < us_take and bist_ranked:
        extra = min(us_take - actual_us, len(bist_ranked) - actual_bist)
        actual_bist += extra
    if actual_bist < bist_take and us_ranked:
        extra = min(bist_take - actual_bist, len(us_ranked) - actual_us)
        actual_us += extra

    selected_symbols = []
    details_rows = []

    for i, (sym, score) in enumerate(us_ranked[:actual_us]):
        selected_symbols.append(sym)
        details_rows.append({"_symbol": sym, "_source": "US", "_score": score, "_rank": i+1})

    offset = len(details_rows)
    for i, (sym, score) in enumerate(bist_ranked[:actual_bist]):
        selected_symbols.append(sym)
        details_rows.append({"_symbol": sym, "_source": "BIST", "_score": score, "_rank": offset+i+1})

    details = pd.DataFrame(details_rows)

    return selected_symbols, details

def get_fallback_symbols():
    from config import FALLBACK_US_STOCKS, FALLBACK_BIST_STOCKS
    return FALLBACK_US_STOCKS + FALLBACK_BIST_STOCKS, len(FALLBACK_US_STOCKS + FALLBACK_BIST_STOCKS)
