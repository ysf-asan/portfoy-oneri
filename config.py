import os

# ============================================================
# DATA SOURCE
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SP500_FUNDAMENTALS_FILE = os.path.join(DATA_DIR, "snp500_companies_description.csv")

KAGGLE_DATASETS = {
    "sp500_fundamentals": "ilyaryabov/financial-performance-of-companies-from-sp500",
    "bist100_prices": "hakanetin/bist100turkishstaockmarketturkhissefiyatlar",
}

HISTORY_PERIOD = "3y"
HISTORY_INTERVAL = "1d"

# ============================================================
# ML FEATURES (canonical name -> Kaggle column aliases)
# ============================================================
ML_FEATURES = {
    "roe": [
        "Return on Equity (ttm)", "returnOnEquity", "ROE", "return_on_equity",
        "Return on Equity", "roe",
    ],
    "net_margin": [
        "Net Profit Margin (ttm)", "netProfitMargin", "NetMargin",
        "net_margin", "Net Profit Margin",
    ],
    "pe": [
        "Price-to-Earnings (ttm)", "priceEarnings", "PE", "pe_ratio",
        "P/E", "PriceEarnings", "priceEarningsRatio",
    ],
    "debt_equity": [
        "Total Debt to Equity (mrq)", "debtEquity", "DebtEquity",
        "debt_equity_ratio", "Debt/Equity", "debtToEquity",
    ],
    "revenue_growth": [
        "Quarterly revenue growth (YoY)", "revenueGrowth", "RevGrowth",
        "revenue_growth", "Revenue Growth",
    ],
    "beta": [
        "Beta", "beta",
    ],
    "volatility": [
        "Volatility (Week, Month)", "Volatility", "volatility",
    ],
    "dividend_yield": [
        "Dividend yield (annual)", "DividendYield", "dividend_yield",
        "Dividend yield",
    ],
    "perf_1y": [
        "Performance (Year)", "Performance (Year)", "perf_1y",
        "Performance (Year)",
    ],
}

INVERTED_FEATURES = {"pe", "debt_equity", "beta", "volatility"}
MIN_FEATURES_FOR_ML = 3

# ============================================================
# PROFILE-SPECIFIC ML WEIGHTS (sum to 1.0 per profile)
# ============================================================
PROFILE_ML_WEIGHTS = {
    "conservative": {
        "roe": 0.10, "net_margin": 0.10, "pe": 0.15, "debt_equity": 0.15,
        "revenue_growth": 0.05, "beta": 0.10, "volatility": 0.10,
        "dividend_yield": 0.15, "perf_1y": 0.10,
    },
    "moderate": {
        "roe": 0.15, "net_margin": 0.10, "pe": 0.10, "debt_equity": 0.10,
        "revenue_growth": 0.15, "beta": 0.05, "volatility": 0.05,
        "dividend_yield": 0.10, "perf_1y": 0.20,
    },
    "aggressive": {
        "roe": 0.20, "net_margin": 0.10, "pe": 0.05, "debt_equity": 0.05,
        "revenue_growth": 0.20, "beta": 0.00, "volatility": 0.00,
        "dividend_yield": 0.05, "perf_1y": 0.35,
    },
}

# ============================================================
# INCOME -> TARGET PORTFOLIO SIZE (cardinality)
# ============================================================
INCOME_TO_CARDINALITY = {0: 5, 1: 7, 2: 8, 3: 10}
STOCK_POOL_FACTOR = 2
STOCK_SELECTOR_CONFIG = {
    "random_state": 42,
}

# ============================================================
# FALLBACK STOCKS
# ============================================================
FALLBACK_US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
FALLBACK_BIST_STOCKS = [
    "THYAO.IS", "GARAN.IS", "SISE.IS", "EREGL.IS",
    "AKBNK.IS", "TUPRS.IS", "KCHOL.IS", "ASELS.IS",
]

# ============================================================
# STATIC NON-STOCK ASSETS
# ============================================================
NON_STOCK_ASSETS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "GC=F": "Altın",
    "CL=F": "Petrol",
    "AGG": "Tahvil ETF",
    "TLT": "Uzun Vadeli Tahvil",
    "USDTRY=X": "USD/TRY",
    "EURUSD=X": "EUR/USD",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

NON_STOCK_CATEGORY = {
    "SPY": "ABD ETF", "QQQ": "ABD ETF",
    "GC=F": "Emtia", "CL=F": "Emtia",

    "AGG": "Tahvil ETF", "TLT": "Tahvil ETF",
    "USDTRY=X": "Forex", "EURUSD=X": "Forex",
    "BTC-USD": "Kripto", "ETH-USD": "Kripto",
}

ASSET_TYPE = {
    "SPY": "equity", "QQQ": "equity", "BTC-USD": "equity", "ETH-USD": "equity",
    "AGG": "bond", "TLT": "bond",
    "GC=F": "commodity", "CL=F": "commodity",
    "USDTRY=X": "forex", "EURUSD=X": "forex",
}

def classify_asset_type(symbol):
    if symbol in ASSET_TYPE:
        return ASSET_TYPE[symbol]
    if symbol.endswith(".IS"):
        return "equity"
    if "-USD" in symbol or symbol.endswith("-US"):
        return "equity"
    if symbol.endswith("=X"):
        return "forex"
    if symbol.endswith("=F"):
        return "commodity"
    return "equity"

def get_symbol_label(symbol):
    if symbol in NON_STOCK_ASSETS:
        return NON_STOCK_ASSETS[symbol]
    return symbol

def get_symbol_category(symbol):
    if symbol in NON_STOCK_CATEGORY:
        return NON_STOCK_CATEGORY[symbol]
    if symbol.endswith(".IS"):
        return "BIST Hisse"
    return "ABD Hisse"

# ============================================================
# RISK PROFILE - QUESTIONNAIRE (weighted scoring)
# ============================================================
QUESTION_WEIGHTS = {
    "age": 0.5,
    "income": 1.0,
    "horizon": 1.0,
    "loss_tolerance": 2.0,
    "goal": 2.0,
}

RISK_QUESTIONNAIRE = [
    {"key": "age", "question": "Yaş aralığınız nedir?",
     "options": [("18-25", 3), ("26-35", 2), ("36-50", 1), ("51+", 0)]},
    {"key": "income", "question": "Aylık gelir aralığınız nedir?",
     "options": [("0-30.000 TL", 0), ("30.001-75.000 TL", 1), ("75.001-150.000 TL", 2), ("150.000+ TL", 3)]},
    {"key": "horizon", "question": "Yatırım vadeniz nedir?",
     "options": [("1 yıldan az", 0), ("1-3 yıl", 1), ("3-5 yıl", 2), ("5 yıldan fazla", 3)]},
    {"key": "loss_tolerance", "question": "Portföyünüzün yüzde kaç değer kaybına dayanabilirsiniz?",
     "options": [("%5'ten az", 0), ("%5-10", 1), ("%10-20", 2), ("%20+", 3)]},
    {"key": "goal", "question": "Yatırım hedefiniz nedir?",
     "options": [("Sermayemi korumak", 0), ("Enflasyonun üzerinde getiri", 1), ("Dengeli büyüme", 2), ("Maksimum büyüme", 3)]},
]

# Max score = 3*0.5 + 3*1 + 3*1 + 3*2 + 3*2 = 1.5 + 3 + 3 + 6 + 6 = 19.5
SCORE_TO_PROFILE = [(13, "aggressive"), (7, "moderate"), (0, "conservative")]

# ============================================================
# PORTFOLIO CONSTRAINTS BY PROFILE
# ============================================================
PROFILE_CONSTRAINTS = {
    "conservative": {
        "label": "Muhafazakâr",
        "max_equity_crypto": 0.20,
        "min_bond_commodity_forex": 0.80,
        "max_single_asset": 0.15,
        "description": "Düşük risk, sermaye koruma odaklı",
    },
    "moderate": {
        "label": "Dengeli",
        "max_equity_crypto": 0.60,
        "min_bond_commodity_forex": 0.20,
        "max_single_asset": 0.20,
        "description": "Orta risk, dengeli büyüme",
    },
    "aggressive": {
        "label": "Agresif",
        "max_equity_crypto": 1.0,
        "min_bond_commodity_forex": 0.0,
        "max_single_asset": 0.30,
        "description": "Yüksek risk, maksimum büyüme",
    },
}

PROFILE_ASSET_FILTER = {
    "conservative": ["SPY", "AGG", "TLT", "GC=F", "USDTRY=X", "EURUSD=X"],
    "moderate": ["SPY", "QQQ", "GC=F", "AGG", "USDTRY=X", "BTC-USD"],
    "aggressive": ["SPY", "QQQ", "BTC-USD", "ETH-USD"],
}
