import numpy as np
import pandas as pd

# Türkiye politika faizi baz alınarak belirlendi (TCMB, 2025 sonu itibarıyla ~%45)
RISK_FREE_RATE_ANNUAL = 0.45

def sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE_ANNUAL):
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / returns.std() if returns.std() > 0 else 0

def sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE_ANNUAL):
    excess = returns - risk_free_rate / 252
    downside = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else returns.std()
    return np.sqrt(252) * excess.mean() / downside_std if downside_std > 0 else 0

def value_at_risk(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

def conditional_var(returns, confidence=0.95):
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def max_drawdown(prices):
    cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_all_metrics(prices, weights):
    available = [s for s in weights if s in prices.columns]
    if not available:
        return {
            "sharpe": 0, "sortino": 0, "annual_vol": 0,
            "var_95": 0, "cvar_95": 0, "max_drawdown": 0, "annual_return": 0,
        }
    w_arr = np.array([weights[s] for s in available])
    filtered = prices[available]
    portfolio_returns = filtered.pct_change().dropna() @ w_arr
    port_prices = filtered @ w_arr

    return {
        "sharpe": round(sharpe_ratio(portfolio_returns), 3),
        "sortino": round(sortino_ratio(portfolio_returns), 3),
        "annual_vol": round(portfolio_returns.std() * np.sqrt(252), 4),
        "var_95": round(value_at_risk(portfolio_returns, 0.95), 4),
        "cvar_95": round(conditional_var(portfolio_returns, 0.95), 4),
        "max_drawdown": round(max_drawdown(port_prices), 4),
        "annual_return": round(portfolio_returns.mean() * 252, 4),
    }
