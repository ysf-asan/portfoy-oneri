import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
from config import classify_asset_type

def _build_masks(symbols):
    eq_mask = np.array([classify_asset_type(s) == "equity" for s in symbols])
    bond_mask = np.array([classify_asset_type(s) in {"bond", "commodity", "forex"} for s in symbols])
    return eq_mask, bond_mask


def _feasible_constraints(symbols, constraints):
    """Kısıtları eldeki varlık sayısına göre uygulanabilir hale getir.

    Örn. tahvil oranı >= %80 isteniyor ama yalnızca 5 tahvil varlığı varsa ve
    her biri en çok %15 olabiliyorsa, ulaşılabilir tavan %75'tir. Bu durumda
    min_bond hedefini ulaşılabilir seviyeye çeker (solver'ın çökmesini önler).
    """
    import copy
    c = copy.deepcopy(constraints)
    n = len(symbols)
    eq_mask, bond_mask = _build_masks(symbols)
    n_bond = int(bond_mask.sum())
    n_eq = int(eq_mask.sum())

    max_single = c["max_single_asset"]
    # Ağırlıklar 1'e toplanmalı: max_single * n >= 1 olmalı
    if max_single * n < 1.0:
        max_single = min(1.0, 1.0 / n + 0.02)
        c["max_single_asset"] = max_single

    # Tahvil/emtia/forex tabanı ulaşılabilir mi?
    max_bond_achievable = n_bond * max_single
    if c["min_bond_commodity_forex"] > max_bond_achievable:
        c["min_bond_commodity_forex"] = round(max(0.0, max_bond_achievable * 0.95), 4)
        c["_relaxed_bond"] = True

    # Equity tavanı: en az n_eq*0 ... yeterli equity yoksa tavanı düşürmeye gerek yok
    # ama equity tavanı, kalan varlıklarla 1'e toplanmayı engellememeli
    min_eq_needed = 1.0 - n_bond * max_single
    if c["max_equity_crypto"] < min_eq_needed:
        c["max_equity_crypto"] = round(min(1.0, min_eq_needed + 0.05), 4)
        c["_relaxed_equity"] = True

    return c

def _setup_ef(prices, constraints):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    symbols = list(prices.columns)
    constraints = _feasible_constraints(symbols, constraints)
    eq_mask, bond_mask = _build_masks(symbols)
    max_single = constraints["max_single_asset"]

    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= max_single)
    if eq_mask.any():
        ef.add_constraint(lambda w: w @ eq_mask <= constraints["max_equity_crypto"])
    if bond_mask.any():
        ef.add_constraint(lambda w: w @ bond_mask >= constraints["min_bond_commodity_forex"])

    return ef, symbols

def markowitz_optimize(prices, constraints):
    ef, symbols = _setup_ef(prices, constraints)
    try:
        weights = ef.max_sharpe()
    except Exception:
        # max_sharpe çözülmüş örneğe kısıt ekleyemez; fallback için yeni örnek kur
        try:
            ef, symbols = _setup_ef(prices, constraints)
            weights = ef.min_volatility()
        except Exception:
            # Solver tamamen başarısız → Markowitz'i atla, Monte Carlo devralsın
            return None
    perf = ef.portfolio_performance(verbose=False)
    return {
        "weights": dict(zip(symbols, list(weights.values()))),
        "expected_return": perf[0],
        "volatility": perf[1],
        "sharpe": perf[2],
    }

def monte_carlo_optimize(prices, constraints, n_simulations=15000):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    symbols = list(prices.columns)
    constraints = _feasible_constraints(symbols, constraints)
    n = len(symbols)
    eq_mask, bond_mask = _build_masks(symbols)
    max_single = constraints["max_single_asset"]
    max_eq = constraints["max_equity_crypto"]
    min_bond = constraints["min_bond_commodity_forex"]
    n_eq = int(eq_mask.sum())
    n_total = n
    bond_share = (1 - max_eq + min_bond) / 2 if eq_mask.any() else 0.5

    # Kişiye özgü seed → aynı profil farklı kişiler için farklı portföy üretir
    seed = int(constraints.get("personal_seed_offset", 0))
    rng = np.random.default_rng(seed)

    results = np.zeros((n_simulations, 3))
    weight_records = []

    for i in range(n_simulations):
        alpha = np.ones(n)
        if bond_mask.any():
            alpha[bond_mask] *= max(1, bond_share * n_total / (n_total - n_eq + 1e-10) * 3)
        w = rng.dirichlet(alpha)
        if eq_mask.any() and w @ eq_mask > max_eq:
            continue
        if bond_mask.any() and w @ bond_mask < min_bond:
            continue
        if np.any(w > max_single):
            continue
        port_ret = np.sum(mean_returns * w) * 252
        port_vol = np.sqrt(w.T @ (cov_matrix * 252) @ w)
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        results[len(weight_records)] = [port_ret, port_vol, sharpe]
        weight_records.append(w)

    if not weight_records:
        # Kısıtları karşılayan portföy bulunamadı → None dön,
        # böylece Markowitz'in (zaten geçerli) sonucu kullanılır.
        return None

    best_idx = np.argmax(results[:len(weight_records), 2])
    best_w = weight_records[best_idx]
    return {
        "weights": dict(zip(symbols, best_w)),
        "expected_return": results[best_idx, 0],
        "volatility": results[best_idx, 1],
        "sharpe": results[best_idx, 2],
        "all_results": results[:len(weight_records)],
    }

def apply_cardinality(weights, card):
    sorted_items = sorted(weights.items(), key=lambda x: -x[1])
    top = {k: v for k, v in sorted_items[:card]}
    total = sum(top.values())
    if total > 0:
        top = {k: v / total for k, v in top.items()}
    return top

def get_efficient_frontier(prices, constraints, n_portfolios=40):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    symbols = list(prices.columns)
    constraints = _feasible_constraints(symbols, constraints)
    n = len(symbols)
    eq_mask, bond_mask = _build_masks(symbols)
    max_single = constraints["max_single_asset"]

    target_returns = np.linspace(mean_returns.min() * 1.1, mean_returns.max() * 0.9, n_portfolios)
    frontier = []

    def portfolio_vol(w):
        return np.sqrt(w.T @ (cov_matrix * 252) @ w)

    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: np.sum(mean_returns * w) * 252 - target},
        ]
        if eq_mask.any():
            cons.append({"type": "ineq", "fun": lambda w: constraints["max_equity_crypto"] - w @ eq_mask})
        if bond_mask.any():
            cons.append({"type": "ineq", "fun": lambda w: w @ bond_mask - constraints["min_bond_commodity_forex"]})
        bounds = tuple((0, max_single) for _ in range(n))
        result = minimize(
            portfolio_vol, np.ones(n) / n, method="SLSQP",
            bounds=bounds, constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if result.success:
            frontier.append({"return": target, "volatility": result.fun})

    return frontier
