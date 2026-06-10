import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from config import get_symbol_label, get_symbol_category, NON_STOCK_ASSETS

def build_asset_labels(symbols):
    return [get_symbol_label(s) for s in symbols]

def _filter_nonzero(weights):
    return {k: v for k, v in weights.items() if v > 1e-6}

def allocation_pie(weights):
    filtered = _filter_nonzero(weights)
    if not filtered:
        return go.Figure()
    symbols = list(filtered.keys())
    values = list(filtered.values())
    labels = build_asset_labels(symbols)

    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors[:len(values)]),
            textinfo="label+percent",
            textposition="outside",
        )
    ])
    fig.update_layout(title="Varlık Dağılımı", height=500, showlegend=False)
    return fig

def category_allocation_chart(weights):
    filtered = _filter_nonzero(weights)
    if not filtered:
        return go.Figure()
    symbols = list(filtered.keys())
    values = list(filtered.values())

    cat_weights = {}
    for sym, w in zip(symbols, values):
        cat = get_symbol_category(sym)
        cat_weights[cat] = cat_weights.get(cat, 0) + w

    cat_labels = list(cat_weights.keys())
    cat_values = [round(v * 100, 1) for v in cat_weights.values()]

    colors = px.colors.qualitative.Bold[:len(cat_labels)]
    fig = go.Figure(data=[
        go.Bar(
            x=cat_labels, y=cat_values,
            marker_color=colors,
            text=[f"%{v}" for v in cat_values],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Varlık Sınıfı Dağılımı (%)",
        xaxis_title="Varlık Sınıfı",
        yaxis_title="Ağırlık (%)",
        height=400,
        yaxis=dict(range=[0, 100]),
    )
    return fig

def efficient_frontier_plot(frontier, optimal_portfolio=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["volatility"] for p in frontier],
        y=[p["return"] for p in frontier],
        mode="lines+markers", name="Efficient Frontier",
        line=dict(color="blue", width=2), marker=dict(size=4),
    ))
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio["volatility"]],
            y=[optimal_portfolio["expected_return"]],
            mode="markers", name="Optimal Portföy",
            marker=dict(color="red", size=15, symbol="star",
                        line=dict(color="black", width=2)),
        ))
    fig.update_layout(
        title="Etkin Sınır (Efficient Frontier)",
        xaxis_title="Volatilite (Risk)",
        yaxis_title="Beklenen Getiri",
        height=500, hovermode="x unified",
    )
    return fig

def monte_carlo_scatter(mc_results):
    all_returns = mc_results["all_results"][:, 0]
    all_vols = mc_results["all_results"][:, 1]
    all_sharpes = mc_results["all_results"][:, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_vols, y=all_returns,
        mode="markers",
        marker=dict(
            color=all_sharpes, colorscale="Viridis", size=4,
            colorbar=dict(title="Sharpe"), showscale=True,
        ),
        name="Simule Edilen Portfoyler",
        hovertemplate="Getiri: %{y:.2%}<br>Risk: %{x:.2%}<br>Sharpe: %{marker.color:.2f}",
    ))
    fig.add_trace(go.Scatter(
        x=[mc_results["volatility"]], y=[mc_results["expected_return"]],
        mode="markers", name="Optimal Portföy",
        marker=dict(color="red", size=15, symbol="star",
                    line=dict(color="black", width=2)),
    ))
    fig.update_layout(
        title="Monte Carlo Simülasyonu",
        xaxis_title="Volatilite (Risk)",
        yaxis_title="Getiri",
        height=600,
    )
    return fig

def price_history_chart(prices, weights):
    filtered = _filter_nonzero(weights)
    if not filtered or prices.empty:
        return go.Figure()
    symbols = list(filtered.keys())
    w_arr = np.array(list(filtered.values()))
    available = [s for s in symbols if s in prices.columns]
    if not available:
        return go.Figure()

    normalized = prices[available] / prices[available].iloc[0]
    w_sub = np.array([filtered[s] for s in available])
    portfolio_value = normalized @ w_sub

    fig = go.Figure()

    for col in available:
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col],
            mode="lines", name=col,
            line=dict(width=1), opacity=0.5,
        ))

    fig.add_trace(go.Scatter(
        x=portfolio_value.index, y=portfolio_value,
        mode="lines", name="Portfoy",
        line=dict(color="black", width=3),
    ))
    fig.update_layout(
        title="Fiyat Geçmişi (Normalize)",
        xaxis_title="Tarih",
        yaxis_title="Normalize Fiyat",
        height=500, hovermode="x unified",
    )
    return fig

def correlation_heatmap(prices):
    corr = prices.pct_change().dropna().corr()
    labels = build_asset_labels(corr.columns)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont={"size": 10},
    ))
    fig.update_layout(
        title="Korelasyon Matrisi (Isı Haritası)",
        height=600, width=800,
        xaxis=dict(tickangle=45),
    )
    return fig
