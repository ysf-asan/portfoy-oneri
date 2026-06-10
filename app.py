import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from config import (
    HISTORY_PERIOD, HISTORY_INTERVAL, NON_STOCK_ASSETS,
    PROFILE_ASSET_FILTER, get_symbol_category, get_symbol_label,
)
from src.data_fetcher import fetch_data, get_correlation, fetch_ticker_info
from src.risk_profile import map_answers_to_profile, get_questionnaire
from src.optimizer import (
    markowitz_optimize, monte_carlo_optimize,
    apply_cardinality, get_efficient_frontier,
)
from src.metrics import calculate_all_metrics
from src.ai_advisor import (
    is_available as ai_available, explain_portfolio,
    answer_question, translate_summaries,
)
from src.visualizer import (
    allocation_pie, category_allocation_chart,
    efficient_frontier_plot, monte_carlo_scatter,
    price_history_chart, correlation_heatmap,
)

def _recommendation_reason(sym, weight, category, metrics, constraints, answers):
    """Bir varlığın neden önerildiğini kısa Türkçe cümleyle açıkla."""
    reasons = []
    goal = answers.get("goal", 1)
    horizon = answers.get("horizon", 1)
    loss_tol = answers.get("loss_tolerance", 1)

    if category in ("Tahvil ETF",):
        reasons.append("Sermayeyi korur, düşük volatilite sağlar")
    elif category == "Emtia" and "GC=F" in sym:
        reasons.append("Enflasyona karşı koruma, döviz koruması")
    elif category == "Forex":
        reasons.append("TL değer kaybına karşı döviz çıpası")
    elif category == "Kripto":
        if goal >= 2:
            reasons.append("Yüksek büyüme potansiyeli (yüksek risk)")
        else:
            reasons.append("Alternatif varlık sınıfı çeşitlendirmesi")
    elif category == "ABD ETF":
        reasons.append("Geniş piyasa maruziyeti, düşük maliyet")
    elif category == "BIST Hisse":
        reasons.append("Türk ekonomisi büyümesine katılım")
    elif category == "ABD Hisse":
        if goal >= 2:
            reasons.append("Küresel büyüme hissesi")
        else:
            reasons.append("Likid, büyük şirket hissesi")

    if weight > 0.20:
        reasons.append(f"Portföyde ağırlıklı yer ({weight*100:.0f}%) — Sharpe optimizasyonu sonucu")
    elif weight < 0.06:
        reasons.append("Çeşitlendirme amaçlı küçük pozisyon")

    if metrics.get("sharpe", 0) > 1.0:
        reasons.append("Yüksek risk/getiri oranı (Sharpe>1)")

    return "; ".join(reasons) if reasons else "Portföy optimizasyonu sonucu seçildi"


st.set_page_config(page_title="Portföy Öneri Sistemi", page_icon="💰", layout="wide")
st.title("Portföy Öneri Sistemi")
st.markdown("Risk profilinize uygun optimal yatırım portföyü oluşturun.")

with st.sidebar:
    st.header("Risk Profili Anketi")
    st.caption("Portföy önerisi için aşağıdaki soruları yanıtlayın.")

    questionnaire = get_questionnaire()
    answers = {}

    for q in questionnaire:
        opts = [opt[0] for opt in q["options"]]
        selected = st.radio(q["question"], opts, index=None, key=f"q_{q['key']}")
        if selected:
            for opt, score in q["options"]:
                if opt == selected:
                    answers[q["key"]] = score

    st.divider()
    st.header("Hisse Seçim Yöntemi")
    data_source = st.selectbox(
        "Veri kaynağı",
        options=["ML Tarama (Kaggle)", "Varsayılan Liste"],
        index=0,
    )

    calculate = st.button("Portföyü Hesapla", type="primary", use_container_width=True)

    st.divider()
    if st.button("🗑️ Cache'i Temizle", use_container_width=True):
        import glob, os
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        bist_cache = os.path.join(os.path.dirname(__file__), "data", "_bist_features.parquet")
        removed = 0
        for f in glob.glob(os.path.join(cache_dir, "*.parquet")):
            os.remove(f)
            removed += 1
        if os.path.exists(bist_cache):
            os.remove(bist_cache)
            removed += 1
        st.success(f"{removed} cache dosyası silindi.")

if calculate and len(answers) == len(questionnaire):
    profile_result = map_answers_to_profile(answers)
    constraints = profile_result["constraints"]
    card = profile_result["cardinality"]
    profile_name = profile_result["profile_name"]

    income_label = next(
        (o[0] for q in questionnaire for o in q["options"] if q["key"] == "income" and o[1] == answers["income"]),
        "-",
    )
    personal_notes = []
    if constraints.get("short_horizon"):
        personal_notes.append("⚡ Kısa vadeli: volatilite limiti düşürüldü")
    if constraints.get("capital_preservation"):
        personal_notes.append("🛡️ Sermaye koruma: tahvil ağırlığı artırıldı")
    if answers.get("age", 1) >= 2 and answers.get("horizon", 1) >= 2:
        personal_notes.append("📈 Genç + uzun vade: equity tavanı yükseltildi")
    personal_str = "\n".join(personal_notes) if personal_notes else "Standart profil kısıtları uygulandı"

    use_ml = data_source == "ML Tarama (Kaggle)"
    stock_symbols = []
    ml_details = None

    if use_ml:
        with st.status("ML hisse taraması yapılıyor...", expanded=True) as status:
            from src.stock_selector import run_selection, get_fallback_symbols
            ml_result = run_selection(profile_name, card)
            if ml_result is not None and ml_result[0]:
                stock_symbols, details_df = ml_result
                us = sum(1 for _, r in details_df.iterrows() if r["_source"] == "US")
                bst = sum(1 for _, r in details_df.iterrows() if r["_source"] == "BIST")
                st.write(f"**496 S&P 500** ve **100 BIST** havuzdan taranıp")
                st.write(f"**{len(stock_symbols)}** hisse seçildi ({us} US + {bst} BIST).")
                ml_details = details_df
            else:
                st.warning("Kaggle verisi bulunamadı, varsayılan listeye dönülüyor.")
                stock_symbols, _ = get_fallback_symbols()
            status.update(label="Hisse seçimi tamam!", state="complete")
    else:
        from src.stock_selector import get_fallback_symbols
        stock_symbols, _ = get_fallback_symbols()

    allowed_non_stocks = PROFILE_ASSET_FILTER.get(profile_name, list(NON_STOCK_ASSETS.keys()))
    all_symbols = list(dict.fromkeys(stock_symbols + allowed_non_stocks))
    n_stocks = len([s for s in all_symbols if s not in NON_STOCK_ASSETS])

    with st.status("Veri çekiliyor...", expanded=True) as status:
        st.write("Yahoo Finance'den veri indiriliyor...")
        prices = fetch_data(all_symbols, HISTORY_PERIOD, HISTORY_INTERVAL)
        valid = len(prices.columns)
        st.write(f"{valid} varlık için veri alındı.")
        status.update(label="Veri hazır!", state="complete")

    if valid < 3:
        st.error("Yeterli veri alınamadı. Lütfen daha sonra tekrar deneyin.")
        st.stop()

    with st.status("Optimizasyon yapılıyor...", expanded=True) as status:
        st.write("Markowitz Etkin Sınır hesaplanıyor...")
        mpt = markowitz_optimize(prices, constraints)
        st.write("Monte Carlo simülasyonu çalıştırılıyor...")
        mc = monte_carlo_optimize(prices, constraints)

        if mpt is None and mc is None:
            best, method = None, None
        elif mpt is None:
            best, method = mc, "Monte Carlo"
        else:
            best = mpt
            method = "Markowitz MPT"
            if mc and mc["sharpe"] > mpt["sharpe"]:
                best, method = mc, "Monte Carlo"

        if best is None:
            st.error("Optimizasyon başarısız oldu. Lütfen tekrar deneyin.")
            st.stop()

        best["weights"] = apply_cardinality(best["weights"], card)
        metrics = calculate_all_metrics(prices, best["weights"])
        frontier = get_efficient_frontier(prices, constraints)
        status.update(label="Optimizasyon tamam!", state="complete")

    nonzero_weights = {k: v for k, v in best["weights"].items() if v > 1e-6}
    with st.spinner("Varlık bilgileri yükleniyor..."):
        ticker_info = fetch_ticker_info(list(nonzero_weights.keys()))
        # İngilizce açıklamaları AI ile Türkçeye çevir (tek seferde)
        if ai_available():
            raw_summaries = {s: ticker_info[s].get("summary", "") for s in ticker_info}
            translated = translate_summaries(raw_summaries)
            for s in ticker_info:
                ticker_info[s]["summary_tr"] = translated.get(s, ticker_info[s].get("summary", ""))

    # Tüm sonuçları session_state'e kaydet → AI butonları sayfayı sıfırlamasın
    st.session_state["result"] = {
        "profile_result": profile_result, "constraints": constraints,
        "card": card, "income_label": income_label, "personal_str": personal_str,
        "use_ml": use_ml, "ml_details": ml_details,
        "n_stocks": n_stocks, "n_symbols": len(all_symbols),
        "prices": prices, "best": best, "method": method, "mc": mc,
        "metrics": metrics, "frontier": frontier,
        "ticker_info": ticker_info, "nonzero_weights": nonzero_weights,
        "answers": answers,
    }
    # Yeni hesaplamada eski AI yanıtını temizle
    st.session_state.pop("ai_explanation", None)

elif calculate:
    st.sidebar.warning("Lütfen tüm soruları yanıtlayın.")


# ========================= RENDER (session_state'ten) =========================
if "result" in st.session_state:
    R = st.session_state["result"]
    profile_result = R["profile_result"]
    constraints = R["constraints"]
    card = R["card"]
    income_label = R["income_label"]
    personal_str = R["personal_str"]
    use_ml = R["use_ml"]
    ml_details = R["ml_details"]
    prices = R["prices"]
    best = R["best"]
    method = R["method"]
    mc = R["mc"]
    metrics = R["metrics"]
    frontier = R["frontier"]
    ticker_info = R["ticker_info"]
    nonzero_weights = R["nonzero_weights"]
    answers = R["answers"]

    st.sidebar.success(
        f"**Profiliniz:** {profile_result['profile_label']}\n\n"
        f"Puan: {profile_result['score']} / {profile_result['max_score']}\n"
        f"Gelir: {income_label}\n"
        f"Hedef varlık adedi: **{card}**\n\n"
        f"{constraints['description']}\n\n"
        f"**Kişisel ayar:**\n{personal_str}"
    )
    st.sidebar.info(f"**Hisse adedi:** {R['n_stocks']}\n**Toplam varlık:** {R['n_symbols']}")

    if 1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Beklenen Getiri (Yillik)", f"%{best['expected_return']*100:.2f}")
        col2.metric("Risk (Volatilite)", f"%{best['volatility']*100:.2f}")
        col3.metric("Sharpe Orani", f"{best['sharpe']:.3f}")

        nonzero_count = sum(1 for v in best["weights"].values() if v > 1e-6)
        st.info(
            f"**Yontem:** {method} | **Profil:** {profile_result['profile_label']} "
            f"| **Parca:** {nonzero_count}"
        )

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Varlık Dağılımı", "Grafikler", "Risk Metrikleri",
            "Korelasyon", "Detaylı Tablo", "🤖 AI Danışman",
        ])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = allocation_pie(best["weights"])
                st.plotly_chart(fig)
            with col_b:
                fig = category_allocation_chart(best["weights"])
                st.plotly_chart(fig)

        with tab2:
            col_c, col_d = st.columns(2)
            with col_c:
                if frontier:
                    fig = efficient_frontier_plot(frontier, best)
                    st.plotly_chart(fig)
            with col_d:
                if mc is not None and mc.get("all_results") is not None and len(mc["all_results"]) > 1:
                    fig = monte_carlo_scatter(mc)
                    st.plotly_chart(fig)
            fig = price_history_chart(prices, best["weights"])
            st.plotly_chart(fig)

        with tab3:
            met_cols = st.columns(4)
            metric_data = [
                ("Sharpe Oranı", metrics["sharpe"]),
                ("Sortino Oranı", metrics["sortino"]),
                ("Yıllık Volatilite", f"%{metrics['annual_vol']*100:.2f}"),
                ("Yıllık Getiri", f"%{metrics['annual_return']*100:.2f}"),
                ("Value at Risk (%95)", f"%{metrics['var_95']*100:.2f}"),
                ("Conditional VaR (%95)", f"%{metrics['cvar_95']*100:.2f}"),
                ("Max Drawdown", f"%{metrics['max_drawdown']*100:.2f}"),
            ]
            for i, (label, value) in enumerate(metric_data):
                with met_cols[i % 4]:
                    st.metric(label, value)
            st.caption("Risk-free rate: %45 (TCMB politika faizi baz alındı)")

        with tab4:
            fig = correlation_heatmap(prices)
            st.plotly_chart(fig)

        with tab5:
            nonzero = nonzero_weights

            rows = []
            for sym, w in sorted(nonzero.items(), key=lambda x: -x[1]):
                info = ticker_info.get(sym, {})
                cat = get_symbol_category(sym)
                reason = _recommendation_reason(sym, w, cat, metrics, constraints, answers)
                mc_val = info.get("market_cap")
                mc_str = f"${mc_val/1e9:.1f}B" if mc_val and mc_val > 1e9 else ("—" if not mc_val else f"${mc_val/1e6:.0f}M")
                rows.append({
                    "Sembol": sym,
                    "Şirket Adı": info.get("name") or get_symbol_label(sym),
                    "Ağırlık": f"%{w*100:.2f}",
                    "Varlık Sınıfı": cat,
                    "Sektör": info.get("sector", "—"),
                    "Ülke": info.get("country", "—"),
                    "Piyasa Değeri": mc_str,
                    "Neden Önerildi": reason,
                })

            weights_df = pd.DataFrame(rows)
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
            st.metric("Portföy Parça Sayısı", len(rows))

            st.subheader("Varlık Açıklamaları")
            for sym, w in sorted(nonzero.items(), key=lambda x: -x[1]):
                info = ticker_info.get(sym, {})
                summary = (info.get("summary_tr") or info.get("summary", "")).strip()
                if summary:
                    with st.expander(f"{sym} — {info.get('name', sym)} (%{w*100:.1f})"):
                        st.write(summary)

            csv = weights_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="CSV İndir", data=csv, file_name="portfoy_dagilimi.csv", mime="text/csv")

        with tab6:
            if not ai_available():
                st.warning(
                    "AI danışman devre dışı. `.env` dosyasına geçerli bir "
                    "`GROQ_API_KEY` ekleyin ve `pip install groq` çalıştırın."
                )
            else:
                st.caption("Groq (Llama 3.3 70B) ile desteklenmektedir. "
                           "Yatırım tavsiyesi değildir, eğitim amaçlıdır.")

                if st.button("📋 Portföyümü AI ile yorumlat", type="primary"):
                    with st.spinner("AI portföyünüzü analiz ediyor..."):
                        explanation = explain_portfolio(
                            profile_result["profile_label"], answers,
                            nonzero_weights, metrics, ticker_info,
                        )
                    if explanation and not explanation.startswith("__ERROR__"):
                        st.session_state["ai_explanation"] = explanation
                    elif explanation:
                        st.error(f"AI hatası: {explanation.replace('__ERROR__:', '')}")

                if st.session_state.get("ai_explanation"):
                    st.markdown(st.session_state["ai_explanation"])

                st.divider()
                st.subheader("Portföyün hakkında soru sor")
                user_q = st.text_input(
                    "Soru", placeholder="Örn: Neden bu kadar altın önerildi?",
                    label_visibility="collapsed",
                )
                if st.button("Sor") and user_q.strip():
                    context_lines = [
                        f"Profil: {profile_result['profile_label']}",
                        f"Yıllık getiri: %{metrics.get('annual_return', 0)*100:.1f}, "
                        f"Volatilite: %{metrics.get('annual_vol', 0)*100:.1f}, "
                        f"Sharpe: {metrics.get('sharpe', 0)}",
                        "Varlıklar:",
                    ]
                    for sym, w in sorted(nonzero_weights.items(), key=lambda x: -x[1]):
                        inf = ticker_info.get(sym, {})
                        context_lines.append(
                            f"  {sym} ({inf.get('name', sym)}, {inf.get('sector', '—')}): %{w*100:.1f}"
                        )
                    with st.spinner("AI yanıtlıyor..."):
                        ans = answer_question(user_q, "\n".join(context_lines))
                    if ans and not ans.startswith("__ERROR__"):
                        st.markdown(ans)
                    elif ans:
                        st.error(f"AI hatası: {ans.replace('__ERROR__:', '')}")

        if use_ml and ml_details is not None and not ml_details.empty:
            with st.expander("ML Tarama Detayları"):
                st.caption(
                    "S&P 500 ve BIST hisseleri, risk profilinize göre ağırlıklandırılmış "
                    "temel analiz metrikleriyle (ROE, F/K, büyüme, volatilite vb.) puanlanıp "
                    "sıralandı. Aşağıda seçilen hisseler ve skorları yer alıyor."
                )
                display_df = ml_details.copy()
                source_tr = {"US": "ABD (S&P 500)", "BIST": "BIST"}
                if "_source" in display_df.columns:
                    display_df["_source"] = display_df["_source"].map(lambda x: source_tr.get(x, x))
                if "_score" in display_df.columns:
                    display_df["_score"] = display_df["_score"].round(3)
                display_df = display_df.rename(columns={
                    "_symbol": "Sembol",
                    "_source": "Kaynak",
                    "_score": "Skor (0-1)",
                    "_rank": "Sıra",
                })
                st.dataframe(display_df, width="stretch", hide_index=True)

elif not calculate:
    st.info("Sol menüde anketi doldurup 'Portföyü Hesapla' butonuna basın.")
