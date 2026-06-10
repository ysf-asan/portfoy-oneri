# 💰 Portföy Öneri Sistemi

Kullanıcının risk profiline uygun, optimal varlık dağılımı üreten ve önerilerini yapay zekâ ile Türkçe açıklayan bir portföy öneri uygulaması.

Markowitz Modern Portföy Teorisi (MPT) ve Monte Carlo simülasyonu ile optimize edilmiş portföyler oluşturur; BIST ve ABD hisselerini temel analiz metrikleriyle tarar; Groq (Llama 3.3 70B) ile portföyü yorumlar.

---

## ✨ Özellikler

- **Risk profili anketi** — 5 soruluk ağırlıklı puanlama ile 3 profil (Muhafazakâr / Dengeli / Agresif)
- **Kişiselleştirme** — Yaş, vade, hedef ve kayıp toleransı portföy kısıtlarını doğrudan etkiler; aynı profildeki farklı kişiler farklı portföyler alır
- **Akıllı hisse seçimi** — S&P 500 ve BIST 100 hisseleri ROE, F/K, büyüme, volatilite gibi metriklerle profile göre puanlanır
- **İki optimizasyon yöntemi** — Markowitz Etkin Sınır + Monte Carlo (15.000 simülasyon), en iyi Sharpe oranına sahip olan seçilir
- **Zengin görselleştirme** — Varlık dağılımı, etkin sınır, Monte Carlo bulutu, fiyat geçmişi, korelasyon ısı haritası (Plotly)
- **AI Danışman** — Portföyü Türkçe yorumlar, "neden bu varlık?" sorularını yanıtlar, İngilizce şirket açıklamalarını Türkçeye çevirir
- **Risk metrikleri** — Sharpe, Sortino, VaR %95, CVaR, Max Drawdown (Türkiye faiz oranı baz alınarak)

---

## 🚀 Kurulum

```bash
# 1. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 2. Groq API anahtarını ayarlayın (AI özellikleri için)
cp .env.example .env
# .env dosyasını açıp GROQ_API_KEY değerini girin
# Anahtar: https://console.groq.com/keys

# 3. Uygulamayı çalıştırın
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

> **Not:** AI özellikleri opsiyoneldir. `GROQ_API_KEY` ayarlı değilse uygulama tüm optimizasyon ve görselleştirme özellikleriyle normal çalışır; yalnızca AI Danışman sekmesi devre dışı kalır.

---

## 🗂️ Proje Yapısı

```
portfoy-oneri/
├── app.py                  → Streamlit arayüzü (anket, sekmeler, AI)
├── config.py               → Varlık listeleri, profil kısıtları, anket, ML ağırlıkları
├── requirements.txt        → Python bağımlılıkları
├── .env                    → Groq API anahtarı (git'e dahil değil)
├── src/
│   ├── data_fetcher.py     → Yahoo Finance + Kaggle veri çekme, önbellek, ticker bilgisi
│   ├── risk_profile.py     → Anket puanlama, profil eşleme, kişiselleştirilmiş kısıtlar
│   ├── stock_selector.py   → ML tabanlı hisse tarama ve puanlama
│   ├── optimizer.py        → Markowitz MPT + Monte Carlo optimizasyonu
│   ├── metrics.py          → Sharpe, Sortino, VaR, CVaR, Max Drawdown
│   ├── visualizer.py       → Plotly grafik fonksiyonları
│   └── ai_advisor.py       → Groq LLM ile yorum, soru-cevap, çeviri
├── data/                   → BIST hisse CSV'leri + S&P 500 fundamentals
├── cache/                  → Fiyat ve ticker önbelleği (parquet)
└── notebooks/              → Keşif notebook'u
```

---

## 📊 Veri Kaynakları

| Kaynak | Ne için | Nasıl |
|--------|---------|-------|
| **Yahoo Finance** (`yfinance`) | Hisse/ETF/kripto/forex fiyat geçmişi, şirket bilgisi | Canlı API, 1 günlük önbellek |
| **Kaggle — S&P 500 Fundamentals** | ABD hisse temel analiz (ROE, F/K, marj...) | `kagglehub` ile indirilir |
| **Kaggle — BIST 100 Fiyatlar** | BIST hisse fiyat geçmişi | `data/` klasöründe yerel CSV'ler |

---

## 🧠 Nasıl Çalışır?

1. **Anket** → Kullanıcı 5 soruyu yanıtlar, ağırlıklı puanla profil belirlenir
2. **Kişiselleştirme** → Yaş/vade/hedef/tolerans portföy kısıtlarını ayarlar
3. **Hisse tarama** → S&P 500 + BIST hisseleri profile göre puanlanıp en iyileri seçilir
4. **Veri çekme** → Seçilen hisseler + profile uygun ETF/emtia/forex/kripto için Yahoo'dan fiyat alınır
5. **Optimizasyon** → Markowitz ve Monte Carlo çalışır, en yüksek Sharpe'lı portföy seçilir
6. **Sunum** → Grafikler, risk metrikleri, detaylı tablo ve AI yorumu gösterilir

---

## ⚠️ Sorumluluk Reddi

Bu uygulama **eğitim ve gösterim** amaçlıdır. Üretilen portföyler ve AI yorumları **yatırım tavsiyesi değildir**. Yatırım kararları için lisanslı bir finansal danışmana başvurun.

---

## 🛠️ Teknoloji Yığını

Python · Streamlit · PyPortfolioOpt · NumPy/Pandas/SciPy · scikit-learn · Plotly · yfinance · kagglehub · Groq (Llama 3.3 70B)
