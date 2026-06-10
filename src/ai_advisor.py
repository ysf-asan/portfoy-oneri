"""Groq LLM tabanlı AI yatırım danışmanı.

Portföy önerilerini doğal dilde yorumlar ve hisse seçimine
profil-bazlı bağlam ekler. API anahtarı .env'den okunur.
"""
import os
import json
from functools import lru_cache

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _load_api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    # .env dosyasını manuel oku (python-dotenv olmadan da çalışsın)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


@lru_cache(maxsize=1)
def get_client():
    if not _GROQ_AVAILABLE:
        return None
    key = _load_api_key()
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None


def is_available():
    return get_client() is not None


def _chat(messages, model=DEFAULT_MODEL, temperature=0.4, max_tokens=1200):
    client = get_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"__ERROR__:{e}"


def explain_portfolio(profile_label, answers, weights, metrics, ticker_info):
    """Önerilen portföyü doğal dilde, kişiye özel açıklar."""
    holdings = []
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w < 1e-6:
            continue
        info = ticker_info.get(sym, {})
        holdings.append(
            f"- {sym} ({info.get('name', sym)}): %{w*100:.1f}, "
            f"sektör: {info.get('sector', '—')}"
        )
    holdings_str = "\n".join(holdings)

    answer_map = {
        "age": "yaş aralığı skoru", "income": "gelir skoru",
        "horizon": "vade skoru", "loss_tolerance": "kayıp toleransı skoru",
        "goal": "hedef skoru",
    }
    profile_lines = [f"{answer_map.get(k, k)}: {v}" for k, v in answers.items()]

    system = (
        "Sen deneyimli bir Türk yatırım danışmanısın. Sade, anlaşılır Türkçe "
        "kullan. Yatırım tavsiyesi değil, eğitim amaçlı açıklama yaptığını "
        "belirt. Abartılı vaatlerden kaçın, riskleri dürüstçe anlat.\n"
        "ÖNEMLİ TUTARLILIK KURALLARI:\n"
        f"- Kullanıcının risk profili '{profile_label}' olarak SABİTTİR. Onu "
        "başka bir profille (örn. agresif/muhafazakâr) karıştırma, çelişme.\n"
        "- Portföyün riskini SANA VERİLEN gerçek metriklere dayandır "
        "(volatilite, max drawdown). Kendi kafandan rakam uydurma.\n"
        "- Kullanıcının profili ile portföyün risk seviyesini tutarlı anlat: "
        f"'{profile_label}' bir yatırımcıya bu portföy neden uygun, bunu "
        "metriklerle bağdaştır. Çelişkili ifadeler kurma."
    )
    user = (
        f"Kullanıcının risk profili: {profile_label} (bu kesindir)\n"
        f"Anket skorları (0=düşük risk eğilimi, 3=yüksek):\n" + "\n".join(profile_lines) + "\n\n"
        f"Önerilen portföy:\n{holdings_str}\n\n"
        f"Bu portföyün GERÇEK ölçülen risk metrikleri:\n"
        f"- Yıllık beklenen getiri: %{metrics.get('annual_return', 0)*100:.1f}\n"
        f"- Yıllık volatilite (oynaklık): %{metrics.get('annual_vol', 0)*100:.1f}\n"
        f"- Sharpe oranı: {metrics.get('sharpe', 0)}\n"
        f"- Maksimum düşüş (max drawdown): %{metrics.get('max_drawdown', 0)*100:.1f}\n\n"
        f"Bu '{profile_label}' profiline sahip kullanıcıya portföyü 3 kısımda açıkla:\n"
        "1) Genel strateji: bu dağılım neden onun risk profiline uygun\n"
        "2) Öne çıkan 2-3 varlığın portföydeki rolü\n"
        "3) Riskler: yukarıdaki gerçek volatilite ve max drawdown rakamlarına "
        "atıfla, dürüstçe ama profille tutarlı şekilde anlat\n"
        "Maddeler kısa olsun, toplam 250 kelimeyi geçme. Profil etiketiyle çelişme."
    )
    return _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


def translate_summaries(summaries):
    """{sembol: ingilizce_ozet} sözlüğünü tek seferde Türkçeye çevirir.

    Dönüş: {sembol: turkce_ozet}. AI yoksa orijinali döndürür.
    """
    items = {k: v for k, v in summaries.items() if v and v.strip()}
    if not items:
        return dict(summaries)
    if get_client() is None:
        return dict(summaries)

    numbered = "\n\n".join(f"[{sym}]\n{txt}" for sym, txt in items.items())
    system = (
        "Sen finansal metin çevirmenisin. Sana [SEMBOL] etiketleriyle ayrılmış "
        "İngilizce şirket/varlık açıklamaları verilecek. Her birini akıcı, sade "
        "Türkçeye çevir. Aynı [SEMBOL] etiketini koruyarak yanıt ver. Sadece "
        "çeviriyi yaz, ek yorum ekleme."
    )
    out = _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": numbered},
    ], temperature=0.2, max_tokens=2000)

    result = dict(summaries)
    if not out or out.startswith("__ERROR__"):
        return result

    # [SEMBOL] etiketlerine göre yanıtı ayrıştır
    import re
    parts = re.split(r"\[([A-Z0-9\.\=\-]+)\]", out)
    # parts: ['', 'SYM1', 'metin1', 'SYM2', 'metin2', ...]
    for i in range(1, len(parts) - 1, 2):
        sym = parts[i].strip()
        txt = parts[i + 1].strip()
        if sym in result and txt:
            result[sym] = txt
    return result


def answer_question(question, context):
    """Kullanıcının portföyü hakkındaki serbest sorusunu yanıtlar."""
    system = (
        "Sen bir yatırım eğitmenisin. Kullanıcının mevcut portföyü hakkında "
        "sorularını sade Türkçe ile yanıtla. Kesin al/sat tavsiyesi verme, "
        "eğitim amaçlı bilgi ver. Bilmediğin şeyi uydurmamalısın."
    )
    user = f"Mevcut portföy bağlamı:\n{context}\n\nSoru: {question}"
    return _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.5, max_tokens=800)
