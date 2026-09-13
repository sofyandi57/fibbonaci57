"""
InvezGo Fibonacci Pullback Analyzer
====================================
Streamlit app yang menganalisis saham IDX dengan strategi
Weak/Strong Pullback Fibonacci (berdasarkan materi webinar
"The Final Hunt" - Muhamad Fatah Al-Falah, RHB Sekuritas).

Data: InvezGo API (https://api.invezgo.com)
- Auth : Authorization: Bearer <API_KEY>  (dari st.secrets)
- OHLCV: GET /analysis/chart/stock/{code}?from=&to=

Disclaimer: aplikasi ini untuk EDUKASI, bukan rekomendasi beli/jual.
"""

import os
from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# KONFIGURASI
# --------------------------------------------------------------------------
BASE_URL = "https://api.invezgo.com"

FIB_LEVELS = [0.0, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT = [1.272, 1.414, 1.618, 2.0, 2.618]
SEQ = [0.382, 0.5, 0.618, 0.786]  # level fib pullback yang dipantau

st.set_page_config(
    page_title="InvezGo Fib Pullback",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# API CLIENT
# --------------------------------------------------------------------------
def get_api_key() -> str:
    """Ambil API key dari Streamlit secrets (bukan hardcode!)."""
    try:
        return st.secrets["INVEZGO_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error(
            "API key tidak ditemukan. Tambahkan `INVEZGO_API_KEY` di "
            "**Settings → Secrets** Streamlit Cloud, atau buat file "
            "`.streamlit/secrets.toml` untuk lokal."
        )
        st.stop()


@st.cache_data(ttl=900, show_spinner="Mengambil data dari InvezGo API...")
def fetch_daily_chart(code: str, days: int):
    """OHLCV harian dari endpoint /analysis/chart/stock/{code}."""
    api_key = get_api_key()
    frm = (date.today() - timedelta(days=days)).isoformat()
    to = date.today().isoformat()
    r = requests.get(
        f"{BASE_URL}/analysis/chart/stock/{code}",
        params={"from": frm, "to": to},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if r.status_code == 401:
        st.error("401: API key tidak valid / expired.")
        st.stop()
    if r.status_code == 402:
        st.error("402: paket berlangganan tidak mencukupi untuk endpoint chart.")
        st.stop()
    if r.status_code == 429:
        st.warning("429: rate limit. Tunggu sebentar lalu coba lagi.")
        st.stop()
    r.raise_for_status()
    if r.status_code == 204 or not r.json():
        return None
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# LOGIKA FIBONACCI (aturan webinar)
# --------------------------------------------------------------------------
def find_swings(df: pd.DataFrame):
    """
    Swing high valid : candle BERIKUTNYA close < body candle SEBELUM puncak
    Swing low  valid : candle BERIKUTNYA close > body candle SEBELUM lembah
    """
    highs, lows = [], []
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    for i in range(1, len(df) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and c[i + 1] < min(o[i - 1], c[i - 1]):
            highs.append((i, h[i]))
        if l[i] < l[i - 1] and l[i] < l[i + 1] and c[i + 1] > max(o[i - 1], c[i - 1]):
            lows.append((i, l[i]))
    return highs, lows


def fib_map(low: float, high: float):
    rng = high - low
    retr = {lvl: high - rng * lvl for lvl in FIB_LEVELS}
    ext = {lvl: high + rng * (lvl - 1) for lvl in FIB_EXT}
    return retr, ext


def market_structure(df, highs, lows):
    """Klasifikasi uptrend / downtrend / sideways berdasarkan swing terakhir."""
    if len(highs) < 2 or len(lows) < 2:
        return "DATA KURANG (swing belum terbentuk)", None
    lh, ll = highs[-1], lows[-1]
    ph, pl = highs[-2], lows[-2]
    if lh[1] > ph[1] and ll[1] > pl[1]:
        trend = "UPTREND"
    elif lh[1] < ph[1] and ll[1] < pl[1]:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"
    return trend, (ll, lh)  # (swing low terakhir, swing high terakhir)


def analyze_signal(df, low_p, high_p, max_risk_pct, entry_tol=0.01):
    """Deteksi weak/strong pullback + rencana entry/exit."""
    retr, ext = fib_map(low_p, high_p)
    close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    bullish_reversal = close > float(df["open"].iloc[-1]) and prev_close < float(
        df["open"].iloc[-2]
    )

    # ---- WEAK PULLBACK: rebound di level fib setelah turun 1 level ----
    for lvl in SEQ:
        level_price = retr[lvl]
        touched = abs(close - level_price) / level_price <= entry_tol
        higher = [x for x in SEQ if x < lvl]
        came_from_above = True
        for k in (2, 3):
            if k <= len(df) - 1:
                ck = float(df["close"].iloc[-k])
                came_from_above &= all(ck > retr[x] for x in higher)
        if touched and bullish_reversal and came_from_above:
            sl = level_price * (1 - max_risk_pct / 200)
            risk = close - sl
            return {
                "signal": "WEAK PULLBACK (speculative buy)",
                "fib_level": f"{lvl*100:.1f}%",
                "entry": close,
                "stop_loss": sl,
                "risk_pct": (close - sl) / close * 100,
                "tp1": close + risk,          # 1 : 1
                "tp2": min(ext[1.272], close + 2 * risk),
                "tp3": ext[1.618],
            }, retr, ext

    # ---- STRONG PULLBACK: tembus 1 level fib → tunggu level bawahnya ----
    for i, lvl in enumerate(SEQ):
        level_price = retr[lvl]
        if prev_close > level_price and close < level_price:
            nxt = retr[SEQ[i + 1]] if i + 1 < len(SEQ) else low_p
            risk_pct = (close - nxt * 0.99) / close * 100
            if abs(risk_pct) <= max_risk_pct:
                return {
                    "signal": "STRONG PULLBACK (wait lower fib)",
                    "fib_level": f"{lvl*100:.1f}% BREAKDOWN",
                    "entry": nxt,
                    "stop_loss": nxt * (1 - max_risk_pct / 100),
                    "risk_pct": abs(risk_pct),
                    "tp1": close,
                    "tp2": retr[lvl],
                    "tp3": ext[1.272],
                }, retr, ext
            return {
                "signal": "SKIP",
                "fib_level": "-",
                "entry": None,
                "stop_loss": None,
                "risk_pct": abs(risk_pct),
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "note": f"jarak ke fib bawah {abs(risk_pct):.1f}% > batas risiko {max_risk_pct}% — jangan dipaksakan",
            }, retr, ext

    return {
        "signal": "TIDAK ADA SETUP",
        "fib_level": "-",
        "entry": None,
        "stop_loss": None,
        "risk_pct": 0,
        "tp1": None,
        "tp2": None,
        "tp3": None,
    }, retr, ext


# --------------------------------------------------------------------------
# CHART
# --------------------------------------------------------------------------
def plot_chart(df, retr, low_p, high_p):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["close"], color="#333", lw=1.5, label="Close")
    colors = {0.0: "red", 0.382: "orange", 0.5: "gold",
              0.618: "green", 0.786: "blue", 1.0: "red"}
    for lvl, price in retr.items():
        ax.axhline(price, ls="--", lw=1, alpha=0.7,
                   color=colors.get(lvl, "gray"))
        ax.text(df["date"].iloc[2], price, f" {lvl*100:.1f}%",
                va="bottom", fontsize=8, color=colors.get(lvl, "gray"))
    ax.axhline(high_p, color="green", ls="-", lw=1, alpha=0.4)
    ax.axhline(low_p, color="red", ls="-", lw=1, alpha=0.4)
    ax.set_title("Harga vs Level Fibonacci (swing low → swing high)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📈 InvezGo — Fibonacci Pullback Analyzer (IDX)")
st.caption(
    "Edukasi teknikal berdasarkan strategi Weak/Strong Pullback Fibonacci. "
    "Bukan rekomendasi beli/jual."
)

with st.sidebar:
    st.header("⚙️ Pengaturan")
    ticker = st.text_input("Kode Saham (IDX)", value="BBCA").upper().strip()
    lookback = st.slider("Lookback (hari)", 60, 365, 200)
    max_risk = st.slider("Batas risiko maksimum (%)", 3, 15, 8)
    run = st.button("🔍 Analisis", type="primary", use_container_width=True)
    st.divider()
    st.caption("API key diambil dari Streamlit Secrets.")

if not run:
    st.info("Masukkan kode saham di sidebar lalu klik **Analisis**.")
    st.stop()

df = fetch_daily_chart(ticker, lookback)
if df is None or len(df) < 30:
    st.warning(f"Data {ticker} tidak tersedia (saham baru IPO/suspend/delisting).")
    st.stop()

highs, lows = find_swings(df)
structure, anchor = market_structure(df, highs, lows)
current_price = float(df["close"].iloc[-1])

# --- Row 1: metrik utama ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Harga Saat Ini", f"{current_price:,.0f}")
c2.metric("Market Structure", structure)
c3.metric("Swing High", f"{anchor[1][1]:,.0f}" if anchor else "-")
c4.metric("Swing Low", f"{anchor[0][1]:,.0f}" if anchor else "-")

if not anchor:
    st.warning("Swing point belum cukup — perpanjang lookback.")
    st.stop()

low_p, high_p = anchor[0][1], anchor[1][1]
retr, ext = fib_map(low_p, high_p)

# --- Row 2: tabel fib ---
st.subheader("🎯 Target Fibonacci (retracement & extension)")
tbl = pd.DataFrame(
    {
        "Level": [f"{l*100:.1f}%" for l in FIB_LEVELS] + [f"{l*100:.1f}%" for l in FIB_EXT],
        "Jenis": ["Retracement"] * len(FIB_LEVELS) + ["Extension"] * len(FIB_EXT),
        "Harga": [retr[l] for l in FIB_LEVELS] + [ext[l] for l in FIB_EXT],
    }
)
tbl["Jarak dari harga sekarang"] = ((tbl["Harga"] / current_price - 1) * 100).round(2).astype(str) + " %"
st.dataframe(tbl, use_container_width=True, hide_index=True)

# --- Row 3: sinyal & rencana trading ---
st.subheader("🚦 Sinyal & Rencana Trading")
sig, _, _ = analyze_signal(df, low_p, high_p, max_risk)
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Sinyal", sig["signal"])
sc2.metric("Level Fib Terkait", sig["fib_level"])
sc3.metric("Risiko", f"{sig['risk_pct']:.2f}%")

if sig.get("note"):
    st.warning(sig["note"])

if sig["entry"]:
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Entry", f"{sig['entry']:,.0f}")
    e2.metric("Stop Loss", f"{sig['stop_loss']:,.0f}", delta=f"-{sig['risk_pct']:.1f}%")
    e3.metric("TP1 (1:1)", f"{sig['tp1']:,.0f}")
    e4.metric("TP2", f"{sig['tp2']:,.0f}")
    e5.metric("TP3 (ext 161.8%)", f"{sig['tp3']:,.0f}")
    rr = (sig["tp1"] - sig["entry"]) / max(sig["entry"] - sig["stop_loss"], 1e-9)
    st.caption(f"Risk : Reward TP1 ≈ 1 : {rr:.2f}")

# --- Row 4: chart ---
st.subheader("📊 Chart")
st.pyplot(plot_chart(df, retr, low_p, high_p), use_container_width=True)

with st.expander("ℹ️ Cara membaca"):
    st.markdown(
        """
        - **Swing high/low** hanya valid jika candle berikutnya *menutupi body*
          candle sebelum titik tersebut (aturan konfirmasi).
        - **Weak pullback** → harga rebound setelah turun satu level fib:
          speculative buy, SL di bawah level fib.
        - **Strong pullback** → harga menembus satu level fib: tunggu di level
          fib berikutnya, asalkan jaraknya masih dalam batas risiko kamu.
        - Fibonacci bukan prediktor — selalu kombinasikan dengan indikator &
          price action lain.
        """
    )
