"""
InvezGo Fibonacci Pullback Analyzer + Screener + Bandar + AI
=============================================================
Streamlit app yang menganalisis saham IDX dengan:
1. Strategi Weak/Strong Pullback Fibonacci (materi webinar
   "The Final Hunt" - Muhamad Fatah Al-Falah, RHB Sekuritas)
2. Screener bandarmologi: deteksi akumulasi/distribusi bandar
   saat harga sideways, SEBELUM muncul volume besar
   (data BDM dari endpoint /analysis/chart/stock/bdm/{code})

Arsitektur:
- Storage cache  : Supabase (Postgres, persisten) jika secrets tersedia,
                   fallback otomatis ke SQLite lokal
- InvezGo API    : sumber data (hanya data yang belum ada di cache)
- Groq API       : komentar analisis AI berbahasa Indonesia

Disclaimer: aplikasi ini untuk EDUKASI, bukan rekomendasi beli/jual.
"""

import os
import sqlite3
import time
from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# KONFIGURASI
# --------------------------------------------------------------------------
BASE_URL = "https://api.invezgo.com"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fib_cache.db")

FIB_LEVELS = [0.0, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT = [1.272, 1.414, 1.618, 2.0, 2.618]
SEQ = [0.382, 0.5, 0.618, 0.786]  # level fib pullback yang dipantau

DEFAULT_WATCHLIST = "BULL,WINS,ANTM,RAJA,DMAS,KIJA,META,SSIA,POWR,MEDC"

# --- parameter screener bandar ---
SIDEWAYS_WINDOW = 40     # hari untuk mengukur lebar sideways
SIDEWAYS_MAX_WIDTH = 0.15  # (max-min)/mean <= 15% dianggap sideways
BDM_WINDOW = 20          # hari untuk menghitung dominasi BDM
BDM_ACC_THRESHOLD = 0.6  # >=60% hari BDM positif = akumulasi
BDM_DIST_THRESHOLD = 0.4  # <=40% hari BDM positif = distribusi
VOL_BREAKOUT_MULT = 1.5  # volume harian > 1.5x rata-rata 20 hari

st.set_page_config(
    page_title="InvezGo Fib Pullback",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# SECRETS
# --------------------------------------------------------------------------
def get_secret(name: str) -> str:
    try:
        return st.secrets[name]
    except (FileNotFoundError, KeyError):
        st.error(
            f"`{name}` tidak ditemukan di Streamlit Secrets. "
            "Tambahkan di **Settings → Secrets** (Cloud) atau "
            "`.streamlit/secrets.toml` (lokal)."
        )
        st.stop()


# --------------------------------------------------------------------------
# STORAGE: Supabase (utama) + SQLite (fallback lokal)
# --------------------------------------------------------------------------
# Secrets untuk Supabase (opsional — tanpa ini app otomatis pakai SQLite):
#   SUPABASE_URL = "https://xxxx.supabase.co"
#   SUPABASE_KEY = "eyJhbGciOi..."   (anon key atau service_role key)

def _use_supabase() -> bool:
    try:
        return bool(st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY"))
    except (FileNotFoundError, KeyError):
        return False


def _sb_headers() -> dict:
    key = st.secrets["SUPABASE_KEY"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",  # upsert
    }


def _sb_get(table: str, code: str, frm: str, to: str):
    """Baca baris dari Supabase (PostgREST). Return DataFrame atau None."""
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/{table}"
    try:
        r = requests.get(
            url,
            headers=_sb_headers(),
            params=[
                ("code", f"eq.{code}"),
                ("date", f"gte.{frm}"),
                ("date", f"lte.{to}"),
                ("order", "date"),
                ("limit", "10000"),
            ],
            timeout=30,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.json():
        return None
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in df.columns:
        if col not in ("code", "date"):
            df[col] = pd.to_numeric(df[col])
    return df.drop(columns=["code"])


def _sb_upsert(table: str, rows: list):
    """Upsert baris ke Supabase. `rows` = list of dict."""
    url = f"{st.secrets['SUPABASE_URL']}/rest/v1/{table}"
    r = requests.post(url, headers=_sb_headers(), json=rows, timeout=60)
    if not r.ok:
        raise RuntimeError(
            f"Supabase upsert ke '{table}' gagal ({r.status_code}): {r.text}"
        )


# ---------- SQLite fallback ----------
def db_conn():
    return sqlite3.connect(DB_PATH)


def db_init():
    with db_conn() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS candles (
                   code   TEXT NOT NULL,
                   date   TEXT NOT NULL,
                   open   REAL, high REAL, low REAL, close REAL, volume REAL,
                   PRIMARY KEY (code, date)
               )"""
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_code ON candles(code)"
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS bdm (
                   code  TEXT NOT NULL,
                   date  TEXT NOT NULL,
                   value REAL,
                   PRIMARY KEY (code, date)
               )"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS kv_cache (
                   key        TEXT PRIMARY KEY,
                   payload    TEXT,
                   fetched_at TEXT
               )"""
        )


db_init()


# ---------- dispatcher: candles ----------
def db_get_candles(code: str, frm: str, to: str):
    if _use_supabase():
        return _sb_get("candles", code, frm, to)
    with db_conn() as con:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM candles "
            "WHERE code = ? AND date BETWEEN ? AND ? ORDER BY date",
            con, params=(code, frm, to),
        )
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def db_insert_candles(code: str, df: pd.DataFrame):
    if _use_supabase():
        rows = [
            {"code": code, "date": str(r.date), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "volume": r.volume}
            for r in df.itertuples(index=False)
        ]
        _sb_upsert("candles", rows)
        return
    rows = [
        (code, str(r.date), r.open, r.high, r.low, r.close, r.volume)
        for r in df.itertuples(index=False)
    ]
    with db_conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO candles "
            "(code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            rows,
        )


# ---------- dispatcher: kv (payload JSON, utk agregat spt broker summary) ----------
def db_get_kv(key: str):
    """Return (payload_str, fetched_at_str) atau (None, None)."""
    if _use_supabase():
        url = f"{st.secrets['SUPABASE_URL']}/rest/v1/kv_cache"
        r = requests.get(
            url, headers=_sb_headers(),
            params={"key": f"eq.{key}", "limit": "1"}, timeout=30,
        )
        if r.status_code != 200 or not r.json():
            return None, None
        row = r.json()[0]
        return row["payload"], row["fetched_at"]
    with db_conn() as con:
        cur = con.execute(
            "SELECT payload, fetched_at FROM kv_cache WHERE key = ?", (key,)
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def db_set_kv(key: str, payload: str, fetched_at: str):
    if _use_supabase():
        _sb_upsert("kv_cache",
                   [{"key": key, "payload": payload, "fetched_at": fetched_at}])
        return
    with db_conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO kv_cache (key, payload, fetched_at) "
            "VALUES (?,?,?)", (key, payload, fetched_at),
        )


# ---------- dispatcher: bdm ----------
def db_get_bdm(code: str, frm: str, to: str):
    if _use_supabase():
        return _sb_get("bdm", code, frm, to)
    with db_conn() as con:
        df = pd.read_sql_query(
            "SELECT date, value FROM bdm "
            "WHERE code = ? AND date BETWEEN ? AND ? ORDER BY date",
            con, params=(code, frm, to),
        )
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def db_insert_bdm(code: str, df: pd.DataFrame):
    if _use_supabase():
        rows = [{"code": code, "date": str(r.date), "value": r.value}
                for r in df.itertuples(index=False)]
        _sb_upsert("bdm", rows)
        return
    rows = [(code, str(r.date), r.value) for r in df.itertuples(index=False)]
    with db_conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO bdm (code, date, value) VALUES (?,?,?)", rows
        )


# --------------------------------------------------------------------------
# DATA FETCH (cache SQLite dulu, API hanya untuk data yang kurang)
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_chart(code: str, days: int):
    """
    Strategi hemat kuota:
    1. Cek SQLite untuk rentang [from, to].
    2. Jika data sudah segar (<=2 hari dari hari ini) -> pakai cache saja.
    3. Jika tidak -> fetch HANYA selisihnya dari API, simpan ke SQLite.
    """
    frm = (date.today() - timedelta(days=days)).isoformat()
    to = date.today().isoformat()

    cached = db_get_candles(code, frm, to)
    if cached is not None and len(cached) >= 20:
        last_dt = cached["date"].iloc[-1]
        if (date.today() - last_dt).days <= 2:  # akhir pekan / hari libur aman
            return cached

    # butuh data baru: fetch hanya dari tanggal terakhir di cache
    api_from = frm
    if cached is not None and len(cached) > 0:
        api_from = (cached["date"].iloc[-1] + timedelta(days=1)).isoformat()
        if api_from > to:
            return cached

    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/chart/stock/{code}",
            params={"from": api_from, "to": to},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except requests.RequestException:
        return cached  # offline/API down -> pakai cache apa adanya
    if r.status_code in (204, 401, 402, 429, 404):
        return cached
    r.raise_for_status()
    payload = r.json()
    if not payload:
        return cached

    new = pd.DataFrame(payload)
    new["date"] = pd.to_datetime(new["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        new[col] = pd.to_numeric(new[col])
    db_insert_candles(code, new)

    merged = (
        pd.concat([cached, new], ignore_index=True)
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
        if cached is not None
        else new.sort_values("date").reset_index(drop=True)
    )
    return merged


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bdm(code: str, days: int):
    """
    Time series indikator bandarmologi (BDM). EOD update jam 18:00 WIB.
    Sama-sama incremental & di-cache di SQLite.
    """
    frm = (date.today() - timedelta(days=days)).isoformat()
    to = date.today().isoformat()

    cached = db_get_bdm(code, frm, to)
    if cached is not None and len(cached) >= 10:
        last_dt = cached["date"].iloc[-1]
        if (date.today() - last_dt).days <= 2:
            return cached

    api_from = frm
    if cached is not None and len(cached) > 0:
        api_from = (cached["date"].iloc[-1] + timedelta(days=1)).isoformat()
        if api_from > to:
            return cached

    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/chart/stock/bdm/{code}",
            params={"from": api_from, "to": to},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
    except requests.RequestException:
        return cached
    if r.status_code in (204, 401, 402, 429, 404):
        return cached
    r.raise_for_status()
    payload = r.json()
    if not payload:
        return cached

    new = pd.DataFrame(payload)
    new["date"] = pd.to_datetime(new["date"]).dt.date
    new["value"] = pd.to_numeric(new["value"])
    db_insert_bdm(code, new)

    merged = (
        pd.concat([cached, new], ignore_index=True)
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
        if cached is not None
        else new.sort_values("date").reset_index(drop=True)
    )
    return merged


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_list():
    """Daftar seluruh kode saham IDX dari /analysis/list/stock."""
    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/list/stock",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return [s["code"] for s in r.json()]


# --------------------------------------------------------------------------
# BROKER SUMMARY (top-3 akumulasi: broker #1 >= 2x broker #2)
# --------------------------------------------------------------------------
from datetime import datetime as _dt
from datetime import time as _dtime


def _wib_now() -> _dt:
    return _dt.utcnow() + timedelta(hours=7)


def _eod_fresh(fetched_at: str) -> bool:
    """Data broker EOD jam 18:00 WIB — segar jika diambil setelah cutoff."""
    try:
        ts = _dt.fromisoformat(fetched_at)
    except (ValueError, TypeError):
        return False
    now = _wib_now()
    cutoff = _dt.combine(now.date(), _dtime(18, 5))
    if now < cutoff:
        cutoff -= timedelta(days=1)
    return ts >= cutoff


def fetch_broker_summary(code: str, window: int):
    """
    Agregat buy/sell per broker untuk rentang [today-window, today].
    Endpoint: GET /analysis/summary/stock/{code} (EOD 18:00 WIB).
    Di-cache di kv_cache — scan ulang tidak memakai kuota API.
    """
    frm = (date.today() - timedelta(days=window)).isoformat()
    to = date.today().isoformat()
    key = f"{code}|brokersum|{window}|{frm}"

    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at and _eod_fresh(fetched_at):
        return pd.read_json(payload)

    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/summary/stock/{code}",
            params={"from": frm, "to": to, "investor": "all", "market": "RG"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
    except requests.RequestException:
        return pd.read_json(payload) if payload else None
    if r.status_code in (204, 401, 402, 429, 404):
        return pd.read_json(payload) if payload else None
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    for col in ["buy_volume", "sell_volume", "net_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    db_set_kv(key, df.to_json(), _wib_now().isoformat())
    return df


def broker_top3_accumulate(summary: pd.DataFrame):
    """
    Filter '3 broker teratas ngumpulin, broker #1 >= 2x broker #2'.
    Return dict info atau None jika data tidak cukup.
    """
    if summary is None or summary.empty:
        return None
    df = summary.copy()
    df["net"] = df["buy_volume"] - df["sell_volume"]
    top = df.sort_values("net", ascending=False).head(3)
    if len(top) < 3:
        return None
    n1, n2, n3 = (float(top["net"].iloc[i]) for i in range(3))
    b1, b2, b3 = (str(top["code"].iloc[i]) for i in range(3))
    passed = (n1 > 0 and n2 > 0 and n3 > 0) and (n1 >= 2 * n2)
    return {
        "b1": b1, "b2": b2, "b3": b3,
        "net1": n1, "net2": n2, "net3": n3,
        "ratio_1v2": (n1 / n2) if n2 > 0 else float("inf"),
        "broker_filter_pass": passed,
    }


# --------------------------------------------------------------------------
# GROQ AI
# --------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="🤖 Groq sedang menganalisis…")
def groq_chat(prompt: str) -> str:
    key = get_secret("GROQ_API_KEY")
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah analis teknikal saham IDX berpengalaman. "
                        "Jawab dalam Bahasa Indonesia, ringkas namun substantif "
                        "(maks ~400 kata). Selalu tutup dengan pengingat bahwa "
                        "analisis bersifat edukasi, bukan rekomendasi beli/jual."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 900,
        },
        timeout=90,
    )
    if r.status_code == 401:
        return "❌ GROQ_API_KEY tidak valid."
    if r.status_code == 429:
        return "⏳ Rate limit Groq — coba beberapa saat lagi."
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def build_analysis_prompt(code, structure, sig, retr, ext, recent_closes):
    closes_str = ", ".join(f"{c:,.0f}" for c in recent_closes)
    fib_str = ", ".join(f"{l*100:.1f}%={retr[l]:,.0f}" for l in FIB_LEVELS)
    return f"""Analisis teknikal singkat saham {code} (IDX):

MARKET STRUCTURE: {structure}
LEVEL FIBONACCI (swing low->high): {fib_str}
SINYAL: {sig['signal']} (level {sig['fib_level']})
RENCANA: entry={sig['entry']}, SL={sig['stop_loss']}, TP1={sig['tp1']}, TP2={sig['tp2']}, TP3={sig['tp3']}, risiko={sig['risk_pct']:.2f}%
10 CLOSE TERAKHIR: {closes_str}

Tugas:
1. Evaluasi kualitas setup ini (apakah weak/strong pullback masuk akal?).
2. Apa konfirmasi tambahan yang sebaiknya ditunggu sebelum entry?
3. Risiko utama skenario ini.
4. Catatan money management."""


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
        return "DATA KURANG", None
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
                "signal": "WEAK PULLBACK",
                "fib_level": f"{lvl*100:.1f}%",
                "entry": close,
                "stop_loss": sl,
                "risk_pct": (close - sl) / close * 100,
                "tp1": close + risk,          # 1 : 1
                "tp2": min(ext[1.272], close + 2 * risk),
                "tp3": ext[1.618],
            }, retr, ext

    # ---- STRONG PULLBACK: tembus 1 level fib -> tunggu level bawahnya ----
    for i, lvl in enumerate(SEQ):
        level_price = retr[lvl]
        if prev_close > level_price and close < level_price:
            nxt = retr[SEQ[i + 1]] if i + 1 < len(SEQ) else low_p
            risk_pct = (close - nxt * 0.99) / close * 100
            if abs(risk_pct) <= max_risk_pct:
                return {
                    "signal": "STRONG PULLBACK",
                    "fib_level": f"{lvl*100:.1f}% BREAK",
                    "entry": nxt,
                    "stop_loss": nxt * (1 - max_risk_pct / 100),
                    "risk_pct": abs(risk_pct),
                    "tp1": close,
                    "tp2": retr[lvl],
                    "tp3": ext[1.272],
                }, retr, ext
            return {
                "signal": "SKIP (risiko kebesar)",
                "fib_level": f"{lvl*100:.1f}% BREAK",
                "entry": None, "stop_loss": None, "risk_pct": abs(risk_pct),
                "tp1": None, "tp2": None, "tp3": None,
            }, retr, ext

    return {
        "signal": "TIDAK ADA SETUP",
        "fib_level": "-",
        "entry": None, "stop_loss": None, "risk_pct": 0,
        "tp1": None, "tp2": None, "tp3": None,
    }, retr, ext


def full_analysis(code: str, lookback: int, max_risk: float):
    """Jalankan seluruh pipeline fib untuk satu ticker."""
    df = fetch_daily_chart(code, lookback)
    if df is None or len(df) < 30:
        return None, None, None
    highs, lows = find_swings(df)
    structure, anchor = market_structure(df, highs, lows)
    row = {
        "code": code,
        "price": float(df["close"].iloc[-1]),
        "structure": structure if anchor else "DATA KURANG",
    }
    if anchor and structure == "UPTREND":
        sig, _, _ = analyze_signal(df, anchor[0][1], anchor[1][1], max_risk)
        row.update(sig)
    else:
        row.update({"signal": "-", "fib_level": "-", "entry": None,
                    "stop_loss": None, "risk_pct": 0,
                    "tp1": None, "tp2": None, "tp3": None})
    return row, df, anchor


# --------------------------------------------------------------------------
# LOGIKA SCREENER BANDAR (akumulasi / distribusi saat sideways)
# --------------------------------------------------------------------------
def sideways_width(closes: pd.Series, window: int = SIDEWAYS_WINDOW):
    """Lebar range (max-min)/mean. None jika data kurang."""
    if len(closes) < window:
        return None
    seg = closes.tail(window)
    mean = float(seg.mean())
    if mean == 0:
        return None
    return (float(seg.max()) - float(seg.min())) / mean


def bandar_classify(df: pd.DataFrame, bdm: pd.DataFrame):
    """
    Klasifikasi akumulasi/distribusi bandar:
    - sideways   : range 40 hari sempit (<15%)
    - acc_pct    : proporsi hari BDM > 0 dalam 20 hari terakhir
    - breakout   : volume hari ini > 1.5x rata-rata volume 20 hari terakhir
    """
    closes = df["close"]
    vols = df["volume"]
    width = sideways_width(closes)
    if width is None:
        return None

    is_sideways = width <= SIDEWAYS_MAX_WIDTH

    # dominasi BDM 20 hari terakhir (join by date agar aman)
    if bdm is None or bdm.empty:
        return None
    tail_dates = df["date"].tail(BDM_WINDOW)
    bdm_seg = bdm[bdm["date"].isin(set(tail_dates))]
    if len(bdm_seg) < BDM_WINDOW // 2:
        return None
    acc_pct = float((bdm_seg["value"] > 0).mean())
    net_bdm = float(bdm_seg["value"].sum())

    # volume breakout: hari ini > 1.5x rata-rata 20 hari terakhir
    vol_base = float(vols.tail(BDM_WINDOW + 1).iloc[:-1].mean())
    vol_today = float(vols.iloc[-1])
    vol_ratio = vol_today / vol_base if vol_base else 0
    vol_breakout = vol_ratio >= VOL_BREAKOUT_MULT

    if acc_pct >= BDM_ACC_THRESHOLD:
        stage = "AKUMULASI + BREAKOUT VOLUME" if vol_breakout else "AKUMULASI DINI (volume belum breakout)"
        kind = "AKUMULASI"
    elif acc_pct <= BDM_DIST_THRESHOLD:
        stage = "DISTRIBUSI + VOLUME BESAR" if vol_breakout else "DISTRIBUSI DINI"
        kind = "DISTRIBUSI"
    else:
        stage = "NETRAL"
        kind = "NETRAL"

    return {
        "sideways": is_sideways,
        "range_pct": round(width * 100, 2),
        "acc_pct": round(acc_pct * 100, 1),
        "net_bdm": round(net_bdm),
        "vol_ratio": round(vol_ratio, 2),
        "vol_breakout": vol_breakout,
        "kind": kind,
        "stage": stage,
    }


def bandar_analysis(code: str, lookback: int):
    """Pipeline lengkap screener bandar untuk satu ticker."""
    need = max(lookback, SIDEWAYS_WINDOW + 5, BDM_WINDOW + 5)
    df = fetch_daily_chart(code, need)
    if df is None or len(df) < SIDEWAYS_WINDOW:
        return None
    bdm = fetch_bdm(code, need)
    info = bandar_classify(df, bdm)
    if info is None:
        return None
    info["code"] = code
    info["price"] = float(df["close"].iloc[-1])
    return info


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


def plot_bandar(df, bdm):
    """Chart harga + bar BDM (hijau akumulasi / merah distribusi)."""
    merged = pd.merge(df[["date", "close"]], bdm, on="date", how="inner")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(merged["date"], merged["close"], color="#333", lw=1.5)
    ax1.set_title("Harga + Indikator Bandarmologi (BDM)")
    ax1.grid(alpha=0.2)
    colors = ["green" if v > 0 else "red" for v in merged["value"]]
    ax2.bar(merged["date"], merged["value"], color=colors, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("BDM")
    fig.autofmt_xdate()
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📈 InvezGo — Fib Pullback + Bandar Screener (IDX)")
st.caption(
    "Edukasi teknikal: strategi Weak/Strong Pullback Fibonacci + screener "
    "bandarmologi. Data di-cache di SQLite (hemat kuota API). "
    "Analisis AI oleh Groq. Bukan rekomendasi beli/jual."
)

with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio(
        "Mode",
        ["Analisis Satu Saham", "Screener Multi-Saham", "Screener Bandar"],
    )
    lookback = st.slider("Lookback (hari)", 60, 365, 200)
    max_risk = st.slider("Batas risiko maksimum (%)", 3, 15, 8)
    st.divider()
    st.caption("API key diambil dari Streamlit Secrets.")

# ==========================================================================
# MODE 1 — ANALISIS SATU SAHAM
# ==========================================================================
if mode == "Analisis Satu Saham":
    ticker = st.text_input("Kode Saham (IDX)", value="BBCA").upper().strip()
    if not st.button("🔍 Analisis", type="primary"):
        st.info("Masukkan kode saham lalu klik **Analisis**.")
        st.stop()

    df = fetch_daily_chart(ticker, lookback)
    if df is None or len(df) < 30:
        st.warning(f"Data {ticker} tidak tersedia (saham baru IPO/suspend/delisting).")
        st.stop()

    highs, lows = find_swings(df)
    structure, anchor = market_structure(df, highs, lows)
    current_price = float(df["close"].iloc[-1])

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

    st.subheader("🎯 Target Fibonacci (retracement & extension)")
    tbl = pd.DataFrame(
        {
            "Level": [f"{l*100:.1f}%" for l in FIB_LEVELS] + [f"{l*100:.1f}%" for l in FIB_EXT],
            "Jenis": ["Retracement"] * len(FIB_LEVELS) + ["Extension"] * len(FIB_EXT),
            "Harga": [retr[l] for l in FIB_LEVELS] + [ext[l] for l in FIB_EXT],
        }
    )
    tbl["Jarak dari harga sekarang"] = (
        ((tbl["Harga"] / current_price - 1) * 100).round(2).astype(str) + " %"
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.subheader("🚦 Sinyal & Rencana Trading")
    sig, _, _ = analyze_signal(df, low_p, high_p, max_risk)
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Sinyal", sig["signal"])
    sc2.metric("Level Fib Terkait", sig["fib_level"])
    sc3.metric("Risiko", f"{sig['risk_pct']:.2f}%")

    if sig["entry"]:
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Entry", f"{sig['entry']:,.0f}")
        e2.metric("Stop Loss", f"{sig['stop_loss']:,.0f}", delta=f"-{sig['risk_pct']:.1f}%")
        e3.metric("TP1 (1:1)", f"{sig['tp1']:,.0f}")
        e4.metric("TP2", f"{sig['tp2']:,.0f}")
        e5.metric("TP3 (ext 161.8%)", f"{sig['tp3']:,.0f}")
        rr = (sig["tp1"] - sig["entry"]) / max(sig["entry"] - sig["stop_loss"], 1e-9)
        st.caption(f"Risk : Reward TP1 ≈ 1 : {rr:.2f}")

    st.subheader("📊 Chart")
    st.pyplot(plot_chart(df, retr, low_p, high_p), use_container_width=True)

    # --- Info bandar untuk saham ini ---
    bdm = fetch_bdm(ticker, lookback)
    if bdm is not None and not bdm.empty:
        st.subheader("🕵️ Aktivitas Bandar (BDM)")
        info = bandar_classify(df, bdm)
        if info:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Stage", info["stage"])
            b2.metric("Hari Akumulasi", f"{info['acc_pct']:.0f}%")
            b3.metric("Volume vs MA20", f"{info['vol_ratio']:.2f}x")
            b4.metric("Lebar Range 40h", f"{info['range_pct']:.1f}%")
        st.pyplot(plot_bandar(df, bdm), use_container_width=True)

    # --- Top-3 broker akumulasi ---
    with st.expander("🏦 Top 3 Broker (net akumulasi 20 hari)"):
        bsum = fetch_broker_summary(ticker, 20)
        binfo = broker_top3_accumulate(bsum)
        if binfo:
            bb1, bb2, bb3, bb4 = st.columns(4)
            bb1.metric(f"#1 {binfo['b1']}", f"{binfo['net1']:,.0f}")
            bb2.metric(f"#2 {binfo['b2']}", f"{binfo['net2']:,.0f}")
            bb3.metric(f"#3 {binfo['b3']}", f"{binfo['net3']:,.0f}")
            bb4.metric("Rasio #1:#2", f"{binfo['ratio_1v2']:.2f}x",
                       delta="LOLOS" if binfo["broker_filter_pass"] else "GAGAL")
        else:
            st.caption("Data broker tidak cukup / endpoint butuh paket tertentu.")

    # --- Analisis AI (Groq) ---
    st.subheader("🤖 Analisis AI (Groq)")
    if st.button("✨ Minta Analisis AI", use_container_width=True):
        prompt = build_analysis_prompt(
            ticker, structure, sig, retr, ext,
            [float(x) for x in df["close"].tail(10)],
        )
        try:
            st.markdown(groq_chat(prompt))
        except Exception as e:
            st.error(f"Groq error: {e}")

# ==========================================================================
# MODE 2 — SCREENER MULTI-SAHAM (FIB)
# ==========================================================================
elif mode == "Screener Multi-Saham":
    st.subheader("🔎 Screener Weak/Strong Pullback")
    st.caption(
        "Screener berjalan client-side (perhitungan Fibonacci butuh OHLCV "
        "historis per saham). Hasil scan otomatis di-cache di SQLite — scan "
        "ulang ticker yang sama tidak memakai kuota API lagi."
    )

    uni_col, lim_col = st.columns(2)
    universe = uni_col.radio(
        "Universe",
        ["Watchlist custom", "Semua saham IDX"],
        help="Semua saham IDX = ±900 ticker, butuh waktu lama & banyak kuota API",
    )
    max_stocks = lim_col.slider("Maks. saham discan", 5, 200, 20)

    if universe == "Watchlist custom":
        tickers_raw = st.text_area(
            "Daftar ticker (pisah koma)", value=DEFAULT_WATCHLIST, height=80
        )
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        all_codes = fetch_stock_list()
        if not all_codes:
            st.error("Gagal ambil daftar saham — cek API key / paket langganan.")
            st.stop()
        tickers = all_codes[:max_stocks]
        st.caption(f"Universe: {len(all_codes)} saham IDX — discan {len(tickers)} pertama.")

    tickers = tickers[:max_stocks]

    if not st.button("▶️ Jalankan Screener", type="primary"):
        st.stop()

    rows, errors = [], 0
    prog = st.progress(0, text="Memulai scan…")
    for i, code in enumerate(tickers):
        prog.progress(
            (i + 1) / len(tickers), text=f"Scan {code} ({i+1}/{len(tickers)})…"
        )
        try:
            row, _, _ = full_analysis(code, lookback, max_risk)
            if row:
                rows.append(row)
        except Exception:
            errors += 1
        time.sleep(0.05)
    prog.empty()

    if not rows:
        st.warning("Tidak ada data yang berhasil diambil.")
        st.stop()

    res = pd.DataFrame(rows)
    res = res.sort_values(
        by="signal",
        key=lambda s: s.map(
            {"WEAK PULLBACK": 0, "STRONG PULLBACK": 1, "SKIP (risiko kebesar)": 2}
        ).fillna(9),
    ).reset_index(drop=True)

    st.success(f"Scan selesai: {len(res)} saham teranalisis, {errors} gagal.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Weak Pullback", int((res["signal"] == "WEAK PULLBACK").sum()))
    c2.metric("Strong Pullback", int((res["signal"] == "STRONG PULLBACK").sum()))
    c3.metric("Uptrend", int((res["structure"] == "UPTREND").sum()))

    f1, f2 = st.columns(2)
    sig_filter = f1.multiselect(
        "Filter sinyal",
        options=res["signal"].unique().tolist(),
        default=[s for s in ["WEAK PULLBACK", "STRONG PULLBACK"] if s in res["signal"].unique()],
    )
    str_filter = f2.multiselect(
        "Filter struktur",
        options=res["structure"].unique().tolist(),
        default=["UPTREND"],
    )
    view = res[res["signal"].isin(sig_filter) & res["structure"].isin(str_filter)]
    st.dataframe(view, use_container_width=True, hide_index=True)

    dl_col, ai_col = st.columns(2)
    dl_col.download_button(
        "⬇️ Download hasil (CSV)",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name=f"fib_screener_{date.today().isoformat()}.csv",
        mime="text/csv",
    )

    if ai_col.button("✨ Ringkasan AI hasil tersaring", use_container_width=True):
        if view.empty:
            st.warning("Hasil tersaring kosong — longgarkan filter.")
        else:
            summary = view.dropna(axis=1, how="all").to_string(index=False)
            prompt = (
                "Berikut hasil screener Fibonacci pullback untuk saham IDX:\n\n"
                f"{summary}\n\n"
                "Buat ringkasan dalam Bahasa Indonesia: saham mana yang paling "
                "menarik dan mengapa, apa risiko umumnya, dan saran "
                "tindak lanjut (bukan rekomendasi beli/jual)."
            )
            try:
                st.markdown(groq_chat(prompt))
            except Exception as e:
                st.error(f"Groq error: {e}")

# ==========================================================================
# MODE 3 — SCREENER BANDAR (AKUMULASI / DISTRIBUSI)
# ==========================================================================
else:
    st.subheader("🕵️ Screener Bandarmologi: Akumulasi vs Distribusi")
    st.markdown(
        """
        Mendeteksi saham **sideways yang sudah diakumulasi/distribusi bandar**
        berdasarkan indikator BDM invEZGo — **termasuk fase dini sebelum
        volume breakout**. Konfirmasi volume: volume hari ini > **1.5x**
        rata-rata 20 hari terakhir.
        """
    )

    uni_col, lim_col = st.columns(2)
    universe = uni_col.radio(
        "Universe",
        ["Watchlist custom", "Semua saham IDX"],
        help="Semua saham IDX = ±900 ticker — perhatikan kuota API",
    )
    max_stocks = lim_col.slider("Maks. saham discan", 5, 200, 20)

    if universe == "Watchlist custom":
        tickers_raw = st.text_area(
            "Daftar ticker (pisah koma)", value=DEFAULT_WATCHLIST, height=80
        )
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        all_codes = fetch_stock_list()
        if not all_codes:
            st.error("Gagal ambil daftar saham — cek API key / paket langganan.")
            st.stop()
        tickers = all_codes[:max_stocks]
        st.caption(f"Universe: {len(all_codes)} saham IDX — discan {len(tickers)} pertama.")

    tickers = tickers[:max_stocks]

    # --- filter kedua: konsentrasi broker ---
    st.markdown("**Filter kedua (opsional): konsentrasi broker**")
    use_broker_filter = st.checkbox(
        "3 broker teratas sedang ngumpulin & broker #1 ≥ 2x broker #2",
        value=False,
    )
    broker_window = st.selectbox("Rentang agregasi broker", [10, 20, 60], index=1)

    run_col, note_col = st.columns([1, 3])
    run_scan = run_col.button("▶️ Scan Bandar", type="primary")
    note_col.caption("Sekali scan — hasil disimpan di session; tombol biru/merah di bawah hanya memfilter (hemat kuota).")

    if run_scan:
        rows, errors = [], 0
        prog = st.progress(0, text="Memulai scan bandar…")
        for i, code in enumerate(tickers):
            prog.progress(
                (i + 1) / len(tickers), text=f"Scan {code} ({i+1}/{len(tickers)})…"
            )
            try:
                row = bandar_analysis(code, lookback)
                if row:
                    if use_broker_filter:
                        binfo = broker_top3_accumulate(
                            fetch_broker_summary(code, broker_window)
                        )
                        row.update(binfo or {})
                    rows.append(row)
            except Exception:
                errors += 1
            time.sleep(0.05)
        prog.empty()
        st.session_state["bandar_rows"] = rows
        st.session_state["bandar_errors"] = errors

    if "bandar_rows" not in st.session_state or not st.session_state["bandar_rows"]:
        if run_scan:
            st.warning("Tidak ada data bandar yang berhasil diambil (BDM butuh paket tertentu).")
        else:
            st.info("Klik **▶️ Scan Bandar** untuk memulai.")
        st.stop()

    rows = st.session_state["bandar_rows"]
    errors = st.session_state.get("bandar_errors", 0)
    res = pd.DataFrame(rows)

    st.success(f"Scan selesai: {len(res)} saham, {errors} gagal.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akumulasi (≥60% BDM+)", int((res["kind"] == "AKUMULASI").sum()))
    c2.metric("— di antaranya breakout vol", int(((res["kind"] == "AKUMULASI") & res["vol_breakout"]).sum()))
    c3.metric("Distribusi (≤40% BDM+)", int((res["kind"] == "DISTRIBUSI").sum()))
    c4.metric("Sideways sempit", int(res["sideways"].sum()))

    only_side = st.checkbox("Hanya tampilkan yang sideways (range ≤15%)", value=True)

    f1, f2, f3 = st.columns(3)
    show_acc = f1.button("🔵 Tampilkan Akumulasi", use_container_width=True)
    show_dist = f2.button("🔴 Tampilkan Distribusi", use_container_width=True)
    show_all = f3.button("⚪ Tampilkan Semua", use_container_width=True)

    if "bandar_filter" not in st.session_state:
        st.session_state["bandar_filter"] = "AKUMULASI"
    if show_acc:
        st.session_state["bandar_filter"] = "AKUMULASI"
    if show_dist:
        st.session_state["bandar_filter"] = "DISTRIBUSI"
    if show_all:
        st.session_state["bandar_filter"] = "ALL"

    active = st.session_state["bandar_filter"]
    view = res.copy()
    if active != "ALL":
        view = view[view["kind"] == active]
    if only_side:
        view = view[view["sideways"]]
    if use_broker_filter and "broker_filter_pass" in view.columns:
        view = view[view["broker_filter_pass"] == True]  # noqa: E712

    cols = ["code", "price", "stage", "range_pct", "acc_pct", "net_bdm",
            "vol_ratio", "sideways"]
    if use_broker_filter and "b1" in view.columns:
        cols += ["b1", "b2", "b3", "ratio_1v2"]
    view = view.sort_values(
        ["vol_breakout", "acc_pct" if active == "DISTRIBUSI" else "acc_pct"],
        ascending=[False, active == "DISTRIBUSI"],
    )
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    if use_broker_filter and view.empty:
        st.warning(
            "Tidak ada yang lolos kombinasi filter — coba longgarkan "
            "(matikan 'hanya sideways' atau perpendek rentang broker)."
        )

    st.download_button(
        "⬇️ Download hasil (CSV)",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name=f"bandar_screener_{date.today().isoformat()}.csv",
        mime="text/csv",
    )

    if st.button("✨ Ringkasan AI hasil tersaring", use_container_width=True):
        if view.empty:
            st.warning("Hasil tersaring kosong — longgarkan filter.")
        else:
            summary = view[cols].to_string(index=False)
            prompt = (
                "Berikut hasil screener bandarmologi (BDM) untuk saham IDX:\n\n"
                f"{summary}\n\n"
                "Buat ringkasan Bahasa Indonesia: mana yang akumulasi paling "
                "kuat, mana yang waspada distribusi, arti fase 'DINI' vs "
                "'BREAKOUT VOLUME', dan hal yang perlu dikonfirmasi sebelum "
                "entry. Ingatkan bahwa ini bukan rekomendasi beli/jual."
            )
            try:
                st.markdown(groq_chat(prompt))
            except Exception as e:
                st.error(f"Groq error: {e}")

# --------------------------------------------------------------------------
with st.expander("ℹ️ Cara membaca"):
    st.markdown(
        """
        - **Cache database** — data OHLCV & BDM tersimpan di **Supabase**
          (persisten, shared antar instance) atau fallback SQLite lokal
          (`fib_cache.db`); app hanya menarik data yang belum ada di cache →
          kuota invEZGo hemat.
        - **Swing high/low** hanya valid jika candle berikutnya *menutupi body*
          candle sebelum titik tersebut (aturan konfirmasi).
        - **Weak pullback** → harga rebound setelah turun satu level fib:
          speculative buy, SL di bawah level fib.
        - **Strong pullback** → harga menembus satu level fib: tunggu di level
          fib berikutnya, asalkan jaraknya masih dalam batas risiko kamu.
        - **Screener Bandar** — `AKUMULASI/DISTRIBUSI DINI` = dominasi BDM
          sudah terlihat tapi volume belum breakout (fase stealth);
          `+ BREAKOUT VOLUME` = volume hari ini > 1.5x rata-rata 20 hari,
          konfirmasi lanjutan.
        - **Analisis AI (Groq)** — komentar LLM untuk second opinion; selalu
          verifikasi sendiri, bukan sinyal otomatis.
        - Fibonacci & BDM bukan prediktor — kombinasikan dengan price action.
        """
    )
