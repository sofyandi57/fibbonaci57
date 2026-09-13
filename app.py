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

import io
import json
import os
import re
import sqlite3
import time
from datetime import date, timedelta

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# KONFIGURASI
# --------------------------------------------------------------------------
BASE_URL = "https://api.invezgo.com"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_DEFAULT = "openai/gpt-oss-120b"
# Fallback singkat & dikurasi (BUKAN daftar tebak-tebakan semua nama model
# Groq) -- limit Groq itu per-model per-organisasi, jadi kalau satu model
# kena rate limit/quota harian/decommissioned, model lain di daftar ini
# biasanya masih longgar. Urutan dari yang paling mirip kualitasnya ke
# model utama.
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.1-8b-instant"]
# Model gpt-oss & qwen3 adalah REASONING model -- diam-diam memakai sebagian
# max_tokens untuk "reasoning_tokens" (chain-of-thought internal) SEBELUM
# menulis jawaban asli. Tanpa reasoning_effort="low", jawaban bisa terpotong
# kosong karena max_tokens habis semua buat reasoning.
GROQ_REASONING_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b"}
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY_SECONDS = 6
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


def _sb_base_url() -> str:
    """SUPABASE_URL tanpa trailing slash atau '/rest/v1' (kalau ikut ditempel)."""
    url = st.secrets["SUPABASE_URL"].rstrip("/")
    if url.endswith("/rest/v1"):
        url = url[: -len("/rest/v1")]
    return url


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
    url = f"{_sb_base_url()}/rest/v1/{table}"
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
    url = f"{_sb_base_url()}/rest/v1/{table}"
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
        url = f"{_sb_base_url()}/rest/v1/kv_cache"
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
        return pd.read_json(io.StringIO(payload))

    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/summary/stock/{code}",
            params={"from": frm, "to": to, "investor": "all", "market": "RG"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
    except requests.RequestException:
        return pd.read_json(io.StringIO(payload)) if payload else None
    if r.status_code in (204, 401, 402, 429, 404):
        return pd.read_json(io.StringIO(payload)) if payload else None
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    for col in ["buy_volume", "sell_volume", "net_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    db_set_kv(key, df.to_json(), _wib_now().isoformat())
    return df


# --------------------------------------------------------------------------
# KEPEMILIKAN (shareholder & insider) -- versi ringkas dari konsep skill
# "analisa-kepemilikan": siapa pemegang saham & apakah insider sedang net
# beli/jual. TIDAK mencoba merekonsiliasi insider ke broker/pasar negosiasi
# atau menggambar graf relasi -- itu laporan multi-lapis terpisah, di luar
# scope screener/analyzer harian app ini.
# --------------------------------------------------------------------------
SHAREHOLDER_CACHE_TTL_HOURS = 24  # komposisi kepemilikan terbit bulanan, tidak perlu re-fetch tiap hari
INSIDER_LOOKBACK_MONTHS = 18


_STATUS_HINT = {
    401: "401 Unauthorized — API key tidak valid.",
    402: "402 Payment Required — endpoint ini butuh paket/langganan lebih tinggi dari yang kamu punya.",
    404: "404 Not Found — path endpoint kemungkinan salah/berbeda dari dokumentasi InvezGo.",
    429: "429 Rate limited — coba lagi beberapa saat.",
}


def fetch_shareholders(code: str):
    """
    Komposisi pemegang saham >1% saat ini (snapshot terbaru).
    Endpoint resmi: GET /analysis/shareholder/{code} -- verified via
    api-1.json (OpenAPI spec InvezGo). Response: list of
    {name, percentage, badge} langsung (bukan dibungkus objek).
    Cache 24 jam di kv_cache -- data ini terbit bulanan, bukan harian.
    Return (df_or_None, error_message_or_None) -- error selalu diteruskan
    ke UI, tidak ditelan diam-diam, supaya 402/404/401 bisa dibedakan.
    """
    key = f"{code}|shareholders"
    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < SHAREHOLDER_CACHE_TTL_HOURS:
                return pd.read_json(io.StringIO(payload)), None
        except (ValueError, TypeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    url = f"{BASE_URL}/analysis/shareholder/{code}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    except requests.RequestException as e:
        cached = pd.read_json(io.StringIO(payload)) if payload else None
        return cached, f"Request error: {e}"
    if not r.ok:
        cached = pd.read_json(io.StringIO(payload)) if payload else None
        hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
        return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
    data = r.json()
    if not data:
        return None, None  # 204/[] -- data belum tersedia (baru IPO/suspend), bukan error
    df = pd.DataFrame(data)
    if "percentage" in df.columns:
        df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
        df = df.sort_values("percentage", ascending=False).reset_index(drop=True)
    if "badge" in df.columns:
        df["badge"] = df["badge"].astype(str).str.strip("{}")
    db_set_kv(key, df.to_json(), _wib_now().isoformat())
    return df, None


def fetch_insider_transactions(code: str, months: int = INSIDER_LOOKBACK_MONTHS):
    """
    Riwayat transaksi insider (direksi/komisaris/pemegang mayoritas wajib
    lapor). Endpoint resmi: GET /analysis/shareholder-insider -- verified
    via api-1.json. Query params (BUKAN path param): code, from, to, page,
    limit. Response: {totalPage, page, nextPage, data: [...]}, setiap
    elemen data punya info kepemilikan sebelum/sesudah PLUS `subrow`: daftar
    transaksi aktual di pasar {date, price, status: "Buy"/"Sell", value}.
    Kita flatten subrow jadi satu baris per transaksi supaya konsisten
    dengan tabel/verdict di UI.
    Return (df_or_None, error_message_or_None).
    """
    to = date.today().isoformat()
    key = f"{code}|insider|{months}m"

    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < SHAREHOLDER_CACHE_TTL_HOURS:
                cached_df = pd.read_json(io.StringIO(payload))
                if "date" in cached_df.columns:
                    cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.date
                return cached_df, None
        except (ValueError, TypeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    url = f"{BASE_URL}/analysis/shareholder-insider"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Endpoint ini pernah balas 500 (server error InvezGo, bukan masalah
    # client) untuk rentang tanggal lebar (18 bulan). Coba mundur ke
    # rentang lebih pendek sebagai fallback sebelum menyerah -- kalau
    # akarnya memang server timeout pada window besar, window kecil masih
    # bisa jalan meski datanya jadi lebih sedikit dari yang diminta.
    last_err = None
    for try_months in sorted({months, 12, 6, 3}, reverse=True):
        if try_months > months:
            continue
        frm = (date.today() - timedelta(days=try_months * 30)).isoformat()
        try:
            r = requests.get(
                url,
                params={"code": code, "from": frm, "to": to, "page": 1, "limit": 50},
                headers=headers, timeout=30,
            )
        except requests.RequestException as e:
            last_err = f"Request error: {e}"
            continue
        if r.status_code == 500:
            last_err = f"GET {url} -> 500 Internal Server Error (window {try_months} bulan) | body: {r.text[:300]}"
            continue  # coba window lebih pendek
        if not r.ok:
            cached = pd.read_json(io.StringIO(payload)) if payload else None
            hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
            return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
        body = r.json()
        records = body.get("data", []) if isinstance(body, dict) else (body or [])
        if try_months < months and records:
            last_err = None  # berhasil dengan window dipersempit -- bukan error lagi
        break
    else:
        cached = pd.read_json(io.StringIO(payload)) if payload else None
        return cached, last_err or "Semua percobaan gagal."

    if last_err and not records:
        cached = pd.read_json(io.StringIO(payload)) if payload else None
        return cached, last_err
    if not records:
        return None, None  # tidak ada laporan insider di periode ini -- bukan error

    rows = []
    for rec in records:
        name = rec.get("name")
        badge = str(rec.get("badge", "")).strip("{}")
        for tx in (rec.get("subrow") or []):
            rows.append({
                "date": tx.get("date"),
                "name": name,
                "badge": badge,
                "action": tx.get("status"),  # "Buy" / "Sell"
                "volume": tx.get("value"),
                "price": tx.get("price"),
            })
    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    db_set_kv(key, df.to_json(), _wib_now().isoformat())
    return df, None


def insider_verdict(insider_df: pd.DataFrame, days: int = 90):
    """Ringkasan: dalam `days` hari terakhir, insider net beli atau net jual?"""
    if insider_df is None or insider_df.empty or "date" not in insider_df.columns:
        return None
    # Kolom "date" bisa berupa python `date` (jalur fresh-fetch, sudah
    # di-.dt.date) ATAU Timestamp/datetime64 (jalur cache, hasil
    # pd.read_json otomatis parse tanggal) -- membandingkan dtype campuran
    # ini langsung raises TypeError di pandas versi baru. Normalisasi ke
    # Timestamp dulu di kedua sisi supaya perbandingan selalu valid.
    dates = pd.to_datetime(insider_df["date"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    recent = insider_df[dates >= cutoff].copy()
    if recent.empty:
        return {"net_transactions": 0, "verdict": "TIDAK ADA TRANSAKSI", "count": 0}

    recent["volume"] = pd.to_numeric(recent["volume"], errors="coerce").fillna(0)
    is_buy = recent["action"].astype(str).str.lower().str.contains("buy|beli")
    net = float(recent.loc[is_buy, "volume"].sum() - recent.loc[~is_buy, "volume"].sum())
    verdict = "NET BELI (akumulasi insider)" if net > 0 else (
        "NET JUAL (distribusi insider)" if net < 0 else "SEIMBANG"
    )
    return {"net_transactions": net, "verdict": verdict, "count": len(recent)}


# --------------------------------------------------------------------------
# LAPIS A LANJUTAN: riwayat bulanan pemegang >1% + graf relasi
# LAPIS B+C: rekonsiliasi insider <-> broker pasar negosiasi (NG)
# Konsep diambil dari analisa-kepemilikan.skill, diringkas untuk konteks
# app interaktif (bukan laporan HTML statis): tanpa penelusuran rantai
# korporasi berjenjang (endpoint tidak menyediakan edge entity->entity) dan
# tanpa scan harian menyeluruh pasar NG di luar tanggal insider (itu akan
# butuh 1 panggilan API per hari dalam window -- terlalu berat untuk app
# yang dipanggil live oleh banyak user, beda dengan skill laporan offline
# yang dijalankan sekali per permintaan oleh agent).
# --------------------------------------------------------------------------

def fetch_shareholder_detail(code: str):
    """
    Riwayat BULANAN pemegang saham >1% (Lapis A time series, beda dari
    fetch_shareholders() yang cuma snapshot terbaru).
    Endpoint: GET /analysis/shareholder-detail/{code} -- verified api-1.json.
    """
    key = f"{code}|shareholder-detail"
    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < SHAREHOLDER_CACHE_TTL_HOURS:
                cached_df = pd.read_json(io.StringIO(payload))
                if "date" in cached_df.columns:
                    cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.date
                return cached_df, None
        except (ValueError, TypeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    url = f"{BASE_URL}/analysis/shareholder-detail/{code}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    except requests.RequestException as e:
        cached = pd.read_json(io.StringIO(payload)) if payload else None
        return cached, f"Request error: {e}"
    if not r.ok:
        cached = pd.read_json(io.StringIO(payload)) if payload else None
        hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
        return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
    data = r.json()
    if not data:
        return None, None
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("percent", "val"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["name", "date"]).reset_index(drop=True)
    db_set_kv(key, df.to_json(), _wib_now().isoformat())
    return df, None


def shareholder_monthly_changes(detail_df: pd.DataFrame, min_change_pct: float = 0.1):
    """
    Dari riwayat bulanan per pemegang, hitung perubahan persentase
    bulan-ke-bulan per nama, filter yang berubah >= min_change_pct poin
    persen -- ini jadi "peristiwa Lapis A" untuk timeline.
    """
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()
    rows = []
    for name, grp in detail_df.groupby("name"):
        grp = grp.sort_values("date")
        prev_pct = None
        for _, row in grp.iterrows():
            if prev_pct is not None and pd.notna(row.get("percent")):
                delta = row["percent"] - prev_pct
                if abs(delta) >= min_change_pct:
                    rows.append({
                        "date": row["date"], "name": name,
                        "percent": row["percent"], "change": delta,
                    })
            prev_pct = row.get("percent")
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def fetch_shareholder_relation(code: str, depth: int = 3, min_percentage: float = 1, max_nodes: int = 80, neighbors: int = 20):
    """
    Graf relasi kepemilikan (irisan kepemilikan antar entitas/saham, BUKAN
    rantai korporasi berjenjang -- endpoint ini tidak menyediakan edge
    entity->entity, hanya entity<->stock berdasarkan siapa memegang saham
    yang sama).
    Endpoint: GET /analysis/shareholder/relation -- verified api-1.json.
    """
    key = f"{code}|relation|{depth}|{min_percentage}|{max_nodes}|{neighbors}"
    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < SHAREHOLDER_CACHE_TTL_HOURS:
                return json.loads(payload), None
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    url = f"{BASE_URL}/analysis/shareholder/relation"
    try:
        r = requests.get(
            url,
            params={"code": code, "depth": depth, "min_percentage": min_percentage,
                    "max_nodes": max_nodes, "neighbors": neighbors},
            headers={"Authorization": f"Bearer {api_key}"}, timeout=45,
        )
    except requests.RequestException as e:
        cached = json.loads(payload) if payload else None
        return cached, f"Request error: {e}"
    if not r.ok:
        cached = json.loads(payload) if payload else None
        hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
        return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
    data = r.json()
    if not data or not data.get("nodes"):
        return None, None
    db_set_kv(key, json.dumps(data), _wib_now().isoformat())
    return data, None


def fetch_ng_daily_summary(code: str, day: date):
    """
    Broker summary pasar NEGOSIASI (NG) untuk SATU hari spesifik -- dipakai
    untuk merekonsiliasi transaksi insider ke broker pelaksana. Beda dari
    fetch_broker_summary() yang agregat multi-hari & market RG.
    """
    day_str = day.isoformat()
    key = f"{code}|ng|{day_str}"
    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        return pd.read_json(io.StringIO(payload)) if payload != "[]" else pd.DataFrame()

    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/summary/stock/{code}",
            params={"from": day_str, "to": day_str, "investor": "all", "market": "NG"},
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30,
        )
    except requests.RequestException:
        return pd.DataFrame()
    if not r.ok:
        return pd.DataFrame()
    data = r.json()
    db_set_kv(key, json.dumps(data) if data else "[]", _wib_now().isoformat())
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    for col in ["buy_volume", "sell_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def reconcile_insider_to_broker(code: str, insider_df: pd.DataFrame, tolerance: float = 0.05):
    """
    Untuk tiap transaksi insider, cari broker di pasar NG pada tanggal yang
    sama dengan volume beli/jual paling mendekati volume transaksi insider.
    Match dianggap 'COCOK PERSIS' kalau selisih <1 lembar, 'COCOK PARSIAL'
    kalau dalam toleransi (`tolerance`), selain itu 'TIDAK DITEMUKAN'
    (transaksi kemungkinan di luar bursa, atau di market RG bukan NG).
    Hanya menyisir tanggal-tanggal yang PUNYA laporan insider -- TIDAK
    menyisir seluruh pasar NG tanpa insider (lihat catatan modul di atas).
    """
    if insider_df is None or insider_df.empty:
        return pd.DataFrame()

    results = []
    for _, tx in insider_df.iterrows():
        ng = fetch_ng_daily_summary(code, tx["date"])
        is_buy = str(tx.get("action", "")).lower().startswith(("buy", "beli"))
        vol_col = "buy_volume" if is_buy else "sell_volume"
        broker_col = "code" if "code" in ng.columns else None
        match_broker, match_status = None, "TIDAK DITEMUKAN"

        if not ng.empty and vol_col in ng.columns and broker_col:
            target = float(tx.get("volume") or 0)
            if target > 0:
                ng = ng.copy()
                ng["_diff"] = (ng[vol_col] - target).abs()
                best = ng.sort_values("_diff").iloc[0]
                rel_diff = best["_diff"] / target if target else 1.0
                if rel_diff < 1e-6:
                    match_broker, match_status = str(best[broker_col]), "COCOK PERSIS"
                elif rel_diff <= tolerance:
                    match_broker, match_status = str(best[broker_col]), "COCOK PARSIAL"

        results.append({
            "date": tx["date"], "name": tx.get("name"), "action": tx.get("action"),
            "volume": tx.get("volume"), "price": tx.get("price"),
            "broker": match_broker, "status": match_status,
        })
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# PROFIL PERUSAHAAN & POST KOMUNITAS ("BERITA")
# --------------------------------------------------------------------------
COMPANY_INFO_CACHE_TTL_HOURS = 24 * 7  # profil perusahaan jarang berubah


def fetch_company_info(code: str):
    """
    Profil perusahaan: nama, sektor/subsektor, industri, alamat, website,
    tanggal listing, direksi/komisaris, notasi khusus (kalau ada).
    Endpoint: GET /analysis/information/{code} -- verified api-1.json.
    """
    key = f"{code}|information"
    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < COMPANY_INFO_CACHE_TTL_HOURS:
                return json.loads(payload), None
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    url = f"{BASE_URL}/analysis/information/{code}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    except requests.RequestException as e:
        cached = json.loads(payload) if payload else None
        return cached, f"Request error: {e}"
    if not r.ok:
        cached = json.loads(payload) if payload else None
        hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
        return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
    data = r.json()
    if not data:
        return None, None
    db_set_kv(key, json.dumps(data), _wib_now().isoformat())
    return data, None


def fetch_stock_posts(code: str, limit: int = 10):
    """
    Postingan komunitas InvezGo terkait saham ini ("space" per-stock) --
    CATATAN: ini bukan berita dari media/wire resmi (InvezGo tidak
    menyediakan endpoint news murni), melainkan diskusi/postingan user di
    platform InvezGo. Perlakukan sebagai sentimen komunitas, bukan berita
    tervalidasi.
    Endpoint: GET /posts/space/{code}.
    """
    key = f"{code}|posts|{limit}"
    payload, fetched_at = db_get_kv(key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < 1:  # postingan bisa sering berubah, cache pendek
                return json.loads(payload), None
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    url = f"{BASE_URL}/posts/space/{code}"
    try:
        r = requests.get(
            url, params={"page": 1, "limit": limit},
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30,
        )
    except requests.RequestException as e:
        cached = json.loads(payload) if payload else None
        return cached, f"Request error: {e}"
    if not r.ok:
        cached = json.loads(payload) if payload else None
        hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
        return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
    data = r.json()
    items = data.get("data", data) if isinstance(data, dict) else data
    if not items:
        return None, None
    db_set_kv(key, json.dumps(data), _wib_now().isoformat())
    return items, None


# --------------------------------------------------------------------------
# FUNDAMENTAL: laporan keuangan (IS/BS/CF) + key statistics/valuasi
# --------------------------------------------------------------------------
FINANCIAL_CACHE_TTL_HOURS = 24


def _fetch_financial_rows(url: str, code: str, params: dict, cache_key: str):
    """Helper generik: GET endpoint berformat {rows, columns}, cache di kv_cache."""
    payload, fetched_at = db_get_kv(cache_key)
    if payload and fetched_at:
        try:
            age_hours = (_wib_now() - _dt.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_hours < FINANCIAL_CACHE_TTL_HOURS:
                return json.loads(payload), None
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    api_key = get_secret("INVEZGO_API_KEY")
    try:
        r = requests.get(url, params=params, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    except requests.RequestException as e:
        cached = json.loads(payload) if payload else None
        return cached, f"Request error: {e}"
    if r.status_code == 204:
        return None, None
    if not r.ok:
        cached = json.loads(payload) if payload else None
        hint = _STATUS_HINT.get(r.status_code, f"{r.status_code}")
        return cached, f"GET {url} -> {hint} | body: {r.text[:300]}"
    try:
        data = r.json()
    except ValueError:
        return None, f"GET {url} -> 200 OK tapi bukan JSON valid."
    if not data or not data.get("rows"):
        return None, None
    db_set_kv(cache_key, json.dumps(data), _wib_now().isoformat())
    return data, None


def fetch_financial_statement(code: str, statement: str = "IS", period_type: str = "Q", limit: int = 8):
    """
    Laporan keuangan mentah. statement: IS (laba rugi) / BS (neraca) / CF (arus kas).
    Endpoint: GET /analysis/financial-statement/{code} -- verified api-1.json.
    """
    url = f"{BASE_URL}/analysis/financial-statement/{code}"
    key = f"{code}|finstat|{statement}|{period_type}|{limit}"
    return _fetch_financial_rows(url, code, {"statement": statement, "type": period_type, "limit": limit}, key)


def fetch_keystat(code: str, period_type: str = "Q", limit: int = 8):
    """
    Key statistics / rasio valuasi (PER, PBV, ROE, dll).
    Endpoint: GET /analysis/keystat/{code} -- verified api-1.json.
    CATATAN dari deskripsi resmi API: "Endpoint ini sedang mengalami proses
    aktualisasi untuk perhitungan yang lebih akurat. Data yang disajikan
    saat ini mungkin kurang tepat" -- jadi tampilkan apa adanya dengan
    disclaimer, jangan dianggap presisi.
    """
    url = f"{BASE_URL}/analysis/keystat/{code}"
    key = f"{code}|keystat|{period_type}|{limit}"
    return _fetch_financial_rows(url, code, {"type": period_type, "limit": limit}, key)


def rows_to_pivot(data: dict) -> pd.DataFrame:
    """Ubah {rows:[{name, values:[{col, amount}]}], columns:[{label}]} jadi tabel lebar name x periode."""
    if not data or not data.get("rows"):
        return pd.DataFrame()
    col_order = [c["label"] for c in data.get("columns", [])]
    records = {}
    for row in data["rows"]:
        vals = {v["col"]: v["amount"] for v in row.get("values", [])}
        records[row["name"]] = vals
    df = pd.DataFrame.from_dict(records, orient="index")
    cols_present = [c for c in col_order if c in df.columns]
    return df[cols_present] if cols_present else df


def find_metric_row(pivot_df: pd.DataFrame, keywords: list):
    """Cari baris pertama yang namanya mengandung salah satu keyword (case-insensitive)."""
    if pivot_df.empty:
        return None
    for idx in pivot_df.index:
        low = str(idx).lower()
        if any(kw in low for kw in keywords):
            return idx
    return None


def find_metric_row_priority(pivot_df: pd.DataFrame, keyword_groups: list):
    """
    Coba tiap grup keyword berurutan, return match pertama dari grup
    ter-spesifik. Istilah "laba bersih" di laporan keuangan IDX (berbasis
    taksonomi XBRL) bervariasi antar-emiten -- "laba (rugi) periode
    berjalan", "laba tahun berjalan", dll -- jadi satu keyword generik
    "laba" saja gampang salah tangkap ("laba bruto"/"laba usaha").
    """
    for group in keyword_groups:
        match = find_metric_row(pivot_df, group)
        if match:
            return match
    return None


def id_number(x, decimals: int = 0) -> str:
    """
    Format angka gaya Indonesia: titik pemisah ribuan, koma pemisah desimal
    (mis. 1.234.567 atau 1.234,56). decimals=0 cocok untuk nilai laporan
    keuangan (rupiah), decimals>0 untuk rasio (PER, PBV, dll).
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    s = f"{x:,.{decimals}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return s


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


def top_accumulator_trend(code: str):
    """
    Identifikasi broker yang net beli terbesar dalam 30 hari, lalu cek
    apakah dia MASIH aktif belanja di 10 hari terakhir (bukan cuma net
    kumulatif lama yang sudah berhenti) -- konsep "diparkir"/"dijaga" ala
    bandarmologi: broker yang terus menambah net beli = masih mengumpulkan,
    broker yang net-nya besar tapi diam/berbalik di window terbaru =
    akumulasi sudah berhenti atau mulai dilepas.
    """
    sum30 = fetch_broker_summary(code, 30)
    sum10 = fetch_broker_summary(code, 10)
    if sum30 is None or sum30.empty:
        return None

    df30 = sum30.copy()
    df30["net"] = df30["buy_volume"] - df30["sell_volume"]
    top = df30.sort_values("net", ascending=False).iloc[0]
    broker, net30 = str(top["code"]), float(top["net"])
    if net30 <= 0:
        return None  # tidak ada broker yang net akumulasi di 30 hari

    net10 = 0.0
    if sum10 is not None and not sum10.empty:
        df10 = sum10.copy()
        df10["net"] = df10["buy_volume"] - df10["sell_volume"]
        match = df10[df10["code"] == broker]
        if not match.empty:
            net10 = float(match["net"].iloc[0])

    # Proporsi net 10 hari terakhir terhadap net 30 hari -- kalau broker
    # yang sama masih menyumbang porsi besar di window terbaru, dia masih
    # aktif; kalau porsinya kecil/negatif, akumulasi sudah melambat/berhenti.
    share_recent = (net10 / net30) if net30 > 0 else 0.0
    still_active = net10 > 0 and share_recent >= 0.2

    return {
        "broker": broker,
        "net_30d": net30,
        "net_10d": net10,
        "share_recent": share_recent,
        "still_active": still_active,
        "verdict": (
            "MASIH AKTIF MENGUMPULKAN" if still_active
            else "AKUMULASI MELAMBAT / BERHENTI"
        ),
    }


# --------------------------------------------------------------------------
# GROQ AI
# --------------------------------------------------------------------------
def _groq_model_order() -> list:
    """Urutan model dicoba: override secrets dulu, lalu model terakhir yang
    terbukti jalan di sesi ini, lalu default + fallback bawaan (tanpa duplikat)."""
    order = []
    try:
        override = st.secrets.get("GROQ_MODEL")
    except (FileNotFoundError, KeyError):
        override = None
    if override:
        order.append(override)
    last_working = st.session_state.get("groq_last_working_model")
    if last_working:
        order.append(last_working)
    order += [GROQ_MODEL_DEFAULT] + GROQ_FALLBACK_MODELS
    seen, dedup = set(), []
    for m in order:
        if m not in seen:
            seen.add(m)
            dedup.append(m)
    return dedup


def _groq_call(key: str, model: str, prompt: str):
    """Satu percobaan panggilan Groq. Return (status, text_or_content, retry_after)."""
    payload = {
        "model": model,
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
    }
    if model in GROQ_REASONING_MODELS:
        payload["reasoning_effort"] = "low"

    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=90,
        allow_redirects=False,
    )
    if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
        return "redirect", r.headers.get("Location", "?"), None
    if r.status_code == 401:
        return "auth", None, None
    if r.status_code == 404:
        return "not_found", r.text, None
    if r.status_code == 429:
        body = r.text.lower()
        # TPD (token per hari) baru reset dalam jam, retry pendek tidak
        # membantu -> pindah model lain segera. TPM (token per menit) itu
        # rolling/leaky bucket, pulih dalam hitungan detik -> retry singkat.
        kind = "daily_quota" if ("tokens per day" in body or "(tpd)" in body) else "rate_limit"
        return kind, r.text, _extract_retry_wait(r.text)
    if not r.ok:
        return "error", f"{r.status_code}: {r.text}", None
    return "ok", r.json()["choices"][0]["message"]["content"], None


def _extract_retry_wait(text: str) -> str | None:
    """Cari perkiraan waktu tunggu dari pesan Groq, mis. 'try again in 8m30.624s'."""
    match = re.search(r"try again in\s+(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?", text, re.IGNORECASE)
    if not match or not (match.group(1) or match.group(2)):
        return None
    minutes = round(float(match.group(1))) if match.group(1) else 0
    seconds = round(float(match.group(2))) if match.group(2) else 0
    parts = []
    if minutes:
        parts.append(f"{minutes} menit")
    if seconds:
        parts.append(f"{seconds} detik")
    return " ".join(parts) or "beberapa detik"


@st.cache_data(ttl=1800, show_spinner="🤖 Groq sedang menganalisis…")
def groq_chat(prompt: str) -> str:
    key = get_secret("GROQ_API_KEY")
    tried, not_found = [], []

    for model in _groq_model_order():
        tried.append(model)

        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            status, payload, wait_hint = _groq_call(key, model, prompt)

            if status == "ok":
                st.session_state["groq_last_working_model"] = model
                return payload
            if status == "auth":
                return "❌ GROQ_API_KEY tidak valid."
            if status == "redirect":
                return (
                    f"❌ Groq redirect ke '{payload}' — permintaan POST berubah jadi "
                    "GET dan gagal. Cek GROQ_URL sudah benar-benar "
                    "'https://api.groq.com/openai/v1/chat/completions'."
                )
            if status == "not_found":
                not_found.append(model)
                break  # jangan retry model yang tidak ada, langsung ke kandidat berikutnya
            if status == "daily_quota":
                break  # tidak akan pulih dalam hitungan detik, pindah model lain segera
            if status == "rate_limit":
                if attempt < RATE_LIMIT_MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_BASE_DELAY_SECONDS * (attempt + 1))
                    continue
                break  # retry TPM habis di model ini, coba kandidat berikutnya
            break  # error lain -> tetap coba kandidat berikutnya, jangan retry model sama

    return (
        f"❌ Semua {len(tried)} model Groq yang dicoba gagal ({', '.join(tried)}). "
        f"Model tidak ditemukan: {', '.join(not_found) or '-'}.\n\n"
        "Cek daftar model aktif untuk akunmu di "
        "https://console.groq.com/docs/models, lalu set `GROQ_MODEL` di "
        "Streamlit Secrets ke salah satu yang tersedia."
    )


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
                "keterangan": f"Rebound bullish tepat di level fib {lvl*100:.1f}%.",
            }, retr, ext

    # ---- STRONG PULLBACK: tembus 1 level fib -> tunggu level bawahnya ----
    # PENTING: pakai level PALING DALAM yang tertembus (bukan yang pertama
    # ditemukan). SEQ terurut naik persentase = turun harga; kalau harga
    # jatuh menembus beberapa level sekaligus (mis. break 61.8% DAN 78.6%
    # di hari yang sama), berhenti di level pertama (61.8%) salah: level
    # "bawahnya" (nxt) yang dihitung dari situ (78.6%) bisa jadi MASIH DI
    # ATAS harga close saat ini -- artinya entry > TP1, dead-on-arrival
    # loss kalau langsung dieksekusi. Level 78.6% (lebih dalam) baru
    # menghasilkan nxt = swing low, yang selalu <= close.
    broken_idx = None
    for i, lvl in enumerate(SEQ):
        if prev_close > retr[lvl] and close < retr[lvl]:
            broken_idx = i  # overwrite terus -> tersisa index terdalam

    if broken_idx is not None:
        lvl = SEQ[broken_idx]
        nxt = retr[SEQ[broken_idx + 1]] if broken_idx + 1 < len(SEQ) else low_p
        if nxt >= close:
            # Jaga-jaga: seharusnya tidak mungkin lagi setelah fix di atas,
            # tapi kalau tetap terjadi (data swing tidak wajar), jangan
            # kasih setup entry>TP yang pasti rugi -- treat sebagai skip.
            return {
                "signal": "SKIP (level tidak konsisten)",
                "fib_level": f"{lvl*100:.1f}% BREAK",
                "entry": None, "stop_loss": None, "risk_pct": 0,
                "tp1": None, "tp2": None, "tp3": None,
                "keterangan": "Level fib berikutnya tidak berada di bawah harga saat ini — setup dilewati.",
            }, retr, ext
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
                "keterangan": f"Harga menembus level fib {lvl*100:.1f}%, tunggu di level bawahnya.",
            }, retr, ext
        return {
            "signal": "SKIP (risiko kebesar)",
            "fib_level": f"{lvl*100:.1f}% BREAK",
            "entry": None, "stop_loss": None, "risk_pct": abs(risk_pct),
            "tp1": None, "tp2": None, "tp3": None,
            "keterangan": (
                f"Menembus level fib {lvl*100:.1f}%, tapi risiko ke level "
                f"berikutnya {abs(risk_pct):.1f}% > batas {max_risk_pct}% yang kamu tetapkan."
            ),
        }, retr, ext

    nearest_lvl = min(SEQ, key=lambda lvl: abs(close - retr[lvl]))
    dist_pct = (close - retr[nearest_lvl]) / retr[nearest_lvl] * 100
    return {
        "signal": "TIDAK ADA SETUP",
        "fib_level": "-",
        "entry": None, "stop_loss": None, "risk_pct": 0,
        "tp1": None, "tp2": None, "tp3": None,
        "keterangan": (
            f"Harga belum menyentuh/menembus level fib manapun (level terdekat "
            f"{nearest_lvl*100:.1f}%, jarak {dist_pct:+.1f}%)."
        ),
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
    elif not anchor:
        row.update({"signal": "-", "fib_level": "-", "entry": None,
                    "stop_loss": None, "risk_pct": 0,
                    "tp1": None, "tp2": None, "tp3": None,
                    "keterangan": "Swing high/low belum cukup — perpanjang lookback."})
    else:
        row.update({"signal": "-", "fib_level": "-", "entry": None,
                    "stop_loss": None, "risk_pct": 0,
                    "tp1": None, "tp2": None, "tp3": None,
                    "keterangan": (
                        f"Struktur {structure}, bukan UPTREND — pullback fib "
                        "hanya dicek saat market structure UPTREND."
                    )})
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


def plot_volume_30d(df, window: int = 30):
    """Bar volume N hari terakhir + garis rata-rata, hijau kalau close naik dari hari sebelumnya."""
    seg = df.tail(window).reset_index(drop=True)
    avg_vol = float(seg["volume"].mean())
    colors = [
        "green" if seg["close"].iloc[i] >= seg["close"].iloc[i - 1] else "red"
        for i in range(len(seg))
    ]
    colors[0] = "gray"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(seg["date"], seg["volume"], color=colors, alpha=0.75)
    ax.axhline(avg_vol, color="#333", ls="--", lw=1, label=f"Rata-rata {window}h")
    ax.set_title(f"Volume {window} Hari Terakhir")
    ax.set_ylabel("Volume (lembar)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return fig


def plot_shareholder_relation(relation_data: dict, root_code: str):
    """Graf relasi kepemilikan (node-link) pakai networkx -- root disorot merah."""
    g = nx.Graph()
    for n in relation_data.get("nodes", []):
        g.add_node(n["id"], label=n.get("label", n["id"]), root=n.get("root", False))
    for e in relation_data.get("edges", []):
        if e["source"] in g.nodes and e["target"] in g.nodes:
            g.add_edge(e["source"], e["target"], weight=e.get("percentage", 1))

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(g, k=0.6, seed=42)
    node_colors = ["#e74c3c" if g.nodes[n].get("root") else "#3498db" for n in g.nodes]
    node_sizes = [700 if g.nodes[n].get("root") else 350 for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, edge_color="#999")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
    labels = {n: g.nodes[n]["label"] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title(f"Graf Relasi Kepemilikan — {root_code} (merah = root)")
    ax.axis("off")
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
    if st.button("🔍 Analisis", type="primary"):
        st.session_state["single_ticker"] = ticker

    if "single_ticker" not in st.session_state:
        st.info("Masukkan kode saham lalu klik **Analisis**.")
        st.stop()

    ticker = st.session_state["single_ticker"]

    # --- Profil perusahaan ---
    info, info_err = fetch_company_info(ticker)
    if info:
        i1, i2 = st.columns([1, 4])
        with i1:
            if info.get("logo"):
                st.image(info["logo"], width=80)
        with i2:
            st.markdown(f"### {info.get('name', ticker)} ({ticker})")
            st.caption(
                f"{info.get('sector', '-')} · {info.get('subsector', '-')} · "
                f"{info.get('industry', '-')} — {info.get('activity', '-')}"
            )
        notations = info.get("notation") or []
        if notations:
            for n in notations:
                st.warning(f"⚠️ Notasi khusus **{n.get('notation')}**: {n.get('description')}")
        with st.expander("ℹ️ Detail Perusahaan"):
            d1, d2 = st.columns(2)
            d1.write(f"**Alamat:** {info.get('address', '-')}")
            d1.write(f"**Website:** {info.get('website', '-')}")
            d1.write(f"**Tanggal Listing:** {info.get('listing_date', '-')}")
            d1.write(f"**Papan:** {info.get('board', '-')}")
            if info.get("category"):
                d2.write(f"**Kategori:** {', '.join(info['category'])}")
            if info.get("director"):
                d2.write("**Direksi:** " + ", ".join(p["name"] for p in info["director"]))
            if info.get("commissioner"):
                d2.write("**Komisaris:** " + ", ".join(p["name"] for p in info["commissioner"]))
    elif info_err:
        st.error(f"❌ Profil perusahaan gagal diambil: {info_err}")

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
    sig, _, _ = analyze_signal(df, low_p, high_p, max_risk)

    tab_teknikal, tab_bandarmologi, tab_fundamental = st.tabs(
        ["📐 Teknikal", "🕵️ Bandarmologi", "📊 Fundamental"]
    )

    # ======================================================================
    # TAB TEKNIKAL — fibonacci, sinyal, chart, volume
    # ======================================================================
    with tab_teknikal:
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

        st.subheader("📶 Volume 30 Hari Terakhir")
        vol_seg = df.tail(30)
        vol_avg30 = float(vol_seg["volume"].mean())
        vol_today = float(df["volume"].iloc[-1])
        avg_recent10 = float(df["volume"].tail(10).mean())
        avg_prior20 = float(df["volume"].tail(30).head(20).mean()) if len(df) >= 30 else avg_recent10
        vtrend = "MENINGKAT" if avg_recent10 > avg_prior20 * 1.1 else (
            "MENURUN" if avg_recent10 < avg_prior20 * 0.9 else "STABIL"
        )
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Volume Hari Ini", f"{vol_today:,.0f}")
        v2.metric("Rata-rata 30 Hari", f"{vol_avg30:,.0f}")
        v3.metric("Vol Hari Ini vs Rata-rata", f"{vol_today / vol_avg30:.2f}x" if vol_avg30 else "-")
        v4.metric("Tren 10h Terakhir", vtrend)
        st.pyplot(plot_volume_30d(df), use_container_width=True)

    # ======================================================================
    # TAB BANDARMOLOGI — BDM, broker, kepemilikan/insider, sentimen komunitas
    # ======================================================================
    with tab_bandarmologi:
        bdm = fetch_bdm(ticker, lookback)
        if bdm is not None and not bdm.empty:
            st.subheader("🕵️ Aktivitas Bandar (BDM)")
            bdm_info = bandar_classify(df, bdm)
            if bdm_info:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Stage", bdm_info["stage"])
                b2.metric("Hari Akumulasi", f"{bdm_info['acc_pct']:.0f}%")
                b3.metric("Volume vs MA20", f"{bdm_info['vol_ratio']:.2f}x")
                b4.metric("Lebar Range 40h", f"{bdm_info['range_pct']:.1f}%")
            st.pyplot(plot_bandar(df, bdm), use_container_width=True)

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

        st.subheader("🧲 Bandar yang Sedang Mengumpulkan")
        trend_info = top_accumulator_trend(ticker)
        if trend_info:
            active = trend_info["still_active"]
            t1, t2, t3 = st.columns(3)
            t1.metric("Broker Akumulator Terbesar (30h)", trend_info["broker"],
                       delta=f"net {trend_info['net_30d']:,.0f}")
            t2.metric("Net 10 Hari Terakhir", f"{trend_info['net_10d']:,.0f}",
                       delta=f"{trend_info['share_recent']*100:.0f}% dari net 30h")
            t3.metric("Status", trend_info["verdict"],
                       delta="AKTIF" if active else "MELAMBAT", delta_color="normal" if active else "inverse")
            st.caption(
                "Dihitung dari net beli–jual (bukan net volume kumulatif harian) per broker "
                "di endpoint summary. 'MASIH AKTIF' berarti broker akumulator terbesar 30 hari "
                "masih menyumbang porsi besar (≥20%) dari net-nya di 10 hari terakhir — bukan "
                "cuma sisa net lama yang sudah berhenti dibeli."
            )
        else:
            st.caption("Tidak ada broker dengan net akumulasi positif dalam 30 hari terakhir, atau data broker tidak cukup.")

        st.subheader("🏛️ Kepemilikan (Shareholder & Insider)")
        st.caption(
            "Porting dari konsep skill analisa-kepemilikan: komposisi terbaru, riwayat "
            "bulanan, graf relasi, rekonsiliasi insider↔broker pasar negosiasi (NG), "
            "dirangkai jadi timeline. TIDAK menelusuri rantai korporasi berjenjang "
            "(endpoint relation cuma punya irisan kepemilikan, bukan edge entity→entity) "
            "dan TIDAK menyisir seluruh pasar NG tanpa laporan insider (butuh 1 panggilan "
            "API per hari dalam window — terlalu berat untuk app live)."
        )
        sh_col, ins_col = st.columns(2)

        with sh_col:
            st.markdown("**Komposisi Pemegang Saham (>1%, snapshot terbaru)**")
            shareholders, sh_err = fetch_shareholders(ticker)
            if shareholders is not None and not shareholders.empty:
                show_cols = [c for c in ["name", "percentage", "badge"] if c in shareholders.columns]
                st.dataframe(
                    shareholders[show_cols] if show_cols else shareholders,
                    use_container_width=True, hide_index=True, height=280,
                )
            elif sh_err:
                st.error(f"❌ Shareholder gagal diambil: {sh_err}")
            else:
                st.caption("Tidak ada data shareholder untuk ticker ini.")

        with ins_col:
            st.markdown(f"**Transaksi Insider ({INSIDER_LOOKBACK_MONTHS} bulan terakhir)**")
            insider_df, ins_err = fetch_insider_transactions(ticker)
            if insider_df is not None and not insider_df.empty:
                verdict = insider_verdict(insider_df)
                if verdict and verdict["count"] > 0:
                    st.metric(
                        "Verdict 90 hari terakhir", verdict["verdict"],
                        delta=f"{verdict['count']} transaksi",
                    )
                show_cols = [c for c in ["date", "name", "badge", "action", "volume", "price"] if c in insider_df.columns]
                st.dataframe(
                    insider_df[show_cols] if show_cols else insider_df,
                    use_container_width=True, hide_index=True, height=230,
                )
            elif ins_err:
                st.error(f"❌ Insider gagal diambil: {ins_err}")
            else:
                st.caption("Tidak ada laporan insider dalam periode ini.")

        with st.expander("🕸️ Graf Relasi Kepemilikan"):
            st.caption(
                "Irisan kepemilikan (siapa memegang saham yang sama) — BUKAN rantai "
                "korporasi berjenjang. Merah = saham ini, biru = entitas/saham terkait."
            )
            relation_data, rel_err = fetch_shareholder_relation(ticker)
            if relation_data:
                st.pyplot(plot_shareholder_relation(relation_data, ticker), use_container_width=True)
            elif rel_err:
                st.error(f"❌ Graf relasi gagal diambil: {rel_err}")
            else:
                st.caption("Tidak ada relasi kepemilikan >1% yang ditemukan untuk ticker ini.")

        with st.expander("📜 Riwayat Bulanan Pemegang >1% (perubahan signifikan)"):
            detail_df, det_err = fetch_shareholder_detail(ticker)
            if det_err:
                st.error(f"❌ Riwayat shareholder gagal diambil: {det_err}")
            elif detail_df is not None and not detail_df.empty:
                changes = shareholder_monthly_changes(detail_df)
                if not changes.empty:
                    st.dataframe(changes, use_container_width=True, hide_index=True, height=250)
                else:
                    st.caption("Tidak ada perubahan kepemilikan signifikan (≥0.1 poin persen) antar-bulan.")
            else:
                st.caption("Tidak ada riwayat bulanan untuk ticker ini.")

        with st.expander("🔍 Rekonsiliasi Insider ↔ Broker Pasar Negosiasi (NG)"):
            st.caption(
                "Untuk tiap transaksi insider, dicari broker NG di tanggal yang sama "
                "dengan volume paling mendekati. Kalau tidak ketemu, transaksinya bisa "
                "di market RG biasa (bukan NG) atau di luar bursa."
            )
            if insider_df is not None and not insider_df.empty:
                recon = reconcile_insider_to_broker(ticker, insider_df)
                st.dataframe(recon, use_container_width=True, hide_index=True, height=250)
            else:
                st.caption("Tidak ada transaksi insider untuk direkonsiliasi.")

        with st.expander("🗓️ Timeline Kepemilikan Gabungan (Lapis A + B)"):
            events = []
            if det_err is None and detail_df is not None and not detail_df.empty:
                changes = shareholder_monthly_changes(detail_df)
                for _, row in changes.iterrows():
                    events.append({
                        "date": row["date"], "lapis": "A — Struktur",
                        "peristiwa": f"{row['name']}: {row['change']:+.2f}pp → {row['percent']:.2f}%",
                    })
            if insider_df is not None and not insider_df.empty:
                for _, row in insider_df.iterrows():
                    events.append({
                        "date": row["date"], "lapis": "B — Insider",
                        "peristiwa": f"{row.get('name')} {row.get('action')} {row.get('volume'):,.0f} lembar @ {row.get('price')}",
                    })
            if events:
                timeline = pd.DataFrame(events).sort_values("date", ascending=False).reset_index(drop=True)
                st.dataframe(timeline, use_container_width=True, hide_index=True, height=300)
            else:
                st.caption("Tidak ada peristiwa kepemilikan (Lapis A/B) untuk dirangkai jadi timeline.")

        st.subheader("📰 Berita & Diskusi Terkini")
        st.caption(
            "InvezGo tidak menyediakan endpoint berita/wire resmi — ini postingan "
            "komunitas dari platform InvezGo terkait saham ini. Perlakukan sebagai "
            "sentimen komunitas, BUKAN berita tervalidasi dari media."
        )
        posts, posts_err = fetch_stock_posts(ticker)
        if posts:
            for p in posts:
                if not isinstance(p, dict):
                    # Skema respons /posts/space/{code} belum terverifikasi ke
                    # dokumentasi -- kalau ternyata bukan objek per field
                    # seperti diasumsikan, tampilkan apa adanya daripada crash.
                    with st.container(border=True):
                        st.write(p)
                    continue
                title = p.get("title") or str(p.get("content", ""))[:80] or "(tanpa judul)"
                body = p.get("content") or p.get("body") or ""
                author = p.get("author", {}).get("name") if isinstance(p.get("author"), dict) else p.get("author")
                created = p.get("created_at") or p.get("date") or p.get("createdAt")
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if body and body != title:
                        body = str(body)
                        st.write(body[:400] + ("..." if len(body) > 400 else ""))
                    meta = " · ".join(str(x) for x in [author, created] if x)
                    if meta:
                        st.caption(meta)
        elif posts_err:
            st.error(f"❌ Postingan gagal diambil: {posts_err}")
        else:
            st.caption("Tidak ada postingan komunitas untuk saham ini.")

    # ======================================================================
    # TAB FUNDAMENTAL — laporan keuangan & key statistics/valuasi
    # ======================================================================
    with tab_fundamental:
        f1, f2 = st.columns(2)
        statement_label = f1.selectbox(
            "Laporan", ["Laba Rugi", "Neraca", "Arus Kas"], key="fund_statement"
        )
        statement_code = {"Laba Rugi": "IS", "Neraca": "BS", "Arus Kas": "CF"}[statement_label]
        period_label = f2.selectbox("Periode", ["Kuartalan", "Tahunan"], key="fund_period")
        period_code = "Q" if period_label == "Kuartalan" else "FY"

        fin_data, fin_err = fetch_financial_statement(ticker, statement_code, period_code)
        if fin_err:
            st.error(f"❌ Laporan keuangan gagal diambil: {fin_err}")
        elif fin_data:
            pivot = rows_to_pivot(fin_data)
            if statement_code == "IS":
                rev_row = find_metric_row(pivot, ["pendapatan", "penjualan"])
                cost_row = find_metric_row(pivot, ["beban pokok", "harga pokok"])
                profit_row = find_metric_row_priority(pivot, [
                    ["laba (rugi) periode berjalan", "laba periode berjalan"],
                    ["laba (rugi) tahun berjalan", "laba tahun berjalan"],
                    ["laba (rugi) bersih", "laba bersih", "laba neto"],
                    ["laba (rugi) komprehensif"],
                    ["laba usaha", "laba (rugi) usaha"],
                    ["jumlah laba", "laba bruto"],
                ])
                m1, m2, m3 = st.columns(3)
                latest_col = pivot.columns[0] if len(pivot.columns) else None
                if rev_row and latest_col:
                    m1.metric(f"Pendapatan ({latest_col})", id_number(pivot.loc[rev_row, latest_col]))
                if profit_row and latest_col:
                    m2.metric(f"Laba ({latest_col})", id_number(pivot.loc[profit_row, latest_col]))
                    st.caption(f"Baris laba yang dipakai: *{profit_row}*")
                if cost_row and latest_col:
                    m3.metric(f"Beban Pokok ({latest_col})", id_number(pivot.loc[cost_row, latest_col]))

                chart_rows = [r for r in [rev_row, cost_row, profit_row] if r]
                if len(chart_rows) >= 2:
                    trend_df = pivot.loc[chart_rows].T.iloc[::-1]  # urut waktu maju
                    rename = {rev_row: "Pendapatan", cost_row: "Beban Pokok", profit_row: "Laba"}
                    trend_df = trend_df.rename(columns=rename)
                    fig, ax = plt.subplots(figsize=(11, 4.5))
                    x = range(len(trend_df))
                    n = len(trend_df.columns)
                    bar_w = 0.8 / n
                    colors = {"Pendapatan": "#3498db", "Beban Pokok": "#e74c3c", "Laba": "#2ecc71"}
                    for i, col in enumerate(trend_df.columns):
                        offset = (i - (n - 1) / 2) * bar_w
                        ax.bar([xi + offset for xi in x], trend_df[col], width=bar_w,
                               label=col, color=colors.get(col, None))
                    ax.set_xticks(list(x))
                    ax.set_xticklabels(trend_df.index, rotation=45, ha="right", fontsize=8)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: id_number(v)))
                    ax.set_title(f"Pendapatan vs Beban Pokok vs Laba per Periode ({statement_label})")
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.2, axis="y")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)

            st.markdown(f"**Detail {statement_label} ({period_label})**")
            st.dataframe(pivot.applymap(id_number), use_container_width=True, height=350)
        else:
            st.caption("Laporan keuangan tidak tersedia untuk ticker/periode ini.")

        st.subheader("📈 Key Statistics / Rasio Valuasi")
        st.caption(
            "⚠️ Menurut dokumentasi resmi InvezGo, endpoint ini masih dalam proses "
            "aktualisasi/kalibrasi — angka bisa kurang akurat, verifikasi silang sebelum dipakai."
        )
        keystat_data, keystat_err = fetch_keystat(ticker, period_code)
        if keystat_err:
            st.error(f"❌ Key statistics gagal diambil: {keystat_err}")
        elif keystat_data:
            keystat_pivot = rows_to_pivot(keystat_data)
            st.dataframe(keystat_pivot.applymap(lambda v: id_number(v, 4)), use_container_width=True, height=350)
        else:
            st.caption("Key statistics tidak tersedia untuk ticker/periode ini.")

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

    if st.button("▶️ Jalankan Screener", type="primary"):
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
        st.session_state["fib_rows"] = rows
        st.session_state["fib_errors"] = errors

    if "fib_rows" not in st.session_state or not st.session_state["fib_rows"]:
        st.info("Klik **▶️ Jalankan Screener** untuk memulai.")
        st.stop()

    rows = st.session_state["fib_rows"]
    errors = st.session_state.get("fib_errors", 0)

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
        default=res["signal"].unique().tolist(),
    )
    str_filter = f2.multiselect(
        "Filter struktur",
        options=res["structure"].unique().tolist(),
        default=res["structure"].unique().tolist(),
    )
    view = res[res["signal"].isin(sig_filter) & res["structure"].isin(str_filter)]
    if view.empty:
        st.warning("Tidak ada saham yang cocok dengan filter — longgarkan pilihan di atas.")
    else:
        cols_order = ["code", "price", "structure", "signal", "keterangan",
                      "fib_level", "entry", "stop_loss", "risk_pct", "tp1", "tp2", "tp3"]
        cols_order = [c for c in cols_order if c in view.columns]
        st.dataframe(view[cols_order], use_container_width=True, hide_index=True)

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
