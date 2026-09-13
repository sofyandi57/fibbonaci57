# 📈 InvezGo — Fibonacci Pullback Analyzer (IDX)

Aplikasi [Streamlit](https://streamlit.io) untuk menganalisis saham di Bursa Efek
Indonesia (BEI) dengan strategi **Weak / Strong Pullback Fibonacci**, berdasarkan
materi webinar *"The Final Hunt"* oleh Muhamad Fatah Al-Falah (RHB Sekuritas).

Data harga diambil dari [InvezGo API](https://invezgo.com).

> ⚠️ **Disclaimer**: aplikasi ini murni untuk **edukasi**. Bukan rekomendasi
> beli/jual. Keuntungan/kerugian adalah tanggung jawab pengguna.

---

## ✨ Fitur

| Fitur | Keterangan |
|---|---|
| Input ticker bebas | ketik kode saham IDX apa pun (mis. `BBCA`, `ANTM`, `WINS`) |
| Harga saat ini | diambil real-time dari endpoint chart harian |
| Market structure | otomatis mengenali uptrend / downtrend / sideways dari swing high–low |
| Target Fibonacci | retracement (0% – 100%) + extension (127.2% – 261.8%) |
| Sinyal pullback | deteksi **weak pullback** & **strong pullback** |
| Rencana trading | entry, stop loss, TP1/TP2/TP3, risk-reward |
| Chart | visualisasi harga vs level fib |
| **Screener multi-saham** | scan watchlist custom / seluruh IDX, tabel sinyal terfilter, export CSV |
| **Database cache** | OHLCV **& BDM** di-cache incremental — **Supabase (Postgres, persisten)** atau fallback SQLite lokal |
| **Analisis AI (Groq)** | komentar teknikal berbahasa Indonesia untuk hasil analisis & screener |
| **Screener Bandar** | deteksi **akumulasi / distribusi bandar saat sideways** — termasuk fase *dini* sebelum volume breakout (filter: volume > 1.5x rata-rata 20 hari), pakai indikator BDM invEZGo |

### Cara kerja Screener Bandar

1. **Sideways detector** — range 40 hari sempit (≤15%).
2. **Dominasi BDM** — proporsi hari BDM > 0 dalam 20 hari terakhir:
   ≥60% → akumulasi; ≤40% → distribusi.
3. **Konfirmasi volume** — volume hari ini > 1.5x rata-rata 20 hari:
   - `DINI` = BDM sudah dominan tapi volume masih tidur (fase stealth —
     inilah yang kamu cari sebelum volume besar muncul)
   - `+ BREAKOUT VOLUME` = konfirmasi breakout sudah terjadi
4. Dua tombol filter: 🔵 Akumulasi / 🔴 Distribusi (scan sekali, filter tanpa
   panggil API lagi).

### Filter kedua: konsentrasi broker (opsional)

Menggunakan endpoint `/analysis/summary/stock/{code}` — agregat net volume
(buy − sell) per broker selama 10/20/60 hari:

- **Lolos** jika 3 broker teratas *semuanya* net positif (ngumpulin) **dan**
  net broker #1 ≥ **2x** net broker #2 (tanda akumulasi terkonsentrasi,
  bukan sebaran rata).
- Hasil agregat di-cache di tabel `kv_cache` — scan ulang tidak memakai kuota.
- Di mode single stock, top-3 broker + rasio #1:#2 ditampilkan di panel
  "🏦 Top 3 Broker".

### Catatan tentang screener

Screener berjalan **client-side** di dalam app (bukan endpoint
`/screener/screen` invEZGo), karena perhitungan Fibonacci memerlukan OHLCV
historis per saham, sedangkan endpoint screener bawaan hanya mendukung formula
sederhana (`prev < close`, dst) dengan rate limit 1 request/menit.

## 🔐 Keamanan API Key

API key **tidak pernah disimpan di repository**. Key diambil dari Streamlit
Secrets saat runtime:

- **Streamlit Cloud**: `Settings → Secrets` → tambahkan:
  ```toml
  INVEZGO_API_KEY = "api_key_invezgo_anda"
  GROQ_API_KEY   = "api_key_groq_anda"   # gratis: https://console.groq.com/keys
  ```
- **Lokal**: salin `.streamlit/secrets.toml.example` menjadi
  `.streamlit/secrets.toml` (file ini di-*ignore* git) dan isi key-nya.

Dapatkan API key di: <https://invezgo.com/id/setting/api> (memerlukan paket
langganan aktif — endpoint chart membutuhkan hak akses berbayar).

## 🗄️ Setup Supabase (cache persisten, opsional tapi disarankan)

Filesystem Streamlit Cloud bersifat *ephemeral* — tanpa database eksternal,
cache hilang tiap redeploy. App ini mendukung **Supabase** otomatis:
jika secrets Supabase ada → dipakai; jika tidak → fallback SQLite lokal.

1. Buat project gratis di [supabase.com](https://supabase.com) → **New project**.
2. Buka **SQL Editor**, jalankan:
   ```sql
   CREATE TABLE candles (
     code   TEXT NOT NULL,
     date   DATE NOT NULL,
     open   REAL, high REAL, low REAL, close REAL, volume REAL,
     PRIMARY KEY (code, date)
   );
   CREATE INDEX idx_candles_code ON candles(code);

   CREATE TABLE bdm (
     code  TEXT NOT NULL,
     date  DATE NOT NULL,
     value REAL,
     PRIMARY KEY (code, date)
   );

   CREATE TABLE kv_cache (
     key        TEXT PRIMARY KEY,
     payload    TEXT,
     fetched_at TEXT
   );
   ```
3. Ambil kredensial di **Settings → API**:
   - `Project URL` → masukkan sebagai `SUPABASE_URL`
   - `anon public key` → masukkan sebagai `SUPABASE_KEY`
     (tabel di atas default-nya bisa diakses anon; kalau kamu lock dengan
     RLS, pakai `service_role` key — jangan pernah commit key ini!)
4. Tambahkan ke Streamlit Secrets (sejajar key lain):
   ```toml
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_KEY = "eyJhbGciOi..."
   ```

Tanpa secrets Supabase, app tetap berjalan penuh dengan SQLite lokal.

## 🚀 Menjalankan Lokal

```bash
git clone https://github.com/<user>/invezgo-fib-app.git
cd invezgo-fib-app

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# buat secrets lokal
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# lalu edit .streamlit/secrets.toml, isi INVEZGO_API_KEY

streamlit run app.py
```

## ☁️ Deploy ke Streamlit Cloud

1. Push repo ini ke GitHub (tanpa `secrets.toml`).
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app** → pilih repo.
3. **Settings → Secrets**:
   ```toml
   INVEZGO_API_KEY = "api_key_anda"
   ```
4. Deploy. Selesai.

## 🧠 Metodologi

### Penentuan swing (aturan konfirmasi)
- **Swing high** valid jika candle berikutnya *close di bawah body* candle
  sebelum puncak.
- **Swing low** valid jika candle berikutnya *close di atas body* candle
  sebelum lembah.

### Level Fibonacci
- Retracement: `0%, 38.2%, 50%, 61.8%, 78.6%, 100%`
- Extension: `127.2%, 141.4%, 161.8%, 200%, 261.8%`
- Catatan: level `23.6%` sengaja tidak dipakai (lemah di back-testing materi
  sumber); `50%` dan `78.6%` dipakai meski bukan golden ratio karena sering
  menjadi area reaksi harga.

### Sinyal
- **Weak pullback** — harga turun satu level fib lalu rebound → *speculative
  buy*, SL di bawah level fib, TP mengikuti level berikutnya (1:1, lalu
  extension).
- **Strong pullback** — harga menembus satu level fib → **tunggu** di level
  fib bawahnya. Entry dibatalkan jika jarak SL melebihi batas risiko yang
  kamu tetapkan (prinsip *jangan dipaksakan*).

## 🗂️ Struktur Proyek

```
invezgo-fib-app/
├── app.py                          # aplikasi Streamlit utama
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # template secrets (JANGAN diisi key asli)
```

## 📚 Endpoint InvezGo yang Dipakai

| Endpoint | Fungsi |
|---|---|
| `GET /analysis/chart/stock/{code}` | OHLCV harian (maks. 2 tahun ke belakang) |
| `GET /analysis/chart/multi-time/{code}` | (opsional) timeframe 1 jam untuk konfirmasi pendek |
| `GET /analysis/chart/stock/bdm/{code}` | Indikator bandarmologi (BDM) harian |
| `GET /analysis/summary/stock/{code}` | Agregat buy/sell per broker (konsentrasi broker, akumulator) |
| `GET /analysis/shareholder/{code}` | Komposisi pemegang saham >1% (snapshot terbaru) |
| `GET /analysis/shareholder-insider` | Riwayat transaksi insider (query: `code`, `from`, `to`, `page`, `limit`) |

Autentikasi: header `Authorization: Bearer <API_KEY>`.

Path dua endpoint kepemilikan di atas sudah diverifikasi langsung ke OpenAPI
spec resmi InvezGo (`api-1.json`).

## 📝 Lisensi & Atribusi

- Data: © InvezGo (pengguna berperan sebagai pihak ketiga, tidak untuk
  distribusi ulang data mentah).
- Struktur strategi: edukasi publik dari webinar RHB Sekuritas.
- Kode: bebas dipakai untuk keperluan pribadi/edukasi.
