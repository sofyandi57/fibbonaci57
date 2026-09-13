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
  INVEZGO_API_KEY = "api_key_anda"
  ```
- **Lokal**: salin `.streamlit/secrets.toml.example` menjadi
  `.streamlit/secrets.toml` (file ini di-*ignore* git) dan isi key-nya.

Dapatkan API key di: <https://invezgo.com/id/setting/api> (memerlukan paket
langganan aktif — endpoint chart membutuhkan hak akses berbayar).

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

Autentikasi: header `Authorization: Bearer <API_KEY>`.

## 📝 Lisensi & Atribusi

- Data: © InvezGo (pengguna berperan sebagai pihak ketiga, tidak untuk
  distribusi ulang data mentah).
- Struktur strategi: edukasi publik dari webinar RHB Sekuritas.
- Kode: bebas dipakai untuk keperluan pribadi/edukasi.
