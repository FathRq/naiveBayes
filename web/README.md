# Dashboard Web — Klasifikasi Berita Indonesia

Dashboard Tailwind untuk **mempresentasikan seluruh hasil proyek** sesuai
Panduan Proyek Akhir Big Data 2026: dataset, arsitektur HDFS–Spark,
evaluasi Spark MLlib, dan hasil NoSQL (MongoDB).

## Arsitektur

```
MongoDB (proyek_bigdata)          data/analisis/
  ringkasan_analisis      ──┐       ringkasan_analisis.json (fallback)
  berita_prediksi (DEDUP) ──┤
                            ▼
                  web/backend/app.py  (Flask API)
                    /api/status  /api/ringkasan
                    /api/agregasi  /api/prediksi
                            ▼
                  web/static/  (Tailwind + Chart.js)
                    index.html + app.js  ->  http://localhost:5000
```

Backend **terhubung ke backend yang sudah ada** (MongoDB dari
`docker-compose.yml`) — tidak ada database baru. Jika MongoDB belum
terisi, `/api/ringkasan` otomatis memakai file
`data/analisis/ringkasan_analisis.json` sebagai fallback agar dashboard
tetap bisa didemokan; status sumber data selalu ditampilkan di header.

## Isi dashboard (mengikuti struktur luaran panduan)

1. **KPI** — akurasi resmi DEDUP 91,27%, weighted F1, total artikel,
   jumlah dokumen prediksi di MongoDB.
2. **Dataset** — sumber ANTARA, keseimbangan kelas per kategori (grafik).
3. **Evaluasi MLlib** — perbandingan eksperimen RAW vs DEDUP (grafik +
   kesimpulan bebas-leakage), metrik per kategori (precision/recall/F1),
   bisa berganti skenario.
4. **Confusion matrix** DEDUP/RAW + **interpretasi** hasil (kategori
   terkuat/terlemah, pasangan paling sering salah, detail model).
5. **NoSQL** — distribusi kategori prediksi dan pasangan salah prediksi
   terbanyak, dihitung dengan pipeline agregasi MongoDB; akurasi
   dihitung ulang langsung dari MongoDB.
6. **Explorer prediksi** — tabel prediksi per artikel dari collection
   `berita_prediksi` dengan filter kategori asli / prediksi / status
   dan pagination.

## Cara menjalankan

Disarankan lewat Docker Compose (service `web` di `docker-compose.yml`
root repo):

```bash
docker compose up -d --build web
# buka http://localhost:5000
```

Prasyarat agar semua bagian aktif: cluster jalan, lalu jalankan pipeline
utama dan pengisian MongoDB seperti di README utama:

```bash
docker exec spark-master /opt/spark/bin/spark-submit /scripts/mllib_naive_bayes_lengkap.py
docker exec spark-master /opt/spark/bin/spark-submit /scripts/simpan_mongodb.py
```

Tanpa Docker (mode fallback file JSON tetap jalan untuk ringkasan):

```bash
pip install -r web/backend/requirements.txt
MONGO_URI=mongodb://localhost:27017 python web/backend/app.py
```
