# Klasifikasi Berita Indonesia Naive Bayes (Spark MLlib)

> Proyek akhir Big Data 2026 Topik 4: Klasifikasi Berita Indonesia
> S1 Informatika, Fakultas Teknik, UNESA

## Apa yang ada di sini

Singkatnya: proyek ini mengklasifikasikan 33.000+ artikel berita ANTARA ke 5 kategori (**hiburan, olahraga, teknologi, politik, ekonomi**) pakai **Naive Bayes (multinomial)** dari **Spark MLlib**. Alurnya:

```
Scraping ANTARA (Python) -> HDFS (data mentah) -> Spark (bersih) -> MLlib (Naive Bayes) -> Evaluasi -> MongoDB (NoSQL) -> Artikel
```

Datasetnya bukan data siap pakai. Saya kumpulkan sendiri dari antaranews.com pakai `scrapping_data.py`, detailnya di bagian Tahap 0 di bawah.

Hasil akhirnya **akurasi 91,84%** (skenario dedup, yang bebas-leakage; eksekusi final di laptop penulis, lihat bagian _Lingkungan eksekusi_). Model saya latih dari nol (`NaiveBayes.fit`) dengan 100% parameter default (`smoothing=1.0`, `modelType=multinomial`) tidak pakai model pre-trained, tidak ada tuning.

README ini saya tulis sebagai info untuk kamu (dan tim): cara setup clusternya, cara run skripnya, dan khusus untuk yang pegang NoSQL apa yang sudah saya siapkan dan apa yang perlu kamu kerjakan.

## Tahap 0: Kumpulkan data (scraping)

Sebelum masuk ke HDFS, datanya saya kumpulkan sendiri dari situs resmi ANTARA News (antaranews.com) dengan skrip `scrapping_data.py`. Skrip ini jalan di mesin lokal (Python biasa, bukan di dalam Docker), jadi satu-satunya tahap yang tidak pakai cluster.

Cara jalankan:

```bash
pip install requests beautifulsoup4 lxml
python scrapping_data.py --contact emailkamu@mhs.unesa.ac.id --delay 0.5 --workers 4
```

Ganti `--contact` dengan email aslimu. Email itu masuk ke User-Agent supaya pengelola situs bisa menghubungimu kalau ada masalah.

Skripnya bekerja dalam 2 fase:

1. RSS resmi per kanal (ambil berita terbaru)
2. Crawl halaman indeks per kategori (politik, ekonomi, olahraga, teknologi, hiburan), round-robin antar kelas supaya datanya seimbang

Beberapa hal yang saya buat di skrip ini:

- Etis: patuh robots.txt (termasuk Crawl-delay), rate-limiter global, backoff otomatis saat HTTP 429/503, dan berhenti aman kalau terdeteksi diblokir (401/403 beruntun). Tidak login, tidak lewat paywall.
- Bisa di-resume: stop kapan saja dengan Ctrl+C, progres tersimpan di checkpoint (CSV + state.json). Jalankan perintah yang sama untuk melanjutkan.
- Finalize: setelah pengumpulan selesai, skrip dedup URL dan menyeimbangkan kelas, lalu menulis `berita_antara_final.csv` (UTF-8, siap dibaca Spark) dan `metadata_pengumpulan.json`. Kalau cuma mau mengekspor ulang tanpa crawl baru: `python scrapping_data.py --only-finalize`.

Output default skrip ada di folder `data_berita/` (CSV mentah, CSV final, state, log). File `berita_antara_final.csv` di root folder proyek adalah hasil dari proses ini (33.830 artikel, 9 kolom).

> **Catatan hak cipta:** setiap artikel ANTARA memuat pemberitahuan yang melarang pengambilan konten, crawling, atau pengindeksan otomatis untuk AI tanpa izin tertulis dari ANTARA. Karena itu **dataset tidak dipublikasikan** (tidak di-commit ke repo, tidak diunggah ke Kaggle/GitHub). Dataset hanya diserahkan kepada dosen pengampu melalui Google Drive untuk keperluan penilaian akademik. Pihak lain yang ingin memakai data perlu mengumpulkan ulang dengan izin dari ANTARA.

## Clusternya: 7 container + 1 dashboard web

| Container        | Image                              | Fungsi                    | Port                  |
| ---------------- | ---------------------------------- | ------------------------- | --------------------- |
| `namenode`       | `apache/hadoop:3.4.3`              | HDFS NameNode             | 9870 (UI), 9000       |
| `datanode1`      | `apache/hadoop:3.4.3`              | HDFS DataNode             |                       |
| `datanode2`      | `apache/hadoop:3.4.3`              | HDFS DataNode             |                       |
| `spark-master`   | `apache/spark:3.5.6`               | Spark Master              | 8080 (UI), 7077, 4040 |
| `spark-worker-1` | `apache/spark:3.5.6`               | Spark Worker (2 core, 1G) | 8081                  |
| `spark-worker-2` | `apache/spark:3.5.6`               | Spark Worker (2 core, 1G) | 8082                  |
| `mongodb-uas`    | `mongo:8`                          | NoSQL (opsional)          | 27017                 |
| `web-dashboard`  | `python:3.12-slim` (build `./web`) | Dashboard web (opsional)  | 5000                  |

Catatan: memori worker sudah saya turunkan ke 1G per worker supaya tidak OOM di laptop dengan RAM 12 GB (11,7 GB usable). Data TF-IDF-nya sparse, jadi 1G cukup.

### Lingkungan eksekusi

Seluruh angka di README ini berasal dari satu laptop (log final tanggal 21 November 2026):

| Komponen   | Spesifikasi                               |
| ---------- | ----------------------------------------- |
| Prosesor   | Intel Core i5-12450HX, 2,40 GHz           |
| RAM        | 12 GB (11,7 GB usable)                    |
| Lingkungan | Docker di atas WSL2, Spark 3.5.6, Java 11 |

Hasil pada Spark tidak persis identik antar-eksekusi (pembagian acak `randomSplit` bisa berbeda tipis walau `seed=42`). Pada beberapa kali eksekusi, akurasi DEDUP berkisar 91,3% sampai 91,9%, jadi selisih RAW vs DEDUP di bawah 0,5 poin masih dalam rentang variasi itu. Angka resmi untuk artikel adalah angka dari log final di bawah.

## Setup (cukup sekali)

1. Nyalakan daemon Docker dulu:

   ```bash
   sudo systemctl start docker
   ```

2. Jalankan clusternya:

   ```bash
   docker compose up -d
   ```

   Tunggu sekitar 1 menit, lalu cek `docker ps` harusnya 7 container `Up` (8 kalau service `web` ikut dijalankan — khusus web, pertama kali perlu `docker compose up -d --build web`).

3. Format NameNode (hanya saat pertama, HDFS masih kosong):

   ```bash
   docker exec -u root namenode hdfs namenode -format -force
   ```

   Hati-hati, perintah ini **menghapus semua data HDFS**. Jadi cuma dilakukan sekali di awal. Setelah data ter-upload, jangan format lagi datanya tersimpan di Docker volume dan tetap ada kalau container di-restart.

4. Upload dataset ke HDFS:

   ```bash
   docker cp berita_antara_final.csv namenode:/tmp/dataset.csv
   docker exec -u root namenode hdfs dfs -mkdir -p /proyek_bigdata/raw
   docker exec -u root namenode hdfs dfs -put /tmp/dataset.csv /proyek_bigdata/raw/dataset.csv
   docker exec -u root namenode hdfs dfs -ls /proyek_bigdata/raw/
   ```

   (File `berita_antara_final.csv` tidak ikut di-commit dan tidak dipublikasikan, lihat catatan hak cipta di Tahap 0. File ini hasil Tahap 0, kalau belum ada jalankan `scrapping_data.py` dulu atau minta salinannya ke saya.)

5. Install dependensi Python di spark-master:

   ```bash
   docker exec -u root spark-master pip install numpy pymongo
   ```

Langkah 3–5 cukup sekali. Setelah itu kalau mau `down` lalu `up` lagi, data HDFS dan MongoDB tetap ada.

## Cara run

Semua skrip dijalankan dari container `spark-master`:

```bash
docker exec spark-master /opt/spark/bin/spark-submit /scripts/<nama>.py
```

Urutannya:

1. **`proses_spark.py`** baca CSV dari HDFS, bersihkan (handle NULL, hapus boilerplate copyright, normalisasi), bikin kolom `teks_bersih`, plus beberapa analisis. Output: `/proyek_bigdata/clean/berita_clean.parquet` (33.776 baris, 0 NULL).

2. **`mllib_naive_bayes_lengkap.py`** ini inti proyeknya. Satu skrip, satu perintah, menjalankan skenario eksperimen lengkap:
   - Skenario **RAW** (33.776 baris) → akurasi 91,72%
   - Skenario **DEDUP** (33.555 baris, 221 near-duplikat dibuang) → **91,84%** ← angka resmi
   - Perbandingan: selisihnya cuma 0,12 poin, jadi akurasi tidak tergelembung oleh duplikat
   - JSON ringkasan analisis disimpan ke 3 tempat: file lokal, HDFS, MongoDB

   Output: 2 Parquet prediksi (raw + dedup), model, dan `ringkasan_analisis.json`.

3. Skrip pendukung (opsional, kalau mau bukti tambahan):
   - `bukti_tidak_leakage.py` bukti visual bahwa model hanya melihat vektor TF-IDF (angka), kolom `kategori` hanya jadi label
   - `audit_duplikat.py` audit duplikasi: tidak ada judul kembar persis; 290 baris (0,9% data) dalam 55 kelompok punya 30 kata pertama identik, terbanyak artikel pembaruan harian IHSG. Audit juga menemukan 103 artikel yang isinya hanya pemberitahuan hak cipta dan 10 artikel yang hanya berisi kredit penerjemah/editor
   - Skrip eksperimen skala data (application name `Eksperimen_Skala_Data`, <!-- sesuaikan nama file skripnya, mis. eksperimen_skala.py -->): melatih ulang model DEDUP pada 25%, 50%, 75%, 100% data dan mencatat waktu serta akurasi
   - `simpan_mongodb.py` prediksi per artikel skenario **DEDUP** → MongoDB (operasi tulis)
   - `baca_mongodb.py` baca kembali MongoDB tanpa Spark (operasi baca + agregasi)

## Hasil

| Metrik             | RAW    | DEDUP (resmi) |
| ------------------ | ------ | ------------- |
| Accuracy           | 91,72% | **91,84%**    |
| Weighted Precision | 0,9184 | 0,9198        |
| Weighted Recall    | 0,9172 | 0,9184        |
| Weighted F1        | 0,9173 | 0,9185        |

Per kategori (dedup):

| Kategori  | Data uji | Precision | Recall | F1     |
| --------- | -------- | --------- | ------ | ------ |
| olahraga  | 1.324    | 0,9954    | 0,9841 | 0,9897 |
| hiburan   | 1.356    | 0,9315    | 0,9432 | 0,9373 |
| ekonomi   | 1.294    | 0,8914    | 0,8941 | 0,8927 |
| teknologi | 1.256    | 0,9296    | 0,8511 | 0,8886 |
| politik   | 1.291    | 0,8490    | 0,9148 | 0,8807 |

Olahraga paling mudah dipisahkan (F1 0,990). Politik yang paling menantang (precision terendah, 0,849); kesalahan terbanyak adalah berita ekonomi yang ditebak politik (100 artikel), diikuti teknologi yang ditebak politik (77) dan politik yang ditebak ekonomi (69). Masuk akal, karena politik dan ekonomi sering beririsan (kebijakan fiskal, regulasi, dll).

### Eksperimen skala data (skenario DEDUP)

| Skala | Jumlah artikel | Waktu (detik) | Accuracy | F1     |
| ----- | -------------- | ------------- | -------- | ------ |
| 25%   | 8.434          | 29,33         | 91,14%   | 0,9118 |
| 50%   | 16.733         | 15,72         | 91,61%   | 0,9162 |
| 75%   | 25.026         | 13,09         | 91,53%   | 0,9155 |
| 100%  | 33.555         | 10,82         | 91,51%   | 0,9152 |

Catatan membaca tabel ini:

- Akurasi stabil di 91,1% sampai 91,6% dan tidak naik berarti setelah 50% data.
- Waktu **menurun** seiring data bertambah, jadi tidak mencerminkan kompleksitas pelatihan. Kemungkinan karena eksekusi pertama (25%) menanggung biaya pemanasan (executor, JVM, pembacaan awal HDFS) dan pengukurannya hanya sekali jalan. Untuk hasil yang andal, jalankan satu putaran pemanasan dulu lalu ukur tiap skala 3 sampai 5 kali.
- Akurasi 100% di sini (91,51%) beda 0,33 poin dari angka resmi (91,84%), kemungkinan karena pembagian acak data latih/uji tidak identik antar-skrip.

## Penyimpanan

HDFS:

```
/proyek_bigdata/
├── raw/dataset.csv                      # dataset mentah, hasil scraping (82,6 MB)
├── clean/berita_clean.parquet           # data bersih (33.776 baris)
├── prediksi/berita_prediksi.parquet     # hasil prediksi skenario RAW
├── prediksi/berita_prediksi_dedup.parquet  # hasil prediksi skenario DEDUP
├── model/naive_bayes_model/             # model Naive Bayes
└── analisis/ringkasan_analisis.json     # JSON ringkasan analisis
```

MongoDB (database `proyek_bigdata`):

- `berita_prediksi` 6.521 dokumen, hasil prediksi per artikel skenario **DEDUP** (angka resmi: 5.989 benar, 532 salah)
- `ringkasan_analisis` 1 dokumen, ringkasan keseluruhan

## NoSQL

- Container `mongodb-uas` (mongo:8) sudah jalan, port **27017**
- `pymongo` sudah terinstall di container `spark-master`
- Skrip `scripts/simpan_mongodb.py` baca prediksi **DEDUP** dari HDFS (`berita_prediksi_dedup.parquet`), tulis ke MongoDB. Skripnya idempoten (collection dibersihkan dulu lalu di-insert ulang), jadi aman di-run berkali-kali. Setelah insert, skrip membuat index pada `id_artikel` (unique), `kategori_prediksi`, `benar`, serta index gabungan `kategori` + `kategori_prediksi` (total 5 index termasuk `_id`).
- Skrip `scripts/baca_mongodb.py` adalah pasangan bacanya: murni PyMongo (tanpa Spark), menampilkan ringkasan, akurasi yang dihitung ulang dari MongoDB, agregasi distribusi per `kategori_prediksi`, dan pasangan salah prediksi terbanyak.
- Kedua collection di atas sudah terisi (setelah `simpan_mongodb.py` dijalankan ulang, isinya skenario DEDUP: 6.521 dokumen).

Catatan pilihan teknologi: integrasi Spark ↔ MongoDB di sini memakai **PyMongo** (DataFrame dikumpulkan ke driver, ±6,5 ribu dokumen, lalu `insert_many`), bukan MongoDB Spark Connector. Alasannya: panduan mengingatkan koneksi Spark–MongoDB harus sesuai versi Spark dan Scala — connector resmi menambah risiko ketidakcocokan versi (Spark 3.5.6 / Scala 2.12) tanpa manfaat untuk volume data sebesar ini. Pola tulis + baca yang disyaratkan panduan 6.D tetap terpenuhi penuh.

String koneksinya:

```
Dari dalam jaringan Docker (container):  mongodb://mongodb:27017
Dari host (localhost):                   mongodb://localhost:27017
```

Yang perlu dikerjakan:

1. **Tunjukkan minimal 1 operasi tulis + 1 operasi baca** (syarat panduan 6.D). Tulisnya `simpan_mongodb.py` (insert + read-back), bacanya bisa juga mandiri lewat `baca_mongodb.py`:

   ```bash
   docker exec spark-master /opt/spark/bin/spark-submit /scripts/simpan_mongodb.py
   docker exec spark-master python3 /scripts/baca_mongodb.py
   ```

   Untuk bukti artikel, simpan output keduanya ke file log:

   ```bash
   docker exec spark-master /opt/spark/bin/spark-submit /scripts/simpan_mongodb.py > 7_log_simpan_mongodb.txt 2>&1
   docker exec spark-master python3 /scripts/baca_mongodb.py > 8_log_baca_mongodb.txt 2>&1
   ```

   Hasil verifikasi di log final: 6.521 dokumen tersisip, 5.989 prediksi benar, 532 salah, akurasi yang dihitung dari MongoDB **0,9184** (sama dengan hasil evaluasi di Spark).

   Hasil baca di `8_log_baca_mongodb.txt`: distribusi kategori prediksi (politik 1.391, hiburan 1.373, olahraga 1.309, ekonomi 1.298, teknologi 1.150) dan pasangan salah prediksi terbanyak (ekonomi ke politik 100, teknologi ke politik 77, politik ke ekonomi 69, teknologi ke ekonomi 55, teknologi ke hiburan 53), semuanya cocok dengan confusion matrix DEDUP dari Spark.

2. **Untuk visualisasi / web app:** baca collection `ringkasan_analisis` (1 dokumen). Isinya sudah lengkap: semua metrik, confusion matrix, interpretasi, dalam format JSON siap-render. Field `eksperimen.raw` dan `eksperimen.dedup` bisa langsung kamu jadikan grafik perbandingan. Dashboard webnya sudah dibuat — lihat bagian **Dashboard Web** di bawah.

3. Kalau butuh data per artikel (mis. untuk tabel atau filter di web app), baca collection `berita_prediksi`.

Skema dokumen:

`berita_prediksi` (1 dokumen = 1 artikel):

```json
{
  "id_artikel": 383659,
  "kategori": "teknologi",
  "kategori_prediksi": "teknologi",
  "model": "Naive Bayes (multinomial)",
  "sumber": "Spark MLlib",
  "skenario": "dedup",
  "benar": true
}
```

`ringkasan_analisis` (1 dokumen = ringkasan keseluruhan):

```json
{
  "proyek":       { "judul": "...", "algoritma": "Naive Bayes (multinomial)", ... },
  "dataset":      { "total_baris_dedup": 33555, "keseimbangan_kelas": {...}, ... },
  "model":        { "tipe": "multinomial", "fitur": "TF-IDF", ... },
  "eksperimen":   {
    "raw":         { "evaluasi": {...}, "metrik_per_kategori": [...], "confusion_matrix": {...} },
    "dedup":       { "evaluasi": {...}, "metrik_per_kategori": [...], "confusion_matrix": {...} },
    "perbandingan":{ "selisih_accuracy": 0.0012, "kesimpulan": "..." }
  },
  "interpretasi": { "ringkasan": "...", "kategori_terkuat": "olahraga", ... }
}
```

## Dashboard Web

Dashboard Tailwind untuk mempresentasikan semua hasil proyek (dataset, evaluasi RAW vs DEDUP, metrik per kategori, confusion matrix, interpretasi, data NoSQL, dan explorer prediksi per artikel). Frontend-nya di folder `web/static`, backend API-nya Flask di `web/backend/app.py` yang membaca **MongoDB yang sudah ada** (fallback ke `data/analisis/ringkasan_analisis.json` kalau MongoDB belum terisi). Detailnya ada di `web/README.md`.

Jalankan:

```bash
docker compose up -d --build web
# buka http://localhost:5000
```

## Kalau ada masalah

| Masalah                               | Solusi                                                                                 |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| `docker` tidak merespons              | `sudo systemctl start docker`                                                          |
| Container tidak `Up`                  | `docker compose up -d` lalu cek `docker ps`                                            |
| `NameNode ... not formatted`          | ulangi langkah format (nomor 3 di atas)                                                |
| Error `numpy` / `ModuleNotFoundError` | `docker exec -u root spark-master pip install numpy`                                   |
| Error `pymongo`                       | `docker exec -u root spark-master pip install pymongo`                                 |
| RAM penuh / OOM                       | tutup aplikasi berat; worker sudah 1G, bisa diturun ke 512M di `docker-compose.yml`    |
| `spark-submit` tidak menemukan skrip  | pastikan file ada di `scripts/` (folder itu ter-mount ke `/scripts` di container)      |
| Port 9870/8080/27017 bentrok          | ubah mapping port di `docker-compose.yml`                                              |
| Dashboard web tidak tampil            | `docker compose up -d --build web`, cek `docker logs web-dashboard`                    |
| Scraper berhenti sendiri (AbortCrawl) | kemungkinan diblokir; tunggu dulu, lalu re-run dengan `--delay` lebih besar (mis. 1.0) |

Beberapa jebakan yang saya temui, biar kamu tidak jatuh ke yang sama:

- Python di container Spark itu 3.8/3.9 jangan pakai backslash di f-string, jangan pakai nested brace di f-string.
- Di Spark 3.5, estimator-nya namanya `IDF` (bukan `IDFTransformer`).
- Metrik per-kelas di `MulticlassClassificationEvaluator` pakai parameter `metricLabel` (bukan `label`).
- Kolom `probability`/`rawPrediction` itu `SparseVector` tidak bisa di-index langsung lewat SQL.

## Log / bukti

File log di root folder (log final, dijalankan di laptop penulis), bisa dilampirkan sebagai screenshot di artikel:

| File                         | Isi                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `1_log_hdfs.txt`             | Bukti file dataset di HDFS (`hdfs dfs -ls -h`, 82,6 MB, replikasi 2)                                                     |
| `2_log_proses_spark.txt`     | Tahap 1: baca HDFS + bersihkan + 4 analisis                                                                              |
| `3_log_mllib_lengkap.txt`    | Tahap 2: RAW + DEDUP + evaluasi + JSON (bukti utama)                                                                     |
| `4_log_audit_duplikat.txt`   | Audit duplikasi data                                                                                                     |
| `5_log_bukti_leakage.txt`    | Bukti tidak ada data leakage                                                                                             |
| `6_log_eksperimen_skala.txt` | Eksperimen skala data 25% sampai 100%                                                                                    |
| `7_log_simpan_mongodb.txt`   | Bukti operasi tulis + verifikasi baca MongoDB                                                                            |
| `8_log_baca_mongodb.txt`     | Bukti operasi baca MongoDB tanpa Spark: ringkasan, akurasi hitung ulang, agregasi distribusi dan pasangan salah prediksi |
| `terminal(HDFS).txt`         | Bukti perintah HDFS (format, upload, ls)                                                                                 |
| `spark_terminal.txt`         | Bukti terminal Spark                                                                                                     |

Web UI untuk screenshot:

- HDFS NameNode: http://localhost:9870
- Spark Master: http://localhost:8080
- Spark App: http://localhost:4040
- Dashboard Web: http://localhost:5000
