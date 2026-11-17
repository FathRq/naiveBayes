# Klasifikasi Berita Indonesia Naive Bayes (Spark MLlib)

> Proyek akhir Big Data 2026 Topik 4: Klasifikasi Berita Indonesia
> S1 Informatika, Fakultas Teknik, UNESA

## Apa yang ada di sini

Singkatnya: proyek ini mengklasifikasikan 33.000+ artikel berita ANTARA ke 5
kategori (**hiburan, olahraga, teknologi, politik, ekonomi**) pakai
**Naive Bayes (multinomial)** dari **Spark MLlib**. Alurnya:

```
HDFS (data mentah) -> Spark (bersih) -> MLlib (Naive Bayes) -> Evaluasi -> MongoDB (NoSQL) -> Artikel
```

Hasil akhirnya **akurasi 91,27%** (skenario dedup, yang bebas-leakage). Model
saya latih dari nol (`NaiveBayes.fit`) dengan 100% parameter default
(`smoothing=1.0`, `modelType=multinomial`) tidak pakai model pre-trained,
tidak ada tuning.

README ini saya tulis sebagai info untuk kamu (dan tim): cara setup clusternya,
cara run skripnya, dan khusus untuk yang pegang NoSQL apa yang sudah saya
siapkan dan apa yang perlu kamu kerjakan.

## Clusternya: 7 container

| Container | Image | Fungsi | Port |
|---|---|---|---|
| `namenode` | `apache/hadoop:3.4.3` | HDFS NameNode | 9870 (UI), 9000 |
| `datanode1` | `apache/hadoop:3.4.3` | HDFS DataNode | |
| `datanode2` | `apache/hadoop:3.4.3` | HDFS DataNode | |
| `spark-master` | `apache/spark:3.5.6` | Spark Master | 8080 (UI), 7077, 4040 |
| `spark-worker-1` | `apache/spark:3.5.6` | Spark Worker (2 core, 1G) | 8081 |
| `spark-worker-2` | `apache/spark:3.5.6` | Spark Worker (2 core, 1G) | 8082 |
| `mongodb-uas` | `mongo:8` | NoSQL (opsional) | 27017 |

Catatan: memori worker sudah saya turunkan ke 1G per worker supaya tidak OOM
di mesin dengan RAM 11 GB. Data TF-IDF-nya sparse, jadi 1G cukup.

## Setup (cukup sekali)

1. Nyalakan daemon Docker dulu:
   ```bash
   sudo systemctl start docker
   ```

2. Jalankan clusternya:
   ```bash
   docker compose up -d
   ```
   Tunggu sekitar 1 menit, lalu cek `docker ps` harusnya 7 container `Up`.

3. Format NameNode (hanya saat pertama, HDFS masih kosong):
   ```bash
   docker exec -u root namenode hdfs namenode -format -force
   ```
   Hati-hati, perintah ini **menghapus semua data HDFS**. Jadi cuma dilakukan
   sekali di awal. Setelah data ter-upload, jangan format lagi datanya
   tersimpan di Docker volume dan tetap ada kalau container di-restart.

4. Upload dataset ke HDFS:
   ```bash
   docker cp berita_antara_final.csv namenode:/tmp/dataset.csv
   docker exec -u root namenode hdfs dfs -mkdir -p /proyek_bigdata/raw
   docker exec -u root namenode hdfs dfs -put /tmp/dataset.csv /proyek_bigdata/raw/dataset.csv
   docker exec -u root namenode hdfs dfs -ls /proyek_bigdata/raw/
   ```
   (File `berita_antara_final.csv` tidak ikut di-commit karena besar, jadi
   pastikan kamu punya salinannya di root folder.)

5. Install dependensi Python di spark-master:
   ```bash
   docker exec -u root spark-master pip install numpy pymongo
   ```

Langkah 3–5 cukup sekali. Setelah itu kalau mau `down` lalu `up` lagi, data
HDFS dan MongoDB tetap ada.

## Cara run

Semua skrip dijalankan dari container `spark-master`:

```bash
docker exec spark-master /opt/spark/bin/spark-submit /scripts/<nama>.py
```

Urutannya:

1. **`proses_spark.py`** baca CSV dari HDFS, bersihkan (handle NULL, hapus
   boilerplate copyright, normalisasi), bikin kolom `teks_bersih`, plus beberapa
   analisis. Output: `/proyek_bigdata/clean/berita_clean.parquet`
   (33.776 baris, 0 NULL).

2. **`mllib_naive_bayes_lengkap.py`** ini inti proyeknya. Satu skrip, satu
   perintah, menjalankan skenario eksperimen lengkap:
   - Skenario **RAW** (33.776 baris) → akurasi 91,69%
   - Skenario **DEDUP** (33.555 baris, 221 near-duplikat dibuang) → **91,27%** ← angka resmi
   - Perbandingan: selisihnya cuma 0,42 poin, jadi akurasi tidak tergelembung
     oleh duplikat
   - JSON ringkasan analisis disimpan ke 3 tempat: file lokal, HDFS, MongoDB

   Output: 2 Parquet prediksi (raw + dedup), model, dan
   `ringkasan_analisis.json`.

3. Skrip pendukung (opsional, kalau mau bukti tambahan):
   - `bukti_tidak_leakage.py` bukti visual bahwa model hanya melihat vektor
     TF-IDF (angka), kolom `kategori` hanya jadi label
   - `audit_duplikat.py` audit 221 near-duplikat (semua artikel IHSG)
   - `simpan_mongodb.py` prediksi per artikel → MongoDB

## Hasil

| Metrik | RAW | DEDUP (resmi) |
|---|---|---|
| Accuracy | 91,69% | **91,27%** |
| Weighted Precision | 0,9182 | 0,9138 |
| Weighted Recall | 0,9169 | 0,9127 |
| Weighted F1 | 0,9170 | 0,9128 |

Per kategori (dedup):

| Kategori | Precision | Recall | F1 |
|---|---|---|---|
| olahraga | 0,9953 | 0,9808 | 0,9880 |
| hiburan | 0,9164 | 0,9393 | 0,9277 |
| teknologi | 0,9250 | 0,8519 | 0,8869 |
| ekonomi | 0,8675 | 0,8965 | 0,8817 |
| politik | 0,8646 | 0,8949 | 0,8795 |

Olahraga paling mudah dipisahkan (F1 0,988). Politik yang paling menantang;
kesalahan terbanyak adalah berita politik yang ditebak ekonomi (91 artikel)
masuk akal, karena kedua topik sering beririsan (kebijakan fiskal, regulasi, dll).

## Penyimpanan

HDFS:

```
/proyek_bigdata/
├── raw/dataset.csv                      # dataset mentah (82,6 MB)
├── clean/berita_clean.parquet           # data bersih (33.776 baris)
├── prediksi/berita_prediksi.parquet     # hasil prediksi skenario RAW
├── prediksi/berita_prediksi_dedup.parquet  # hasil prediksi skenario DEDUP
├── model/naive_bayes_model/             # model Naive Bayes
└── analisis/ringkasan_analisis.json     # JSON ringkasan analisis
```

MongoDB (database `proyek_bigdata`):
- `berita_prediksi` 6.556 dokumen, hasil prediksi per artikel
- `ringkasan_analisis` 1 dokumen, ringkasan keseluruhan

## NoSQL
- Container `mongodb-uas` (mongo:8) sudah jalan, port **27017**
- `pymongo` sudah terinstall di container `spark-master`
- Skrip `scripts/simpan_mongodb.py` baca prediksi dari HDFS, tulis ke
  MongoDB. Skripnya idempoten (collection dibersihkan dulu lalu di-insert
  ulang), jadi aman di-run berkali-kali
- Kedua collection di atas sudah terisi

String koneksinya:

```
Dari dalam jaringan Docker (container):  mongodb://mongodb:27017
Dari host (localhost):                   mongodb://localhost:27017
```

Yang perlu dikerjakan:

1. **Tunjukkan minimal 1 operasi tulis + 1 operasi baca** (syarat panduan
   6.D). `simpan_mongodb.py` sudah melakukan keduanya (insert + read-back)
   kamu bisa langsung re-run, atau bikin skripmu sendiri.
   ```bash
   docker exec spark-master /opt/spark/bin/spark-submit /scripts/simpan_mongodb.py
   ```
2. **Untuk visualisasi / web app:** baca collection `ringkasan_analisis`
   (1 dokumen). Isinya sudah lengkap: semua metrik, confusion matrix,
   interpretasi, dalam format JSON siap-render. Field `eksperimen.raw` dan
   `eksperimen.dedup` bisa langsung kamu jadikan grafik perbandingan.
3. Kalau butuh data per artikel (mis. untuk tabel atau filter di web app),
   baca collection `berita_prediksi`.

Skema dokumen:

`berita_prediksi` (1 dokumen = 1 artikel):
```json
{
  "id_artikel":        4525309,
  "kategori":          "hiburan",
  "kategori_prediksi": "hiburan",
  "model":             "Naive Bayes (multinomial)",
  "sumber":            "Spark MLlib",
  "benar":             true
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
    "perbandingan":{ "selisih_accuracy": 0.0042, "kesimpulan": "..." }
  },
  "interpretasi": { "ringkasan": "...", "kategori_terkuat": "olahraga", ... }
}
```

## Kalau ada masalah

| Masalah | Solusi |
|---|---|
| `docker` tidak merespons | `sudo systemctl start docker` |
| Container tidak `Up` | `docker compose up -d` lalu cek `docker ps` |
| `NameNode ... not formatted` | ulangi langkah format (nomor 3 di atas) |
| Error `numpy` / `ModuleNotFoundError` | `docker exec -u root spark-master pip install numpy` |
| Error `pymongo` | `docker exec -u root spark-master pip install pymongo` |
| RAM penuh / OOM | tutup aplikasi berat; worker sudah 1G, bisa diturun ke 512M di `docker-compose.yml` |
| `spark-submit` tidak menemukan skrip | pastikan file ada di `scripts/` (folder itu ter-mount ke `/scripts` di container) |
| Port 9870/8080/27017 bentrok | ubah mapping port di `docker-compose.yml` |

Beberapa jebakan yang saya temui, biar kamu tidak jatuh ke yang sama:
- Python di container Spark itu 3.8/3.9 jangan pakai backslash di f-string,
  jangan pakai nested brace di f-string.
- Di Spark 3.5, estimator-nya namanya `IDF` (bukan `IDFTransformer`).
- Metrik per-kelas di `MulticlassClassificationEvaluator` pakai parameter
  `metricLabel` (bukan `label`).
- Kolom `probability`/`rawPrediction` itu `SparseVector` tidak bisa
  di-index langsung lewat SQL.

## Log / bukti

File log di root folder, bisa dilampirkan sebagai screenshot di artikel:

| File | Isi |
|---|---|
| `logs_proses_spark.txt` | Tahap 1: baca HDFS + bersihkan + analisis |
| `logs_mllib_lengkap.txt` | Tahap 2: RAW + DEDUP + evaluasi + JSON (bukti utama) |
| `logs_bukti_tidak_leakage.txt` | Bukti tidak ada data leakage |
| `terminal(HDFS).txt` | Bukti perintah HDFS (format, upload, ls) |
| `spark_terminal.txt` | Bukti terminal Spark |

Web UI untuk screenshot:
- HDFS NameNode: http://localhost:9870
- Spark Master: http://localhost:8080
- Spark App: http://localhost:4040
