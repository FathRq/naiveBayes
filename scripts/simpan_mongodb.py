# ==============================================================================
# TITIK KONEKSI: SPARK -> MONGODB (opsional, dikerjakan oleh tim NoSQL)
# ------------------------------------------------------------------------------
# Skrip ini membaca hasil prediksi Naive Bayes (Parquet di HDFS) dan menyimpan
# -nya ke MongoDB sebagai dokumen JSON. Ini adalah "titik koneksi" antara
# bagian Spark/MLlib (kita) dengan bagian NoSQL (teman).
#
# PERUBAHAN (2026-11-19):
#   - Sumber data diganti dari skenario RAW ke skenario DEDUP
#     (berita_prediksi_dedup.parquet), karena DEDUP adalah ANGKA RESMI
#     proyek ini (akurasi 91,27%, bebas-leakage). Sebelumnya collection
#     berita_prediksi berisi prediksi RAW (6.556 dokumen, akurasi 91,69%)
#     sehingga tidak konsisten dengan ringkasan_analisis.
#   - Setiap dokumen sekarang membawa field "skenario": "dedup".
#   - Setelah insert, dibuat index pada id_artikel (unique),
#     kategori_prediksi, dan benar, agar query aplikasi web cepat.
#
# DATA CONTRACT (skema dokumen):
#   database  : proyek_bigdata
#   collection: berita_prediksi
#   dokumen   : {
#       "id_artikel"       : long,     # id artikel dari dataset
#       "kategori"         : string,   # label sebenarnya (ground truth)
#       "kategori_prediksi": string,   # hasil prediksi model
#       "model"            : string,   # "Naive Bayes (multinomial)"
#       "sumber"           : string,   # "Spark MLlib"
#       "skenario"         : string,   # "dedup" (skenario resmi)
#       "benar"            : bool      # prediksi == label sebenarnya?
#   }
#
# CARA JALANKAN (dari host, container spark-master):
#   1. Install driver di container:
#        docker exec -u root spark-master pip install pymongo
#   2. Submit:
#        docker exec spark-master /opt/spark/bin/spark-submit /scripts/simpan_mongodb.py
#
# Jika pymongo belum terinstall, skrip ini AKAN berhenti dengan pesan
# instruksi (tidak akan error/crash) — sehingga aman dijalankan kapan saja.
# ==============================================================================

import sys

# ------------------------------------------------------------------
# 0. Cek dependensi (pymongo) — berhenti sopan jika belum ada
# ------------------------------------------------------------------
try:
    import pymongo
except ImportError:
    print("=" * 70)
    print("PANDUAN: driver MongoDB (pymongo) belum terinstall di container.")
    print("Jalankan perintah ini dulu, lalu submit ulang skrip ini:")
    print()
    print("    docker exec -u root spark-master pip install pymongo")
    print()
    print("Setelah itu jalankan:")
    print("    docker exec spark-master /opt/spark/bin/spark-submit /scripts/simpan_mongodb.py")
    print("=" * 70)
    sys.exit(0)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ------------------------------------------------------------------
# Konfigurasi koneksi
# ------------------------------------------------------------------
MONGO_URI     = "mongodb://mongodb:27017"   # hostname service compose = "mongodb"
MONGO_DB      = "proyek_bigdata"
MONGO_COLL    = "berita_prediksi"
# Skenario DEDUP = angka resmi (bebas-leakage). BUKAN file RAW.
PATH_PREDIKSI = "hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi_dedup.parquet"
MODEL_NAME    = "Naive Bayes (multinomial)"
SUMBER        = "Spark MLlib"
SKENARIO      = "dedup"

# ------------------------------------------------------------------
# 1. Spark session + baca hasil prediksi dari HDFS
# ------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("BagianD_SparkKeMongoDB")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n=== 1. MEMBACA HASIL PREDIKSI DARI HDFS ===")
print("Sumber: " + PATH_PREDIKSI + " (skenario " + SKENARIO.upper() + ")")
df = spark.read.parquet(PATH_PREDIKSI)
df = (df
      .withColumn("benar", col("kategori") == col("kategori_prediksi"))
      .select("id_artikel", "kategori", "kategori_prediksi", "benar"))
total = df.count()
print("Total dokumen yang akan disimpan: " + str(total))

# ------------------------------------------------------------------
# 2. Convert DataFrame -> list dict (sekitar 6,5 ribu baris, aman di driver)
# ------------------------------------------------------------------
print("\n=== 2. KONVERSI KE DOKUMEN JSON ===")
docs = []
for r in df.collect():
    docs.append({
        "id_artikel":        int(r["id_artikel"]),
        "kategori":          r["kategori"],
        "kategori_prediksi": r["kategori_prediksi"],
        "model":             MODEL_NAME,
        "sumber":            SUMBER,
        "skenario":          SKENARIO,
        "benar":             bool(r["benar"]),
    })

# ------------------------------------------------------------------
# 3. OPERASI TULIS: insert ke MongoDB
# ------------------------------------------------------------------
print("\n=== 3. OPERASI TULIS (INSERT) KE MONGODB ===")
client = pymongo.MongoClient(MONGO_URI)
db     = client[MONGO_DB]
coll   = db[MONGO_COLL]

# bersihkan data lama agar re-run idempoten
coll.delete_many({})
result = coll.insert_many(docs)
print("Dokumen disisipkan: " + str(len(result.inserted_ids)))

# ------------------------------------------------------------------
# 4. BUAT INDEX (agar query baca / aplikasi web cepat)
# ------------------------------------------------------------------
print("\n=== 4. MEMBUAT INDEX ===")
coll.create_index("id_artikel", unique=True)
coll.create_index("kategori_prediksi")
coll.create_index("benar")
coll.create_index([("kategori", 1), ("kategori_prediksi", 1)])
for nama, info in coll.index_information().items():
    print("  index: " + nama + " -> " + str(info.get("key")))

# ------------------------------------------------------------------
# 5. OPERASI BACA: verifikasi (count + 1 dokumen sampel)
# ------------------------------------------------------------------
print("\n=== 5. OPERASI BACA (VERIFIKASI) ===")
total_coll  = coll.count_documents({})
total_benar = coll.count_documents({"benar": True})
total_salah = coll.count_documents({"benar": False})
print("Jumlah dokumen di collection : " + str(total_coll))
print("Jumlah prediksi BENAR        : " + str(total_benar))
print("Jumlah prediksi SALAH        : " + str(total_salah))
if total_coll:
    akurasi = total_benar / total_coll
    print("Akurasi dari MongoDB         : " + format(akurasi, ".4f"))

print("\n1 dokumen sampel:")
sample = coll.find_one()
import json
print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))

client.close()
spark.stop()
print("\nSelesai. Data prediksi DEDUP berhasil dipindahkan dari HDFS (Spark) ke MongoDB.")
