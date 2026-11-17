# ==============================================================================
# TITIK KONEKSI: SPARK -> MONGODB (opsional, dikerjakan oleh tim NoSQL)
# ------------------------------------------------------------------------------
# Skrip ini membaca hasil prediksi Naive Bayes (Parquet di HDFS) dan menyimpan
# -nya ke MongoDB sebagai dokumen JSON. Ini adalah "titik koneksi" antara
# bagian Spark/MLlib (kita) dengan bagian NoSQL (teman).
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
MONGO_URI      = "mongodb://mongodb:27017"   # hostname service compose = "mongodb"
MONGO_DB       = "proyek_bigdata"
MONGO_COLL     = "berita_prediksi"
PATH_PREDIKSI  = "hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi.parquet"
MODEL_NAME     = "Naive Bayes (multinomial)"
SUMBER         = "Spark MLlib"

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
df = spark.read.parquet(PATH_PREDIKSI)
df = (df
      .withColumn("benar", col("kategori") == col("kategori_prediksi"))
      .select("id_artikel", "kategori", "kategori_prediksi", "benar"))
total = df.count()
print(f"Total dokumen yang akan disimpan: {total}")

# ------------------------------------------------------------------
# 2. Convert DataFrame -> list dict (6.5k baris, aman di driver)
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
print(f"Dokumen disisipkan: {len(result.inserted_ids)}")

# ------------------------------------------------------------------
# 4. OPERASI BACA: verifikasi (count + 1 dokumen sampel)
# ------------------------------------------------------------------
print("\n=== 4. OPERASI BACA (VERIFIKASI) ===")
total_coll = coll.count_documents({})
total_benar = coll.count_documents({"benar": True})
total_salah = coll.count_documents({"benar": False})
print(f"Jumlah dokumen di collection : {total_coll}")
print(f"Jumlah prediksi BENAR        : {total_benar}")
print(f"Jumlah prediksi SALAH        : {total_salah}")
print("\n1 dokumen sampel:")
sample = coll.find_one()
import json
print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))

client.close()
spark.stop()
print("\nSelesai. Data prediksi berhasil dipindahkan dari HDFS (Spark) ke MongoDB.")
