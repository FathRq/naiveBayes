# ==============================================================================
# OPERASI BACA MONGODB (NoSQL) — skrip mandiri, TANPA Spark
# ------------------------------------------------------------------------------
# Skrip ini adalah pasangan dari simpan_mongodb.py:
#   simpan_mongodb.py = OPERASI TULIS (Spark/HDFS -> MongoDB)
#   baca_mongodb.py   = OPERASI BACA  (MongoDB -> terminal / aplikasi)
#
# Sesuai Panduan Proyek Akhir Big Data 2026 bagian 6.D: jika memakai
# MongoDB, tunjukkan minimal satu operasi penyimpanan dan satu operasi
# pembacaan data. Skrip ini menunjukkan pembacaan dalam beberapa bentuk:
#   1. find_one()          : 1 dokumen ringkasan_analisis
#   2. count_documents()   : jumlah dokumen + akurasi dihitung dari MongoDB
#   3. aggregate()         : distribusi per kategori_prediksi, distribusi
#                            per kategori asli, dan pasangan salah prediksi
#                            terbanyak (pipeline agregasi khas NoSQL)
#   4. find() + limit      : contoh dokumen yang prediksinya salah
#
# CARA JALANKAN:
#   Di dalam container spark-master (hostname mongodb):
#     docker exec spark-master python3 /scripts/baca_mongodb.py
#   Dari host (MongoDB ter-map ke localhost:27017):
#     pip install pymongo
#     MONGO_URI=mongodb://localhost:27017 python baca_mongodb.py
#   (di Windows PowerShell: $env:MONGO_URI="mongodb://localhost:27017")
# ==============================================================================

import json
import os
import sys

try:
    import pymongo
except ImportError:
    print("pymongo belum terinstall. Jalankan: pip install pymongo")
    sys.exit(1)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB  = os.environ.get("MONGO_DB", "proyek_bigdata")

client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
except Exception as e:
    print("Tidak bisa terhubung ke MongoDB di " + MONGO_URI)
    print("Pastikan container mongodb-uas jalan (docker compose up -d).")
    print("Detail: " + str(e))
    sys.exit(1)

db = client[MONGO_DB]
print("Terhubung ke MongoDB: " + MONGO_URI + " / db=" + MONGO_DB)
print("Collections: " + ", ".join(db.list_collection_names()))

# ------------------------------------------------------------------
# 1. BACA ringkasan_analisis (find_one)
# ------------------------------------------------------------------
print("\n=== 1. RINGKASAN ANALISIS (collection: ringkasan_analisis) ===")
ringkasan = db["ringkasan_analisis"].find_one()
if ringkasan is None:
    print("Collection ringkasan_analisis masih kosong.")
    print("Jalankan dulu: spark-submit /scripts/mllib_naive_bayes_lengkap.py")
else:
    ev_raw   = ringkasan["eksperimen"]["raw"]["evaluasi"]
    ev_dedup = ringkasan["eksperimen"]["dedup"]["evaluasi"]
    print("Judul proyek : " + ringkasan["proyek"]["judul"])
    print("Dataset      : " + str(ringkasan["dataset"]["total_baris_raw"]) +
          " baris mentah, " + str(ringkasan["dataset"]["total_baris_dedup"]) +
          " baris setelah dedup")
    print("Accuracy RAW   : " + format(ev_raw["accuracy"], ".4f") +
          "  (F1 " + format(ev_raw["weighted_f1"], ".4f") + ")")
    print("Accuracy DEDUP : " + format(ev_dedup["accuracy"], ".4f") +
          "  (F1 " + format(ev_dedup["weighted_f1"], ".4f") + ")  <- angka resmi")
    print("\nMetrik per kategori (DEDUP):")
    print("  kategori     precision   recall       f1")
    for m in ringkasan["eksperimen"]["dedup"]["metrik_per_kategori"]:
        print("  " + m["kategori"].ljust(12) + " " +
              format(m["precision"], ".4f").ljust(11) + " " +
              format(m["recall"], ".4f").ljust(12) + " " +
              format(m["f1"], ".4f"))
    print("\nInterpretasi: " + ringkasan["interpretasi"]["ringkasan"])

# ------------------------------------------------------------------
# 2. BACA berita_prediksi (count + akurasi dihitung dari MongoDB)
# ------------------------------------------------------------------
print("\n=== 2. PREDIKSI PER ARTIKEL (collection: berita_prediksi) ===")
coll = db["berita_prediksi"]
total = coll.count_documents({})
if total == 0:
    print("Collection berita_prediksi masih kosong.")
    print("Jalankan dulu: spark-submit /scripts/simpan_mongodb.py")
    client.close()
    sys.exit(0)

benar = coll.count_documents({"benar": True})
salah = coll.count_documents({"benar": False})
print("Total dokumen : " + str(total))
print("Benar         : " + str(benar))
print("Salah         : " + str(salah))
print("Akurasi (dihitung dari MongoDB): " + format(benar / total, ".4f"))

# ------------------------------------------------------------------
# 3. AGREGASI: distribusi per kategori_prediksi
# ------------------------------------------------------------------
print("\n=== 3. AGREGASI: DISTRIBUSI PER KATEGORI PREDIKSI ===")
pipeline_pred = [
    {"$group": {"_id": "$kategori_prediksi", "jumlah": {"$sum": 1}}},
    {"$sort": {"jumlah": -1}},
]
for row in coll.aggregate(pipeline_pred):
    print("  " + str(row["_id"]).ljust(12) + ": " + str(row["jumlah"]))

print("\n=== 4. AGREGASI: PASANGAN SALAH PREDIKSI TERBANYAK ===")
pipeline_salah = [
    {"$match": {"benar": False}},
    {"$group": {"_id": {"asli": "$kategori", "prediksi": "$kategori_prediksi"},
                "jumlah": {"$sum": 1}}},
    {"$sort": {"jumlah": -1}},
    {"$limit": 5},
]
for row in coll.aggregate(pipeline_salah):
    print("  " + row["_id"]["asli"] + " -> " + row["_id"]["prediksi"] +
          " : " + str(row["jumlah"]) + " artikel")

# ------------------------------------------------------------------
# 4. BACA contoh dokumen salah prediksi (find + limit)
# ------------------------------------------------------------------
print("\n=== 5. CONTOH 5 DOKUMEN SALAH PREDIKSI ===")
for doc in coll.find({"benar": False}, {"_id": 0}).limit(5):
    print("  " + json.dumps(doc, ensure_ascii=False))

client.close()
print("\nSelesai. Semua data di atas dibaca langsung dari MongoDB (NoSQL).")
