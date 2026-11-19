# ==============================================================================
# BACKEND API DASHBOARD — Klasifikasi Berita Indonesia (Naive Bayes)
# ------------------------------------------------------------------------------
# Backend kecil (Flask) yang membaca hasil proyek dari MongoDB (NoSQL):
#   - collection ringkasan_analisis : 1 dokumen ringkasan (metrik, confusion
#                                     matrix, interpretasi)
#   - collection berita_prediksi    : prediksi per artikel (skenario DEDUP)
#
# Inilah "aplikasi" yang dimaksud Panduan 6.D: hasil prediksi yang disimpan
# di MongoDB dibaca kembali oleh aplikasi (dashboard web di web/static).
#
# FALLBACK: jika MongoDB belum terisi / tidak terjangkau, /api/ringkasan
# membaca file data/analisis/ringkasan_analisis.json dari repo, supaya
# dashboard tetap bisa didemokan. /api/status memberi tahu sumber aktif.
#
# ENDPOINT:
#   GET /                       -> dashboard (static/index.html)
#   GET /api/status             -> status koneksi + sumber data aktif
#   GET /api/ringkasan          -> dokumen ringkasan_analisis
#   GET /api/agregasi           -> distribusi prediksi + pasangan salah
#                                  prediksi terbanyak (agregasi MongoDB)
#   GET /api/prediksi           -> daftar prediksi per artikel, dengan
#                                  filter & pagination:
#                                  ?kategori=politik&prediksi=ekonomi
#                                  &benar=false&page=1&per_page=20
#
# JALANKAN:
#   Lewat Docker Compose (disarankan):  docker compose up -d --build web
#                                        lalu buka http://localhost:5000
#   Tanpa Docker (dari folder repo):
#     pip install -r web/backend/requirements.txt
#     MONGO_URI=mongodb://localhost:27017 python web/backend/app.py
# ==============================================================================

import json
import os

from flask import Flask, jsonify, request, send_from_directory

try:
    import pymongo
except ImportError:  # pymongo tidak ada -> mode fallback file JSON
    pymongo = None

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB  = os.environ.get("MONGO_DB", "proyek_bigdata")

# Lokasi file fallback ringkasan_analisis.json (dicoba berurutan)
FALLBACK_CANDIDATES = [
    os.environ.get("FALLBACK_JSON", ""),
    "/fallback/ringkasan_analisis.json",
    os.path.join(BASE_DIR, "..", "..", "data", "analisis", "ringkasan_analisis.json"),
]

app = Flask(__name__, static_folder=None)

_client = None


def get_db():
    """Kembalikan database MongoDB jika terjangkau, selain itu None."""
    global _client
    if pymongo is None:
        return None
    try:
        if _client is None:
            _client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        _client.admin.command("ping")
        return _client[MONGO_DB]
    except Exception:
        return None


def load_ringkasan_file():
    """Baca ringkasan_analisis.json dari file lokal (mode fallback)."""
    for path in FALLBACK_CANDIDATES:
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def get_ringkasan():
    """Ambil dokumen ringkasan: MongoDB dulu, fallback ke file JSON."""
    db = get_db()
    if db is not None:
        doc = db["ringkasan_analisis"].find_one({}, {"_id": 0})
        if doc:
            return doc, "MongoDB (collection ringkasan_analisis)"
    doc = load_ringkasan_file()
    if doc:
        return doc, "file JSON lokal (fallback — MongoDB belum terisi)"
    return None, None


# ------------------------------------------------------------------
# Halaman dashboard
# ------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(os.path.abspath(STATIC_DIR), "index.html")


@app.route("/static/<path:nama_file>")
def statis(nama_file):
    return send_from_directory(os.path.abspath(STATIC_DIR), nama_file)


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    db = get_db()
    terhubung = db is not None
    jumlah_prediksi = 0
    ada_ringkasan = False
    if terhubung:
        jumlah_prediksi = db["berita_prediksi"].count_documents({})
        ada_ringkasan = db["ringkasan_analisis"].count_documents({}) > 0
    if not ada_ringkasan:
        ada_ringkasan = load_ringkasan_file() is not None
    _, sumber_ringkasan = get_ringkasan()
    return jsonify({
        "mongodb_terhubung": terhubung,
        "mongo_uri_host": MONGO_URI.split("@")[-1],
        "database": MONGO_DB,
        "jumlah_dokumen_prediksi": jumlah_prediksi,
        "ringkasan_tersedia": ada_ringkasan,
        "sumber_ringkasan": sumber_ringkasan,
        "arsitektur": "Scraping ANTARA -> HDFS -> Spark -> MLlib Naive Bayes "
                      "-> Evaluasi -> MongoDB (NoSQL) -> Dashboard Web",
    })


@app.route("/api/ringkasan")
def api_ringkasan():
    doc, sumber = get_ringkasan()
    if doc is None:
        return jsonify({"error": "ringkasan_analisis belum tersedia. "
                                 "Jalankan mllib_naive_bayes_lengkap.py dulu."}), 404
    return jsonify({"sumber": sumber, "data": doc})


@app.route("/api/agregasi")
def api_agregasi():
    db = get_db()
    if db is None:
        return jsonify({"error": "MongoDB tidak terhubung — agregasi hanya "
                                 "tersedia dari MongoDB.",
                        "distribusi_prediksi": [],
                        "pasangan_salah_terbanyak": []}), 200
    coll = db["berita_prediksi"]
    distribusi = [
        {"kategori": r["_id"], "jumlah": r["jumlah"]}
        for r in coll.aggregate([
            {"$group": {"_id": "$kategori_prediksi", "jumlah": {"$sum": 1}}},
            {"$sort": {"jumlah": -1}},
        ])
    ]
    pasangan = [
        {"kategori": r["_id"]["asli"], "kategori_prediksi": r["_id"]["prediksi"],
         "jumlah": r["jumlah"]}
        for r in coll.aggregate([
            {"$match": {"benar": False}},
            {"$group": {"_id": {"asli": "$kategori",
                                "prediksi": "$kategori_prediksi"},
                        "jumlah": {"$sum": 1}}},
            {"$sort": {"jumlah": -1}},
            {"$limit": 5},
        ])
    ]
    total = coll.count_documents({})
    benar = coll.count_documents({"benar": True})
    return jsonify({
        "total_dokumen": total,
        "benar": benar,
        "salah": total - benar,
        "akurasi_dari_mongodb": (benar / total) if total else None,
        "distribusi_prediksi": distribusi,
        "pasangan_salah_terbanyak": pasangan,
    })


@app.route("/api/prediksi")
def api_prediksi():
    db = get_db()
    if db is None:
        return jsonify({
            "error": "MongoDB tidak terhubung. Jalankan cluster "
                     "(docker compose up -d) dan simpan_mongodb.py agar "
                     "prediksi per artikel tersedia.",
            "total": 0, "page": 1, "per_page": 0, "data": [],
        }), 200

    query = {}
    kategori = request.args.get("kategori")
    prediksi = request.args.get("prediksi")
    benar = request.args.get("benar")
    if kategori:
        query["kategori"] = kategori
    if prediksi:
        query["kategori_prediksi"] = prediksi
    if benar in ("true", "false"):
        query["benar"] = (benar == "true")

    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    except ValueError:
        page, per_page = 1, 20

    coll = db["berita_prediksi"]
    total = coll.count_documents(query)
    cursor = (coll.find(query, {"_id": 0})
              .sort("id_artikel", 1)
              .skip((page - 1) * per_page)
              .limit(per_page))
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_halaman": (total + per_page - 1) // per_page if total else 0,
        "data": list(cursor),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
