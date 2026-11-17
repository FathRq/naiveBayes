# ==============================================================================
# HASIL ANALISIS -> FILE JSON  (untuk NoSQL / MongoDB)
# ------------------------------------------------------------------------------
# Sesuai panduan (bagian 6.D NoSQL):
#   "MongoDB dapat digunakan untuk menyimpan dokumen JSON, RINGKASAN ANALISIS,
#    atau hasil prediksi yang akan dibaca kembali oleh aplikasi."
#
# Skrip ini MENGHASILKAN file JSON "ringkasan analisis" yang memuat:
#   - info proyek, dataset, model, split
#   - evaluasi  : accuracy, weighted precision/recall/F1
#   - metrik per kategori (precision, recall, F1)
#   - confusion matrix
#   - interpretasi (makna hasil, bukan hanya angka)
#
# File JSON disimpan ke 3 tempat:
#   [1] File lokal : /data/analisis/ringkasan_analisis.json
#                    (= ./data/analisis/ringkasan_analisis.json di host)
#   [2] HDFS       : /proyek_bigdata/analisis/ringkasan_analisis.json
#   [3] MongoDB    : db=proyek_bigdata, collection=ringkasan_analisis
#
# Pipeline IDENTIK dengan run dedup resmi (angka 91.27%).
#
# CARA JALANKAN:
#   docker exec spark-master /opt/spark/bin/spark-submit /scripts/hasil_analisis_json.py
# ==============================================================================
import json
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, row_number, when
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SEED = 42
PATH_PARQUET          = "hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet"
PATH_PREDIKSI_DEDUP   = "hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi_dedup.parquet"
PATH_HDFS_JSON        = "/proyek_bigdata/analisis/ringkasan_analisis.json"
PATH_LOCAL_JSON       = "/data/analisis/ringkasan_analisis.json"

MONGO_URI  = "mongodb://mongodb:27017"
MONGO_DB   = "proyek_bigdata"
MONGO_COLL = "ringkasan_analisis"

MODEL_NAME = "Naive Bayes (multinomial)"
SUMBER     = "Spark MLlib"

spark = (
    SparkSession.builder.appName("Hasil_Analisis_JSON")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------
# 1. BACA + DEDUP (identik dengan run resmi)
# ------------------------------------------------------------------
print("\n=== 1. MEMBACA + DEDUP ===")
df = spark.read.parquet(PATH_PARQUET)
n_sebelum = df.count()
df_sig = df.withColumn("signature", substring(col("teks_bersih"), 1, 280))
w = Window.partitionBy("signature").orderBy(col("id_artikel"))
df_dedup = (df_sig
            .withColumn("rn", row_number().over(w))
            .filter(col("rn") == 1)
            .drop("signature", "rn"))
n_sesudah = df_dedup.count()
print(f"Sebelum dedup : {n_sebelum}")
print(f"Setelah dedup : {n_sesudah}  (dibuang {n_sebelum - n_sesudah} near-duplikat)")

# keseimbangan kelas
df_kelas = df_dedup.groupBy("kategori").count().orderBy("kategori")
keseimbangan = {r["kategori"]: int(r["count"]) for r in df_kelas.collect()}
print("Keseimbangan kelas:", keseimbangan)

# ------------------------------------------------------------------
# 2. EKSTRAKSI FITUR (identik)
# ------------------------------------------------------------------
print("\n=== 2. EKSTRAKSI FITUR (TF-IDF) ===")
tokenizer  = Tokenizer(inputCol="teks_bersih", outputCol="kata")
hashing_tf = HashingTF(inputCol="kata", outputCol="tf", numFeatures=100000)
idf        = IDF(inputCol="tf", outputCol="fitur")

df_subset = df_dedup.select("id_artikel", "kategori", "teks_bersih")
df_kata   = tokenizer.transform(df_subset)
df_tf     = hashing_tf.transform(df_kata)
idf_model = idf.fit(df_tf)
df_fitur  = idf_model.transform(df_tf)

indexer       = StringIndexer(inputCol="kategori", outputCol="label")
indexer_model = indexer.fit(df_fitur)
df_final      = indexer_model.transform(df_fitur)
labels = list(indexer_model.labels)
print("Peta indeks -> kategori:")
for i, nama in enumerate(labels):
    print(f"  {i} -> {nama}")

# ------------------------------------------------------------------
# 3. SPLIT 80/20 + TRAIN + PREDIKSI
# ------------------------------------------------------------------
print("\n=== 3. SPLIT 80/20 (seed=42) + TRAIN + PREDIKSI ===")
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=SEED)
n_train = train_df.count()
n_test  = test_df.count()
print(f"Baris train : {n_train}")
print(f"Baris test  : {n_test}")

nb    = NaiveBayes(featuresCol="fitur", labelCol="label", modelType="multinomial")
model = nb.fit(train_df)
prediksi = model.transform(test_df)

# bonus: simpan prediksi dedup (angka resmi) ke Parquet untuk tim NoSQL
def map_idx_to_nama(c):
    expr = when(col(c) == 0, labels[0])
    for i in range(1, len(labels)):
        expr = expr.when(col(c) == i, labels[i])
    return expr.otherwise(col(c).cast("string"))

df_prediksi_simpan = (
    prediksi
    .withColumn("kategori_prediksi", map_idx_to_nama("prediction"))
    .select("id_artikel", "kategori", "kategori_prediksi")
)
df_prediksi_simpan.write.mode("overwrite").parquet(PATH_PREDIKSI_DEDUP)
print(f"Prediksi dedup disimpan ke: {PATH_PREDIKSI_DEDUP}")

# ------------------------------------------------------------------
# 4. EVALUASI
# ------------------------------------------------------------------
print("\n=== 4. EVALUASI ===")
acc_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_eval.evaluate(prediksi)
print(f"[Accuracy] : {accuracy:.4f}  ({accuracy*100:.2f}%)")

weighted = {}
for metrik, nama_api in [("precision", "weightedPrecision"),
                         ("recall",    "weightedRecall"),
                         ("f1",        "weightedFMeasure")]:
    ev = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName=nama_api)
    weighted[metrik] = ev.evaluate(prediksi)
    print(f"  {nama_api:18s}: {weighted[metrik]:.4f}")

print("\n[Metrik per kategori]:")
per_kelas = []
for idx, nama in enumerate(labels):
    p = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
        metricName="precisionByLabel", metricLabel=idx).evaluate(prediksi)
    r = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
        metricName="recallByLabel", metricLabel=idx).evaluate(prediksi)
    f = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
        metricName="fMeasureByLabel", metricLabel=idx).evaluate(prediksi)
    per_kelas.append({"kategori": nama,
                      "precision": round(p, 4),
                      "recall":    round(r, 4),
                      "f1":        round(f, 4)})
per_kelas.sort(key=lambda x: x["f1"], reverse=True)
for kk in per_kelas:
    print(f"  {kk['kategori']:12s}  P={kk['precision']:.4f}  R={kk['recall']:.4f}  F1={kk['f1']:.4f}")

# confusion matrix
cm = prediksi.groupBy("label", "prediction").count().orderBy("label", "prediction").collect()
n = len(labels)
matrix = [[0] * n for _ in range(n)]
for row in cm:
    matrix[int(row["label"])][int(row["prediction"])] = int(row["count"])

# ------------------------------------------------------------------
# 5. INTERPRETASI (otomatis dari angka)
# ------------------------------------------------------------------
print("\n=== 5. INTERPRETASI ===")
terkuat  = per_kelas[0]
terlemah = per_kelas[-1]
max_off, max_pair = 0, ("?", "?")
for i in range(n):
    for j in range(n):
        if i != j and matrix[i][j] > max_off:
            max_off  = matrix[i][j]
            max_pair = (labels[i], labels[j])
print(f"Terkuat             : {terkuat['kategori']}  (F1={terkuat['f1']})")
print(f"Terlemah            : {terlemah['kategori']}  (F1={terlemah['f1']})")
print(f"Paling sering salah : {max_pair[0]} -> {max_pair[1]}  ({max_off} artikel)")

interpretasi = {
    "ringkasan": (
        "Model Naive Bayes (multinomial) dengan fitur TF-IDF mencapai akurasi "
        f"{accuracy*100:.2f}% pada {n_test} artikel uji. Kategori '{terkuat['kategori']}' "
        f"terklasifikasi paling baik (F1={terkuat['f1']}), sedangkan '{terlemah['kategori']}' "
        f"paling menantang (F1={terlemah['f1']}). Kesalahan terbesar terjadi ketika berita "
        f"'{max_pair[0]}' ditebak sebagai '{max_pair[1]}' ({max_off} artikel), yang masuk "
        "akal karena kedua topik sering membahas hal yang beririsan."
    ),
    "kategori_terkuat"  : terkuat["kategori"],
    "kategori_terlemah" : terlemah["kategori"],
    "paling_sering_salah": f"{max_pair[0]} -> {max_pair[1]} ({max_off} artikel)"
}

# ------------------------------------------------------------------
# 6. BANGUN DOKUMEN JSON (ringkasan analisis)
# ------------------------------------------------------------------
print("\n=== 6. BANGUN DOKUMEN JSON ===")
ringkasan = {
    "proyek": {
        "judul"     : "Klasifikasi Berita Indonesia Menggunakan Naive Bayes pada Spark MLlib",
        "topik"     : "Klasifikasi berita Indonesia (Topik 4)",
        "algoritma" : MODEL_NAME,
        "sumber"    : SUMBER,
        "arsitektur": "HDFS -> Spark -> Spark MLlib -> Evaluasi -> NoSQL (MongoDB)"
    },
    "dataset": {
        "nama_file"          : "berita_antara_final.csv",
        "sumber"             : "ANTARA News",
        "total_baris_raw"    : int(n_sebelum),
        "total_baris_dedup"  : int(n_sesudah),
        "near_duplikat_dibuang": int(n_sebelum - n_sesudah),
        "kategori"           : labels,
        "keseimbangan_kelas" : keseimbangan
    },
    "model": {
        "nama"      : "Naive Bayes",
        "tipe"      : "multinomial",
        "parameter" : {"smoothing": 1.0, "modelType": "multinomial"},
        "fitur"     : "TF-IDF: Tokenizer -> HashingTF(100000) -> IDF",
        "fitur_dari": "teks_bersih (judul + isi)",
        "catatan"   : "Model dilatih dari nol (fit) pada data latih; 100% parameter default, tanpa tuning"
    },
    "split": {
        "train_ratio": 0.8,
        "test_ratio" : 0.2,
        "seed"       : SEED,
        "baris_train": int(n_train),
        "baris_test" : int(n_test)
    },
    "evaluasi": {
        "accuracy"          : round(accuracy, 4),
        "weighted_precision": round(weighted["precision"], 4),
        "weighted_recall"   : round(weighted["recall"], 4),
        "weighted_f1"       : round(weighted["f1"], 4)
    },
    "metrik_per_kategori": per_kelas,
    "confusion_matrix": {
        "keterangan": "baris = label sebenarnya, kolom = prediksi",
        "label"     : labels,
        "matriks"   : matrix
    },
    "interpretasi": interpretasi
}

json_str = json.dumps(ringkasan, indent=2, ensure_ascii=False)
print("Dokumen JSON (ringkasan analisis):")
print(json_str)

# ------------------------------------------------------------------
# 7. SIMPAN KE 3 TEMPAT
# ------------------------------------------------------------------
print("\n=== 7. SIMPAN ===")

# [1] File lokal (container /data -> host ./data)
os.makedirs(os.path.dirname(PATH_LOCAL_JSON), exist_ok=True)
with open(PATH_LOCAL_JSON, "w", encoding="utf-8") as f:
    f.write(json_str)
print(f"[1] File lokal : {PATH_LOCAL_JSON}")

# [2] HDFS
try:
    jvm   = spark.sparkContext._jvm
    conf  = spark.sparkContext._jsc.hadoopConfiguration()  # conf Spark (sudah punya fs.defaultFS=hdfs://namenode:9000)
    fs    = jvm.org.apache.hadoop.fs.FileSystem.get(conf)
    fs.mkdirs(jvm.org.apache.hadoop.fs.Path("/proyek_bigdata/analisis"))
    out   = fs.create(jvm.org.apache.hadoop.fs.Path(PATH_HDFS_JSON), True)
    out.write(json_str.encode("utf-8"))
    out.close()
    print(f"[2] HDFS     : hdfs://namenode:9000{PATH_HDFS_JSON}")
except Exception as e:
    print(f"[2] HDFS     : GAGAL - {e}")

# [3] MongoDB (graceful: skip sopan jika pymongo belum ada)
try:
    import pymongo
    client = pymongo.MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    coll   = db[MONGO_COLL]
    coll.delete_many({})
    coll.insert_one(ringkasan)
    total = coll.count_documents({})
    print(f"[3] MongoDB  : {MONGO_DB}.{MONGO_COLL}  ({total} dokumen)")
    # verifikasi: baca kembali
    doc = coll.find_one()
    print("    Verifikasi (baca kembali) - field evaluasi:")
    print(f"      accuracy          = {doc['evaluasi']['accuracy']}")
    print(f"      weighted_f1       = {doc['evaluasi']['weighted_f1']}")
    print(f"      kategori_terkuat  = {doc['interpretasi']['kategori_terkuat']}")
    client.close()
except ImportError:
    print("[3] MongoDB  : SKIP (pymongo belum terinstall di container)")
    print("    Install dulu:  docker exec -u root spark-master pip install pymongo")

print("\nSelesai. Ringkasan analisis berhasil di-generate & disimpan.")
spark.stop()
