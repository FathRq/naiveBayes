# ==============================================================================
# SPARK MLlib NAIVE BAYES - VERSI LENGKAP (RAW + DEDUP + JSON)
# ------------------------------------------------------------------------------
# SATU SKRIP, SATU PERINTAH: menjalankan SKENARIO EKSPERIMEN lengkap sesuai
# panduan (Tahap 6 Evaluasi: "berdasarkan skenario eksperimen"):
#
#   Skenario A (RAW)   : semua baris, TANPA dedup  -> baseline
#   Skenario B (DEDUP) : buang near-duplikat        -> ANGKA RESMI
#   Perbandingan       : selisih akurasi -> bukti bebas-leakage
#
# Output:
#   [1] 1 file JSON "ringkasan analisis"
#       (isi: raw + dedup + perbandingan + interpretasi)
#       disimpan ke: file lokal + HDFS + MongoDB
#   [2] 2 Parquet prediksi  (raw + dedup)
#   [3] 1 model Naive Bayes (versi dedup, yang resmi)
#
# CARA JALANKAN:
#   docker exec spark-master /opt/spark/bin/spark-submit /scripts/mllib_naive_bayes_lengkap.py
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
PATH_PREDIKSI_RAW     = "hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi.parquet"
PATH_PREDIKSI_DEDUP   = "hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi_dedup.parquet"
PATH_MODEL            = "hdfs://namenode:9000/proyek_bigdata/model/naive_bayes_model"
PATH_HDFS_JSON        = "/proyek_bigdata/analisis/ringkasan_analisis.json"
PATH_LOCAL_JSON       = "/data/analisis/ringkasan_analisis.json"

MONGO_URI  = "mongodb://mongodb:27017"
MONGO_DB   = "proyek_bigdata"
MONGO_COLL = "ringkasan_analisis"

MODEL_NAME = "Naive Bayes (multinomial)"
SUMBER     = "Spark MLlib"

spark = (
    SparkSession.builder.appName("MLlib_NaiveBayes_LENGKAP")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------------
# FUNGSI: pipeline TF-IDF -> StringIndexer -> split -> NaiveBayes -> evaluasi
# Dipakai 2x (skenario RAW dan DEDUP) agar kode tidak duplikat.
# ------------------------------------------------------------------
def jalankan_pipeline(df, nama_skenario):
    print(f"\n--- Skenario: {nama_skenario} ---")

    # (1) Fitur: TF-IDF dari teks_bersih (judul + isi). TIDAK dari kategori.
    tokenizer  = Tokenizer(inputCol="teks_bersih", outputCol="kata")
    hashing_tf = HashingTF(inputCol="kata", outputCol="tf", numFeatures=100000)
    idf        = IDF(inputCol="tf", outputCol="fitur")
    df_kata   = tokenizer.transform(df)
    df_tf     = hashing_tf.transform(df_kata)
    idf_model = idf.fit(df_tf)
    df_fitur  = idf_model.transform(df_tf)

    # (2) Label: kategori -> indeks numerik (kunci jawaban)
    indexer       = StringIndexer(inputCol="kategori", outputCol="label")
    indexer_model = indexer.fit(df_fitur)
    df_final      = indexer_model.transform(df_fitur)
    labels = list(indexer_model.labels)

    # (3) Split 80/20 (seed=42) + train
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=SEED)
    n_train = train_df.count()
    n_test  = test_df.count()
    nb    = NaiveBayes(featuresCol="fitur", labelCol="label", modelType="multinomial")
    model = nb.fit(train_df)
    prediksi = model.transform(test_df)

    # (4) Evaluasi: accuracy + weighted P/R/F1
    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(prediksi)
    weighted = {}
    for m, api in [("precision", "weightedPrecision"),
                   ("recall",    "weightedRecall"),
                   ("f1",        "weightedFMeasure")]:
        weighted[m] = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=api).evaluate(prediksi)

    # (5) Metrik per kategori (precision/recall/F1 by label)
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

    # (6) Confusion matrix
    cm = prediksi.groupBy("label", "prediction").count().orderBy("label", "prediction").collect()
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for row in cm:
        matrix[int(row["label"])][int(row["prediction"])] = int(row["count"])

    # (7) Siapkan DataFrame prediksi (untuk disimpan ke Parquet)
    def map_idx(c):
        expr = when(col(c) == 0, labels[0])
        for i in range(1, len(labels)):
            expr = expr.when(col(c) == i, labels[i])
        return expr.otherwise(col(c).cast("string"))
    df_pred = (prediksi
               .withColumn("kategori_prediksi", map_idx("prediction"))
               .select("id_artikel", "kategori", "kategori_prediksi"))

    print(f"  baris train={n_train}  test={n_test}  accuracy={acc:.4f}")
    return {
        "model"             : model,
        "labels"            : labels,
        "n_train"           : int(n_train),
        "n_test"            : int(n_test),
        "evaluasi"          : {"accuracy": round(acc, 4),
                               "weighted_precision": round(weighted["precision"], 4),
                               "weighted_recall":    round(weighted["recall"], 4),
                               "weighted_f1":        round(weighted["f1"], 4)},
        "metrik_per_kategori": per_kelas,
        "confusion_matrix"  : {"keterangan": "baris=sebenarnya, kolom=prediksi",
                               "label": labels, "matriks": matrix},
        "df_pred"           : df_pred,
    }


# ==============================================================================
# 1. BACA DATA
# ==============================================================================
print("\n=== 1. MEMBACA DATA ===")
df = spark.read.parquet(PATH_PARQUET)
n_raw = df.count()
print(f"Total baris (raw) : {n_raw}")


# ==============================================================================
# 2. SKENARIO A: RAW (tanpa dedup) -> baseline
# ==============================================================================
print("\n=== 2. SKENARIO A: RAW (tanpa dedup) ===")
res_raw = jalankan_pipeline(df.select("id_artikel", "kategori", "teks_bersih"), "RAW")
res_raw["df_pred"].write.mode("overwrite").parquet(PATH_PREDIKSI_RAW)
print(f"  Prediksi RAW disimpan : {PATH_PREDIKSI_RAW}")


# ==============================================================================
# 3. SKENARIO B: DEDUP (buang near-duplikat) -> ANGKA RESMI
# ==============================================================================
print("\n=== 3. SKENARIO B: DEDUP (buang near-duplikat) ===")
df_sig = df.withColumn("signature", substring(col("teks_bersih"), 1, 280))
w = Window.partitionBy("signature").orderBy(col("id_artikel"))
df_dedup = (df_sig
            .withColumn("rn", row_number().over(w))
            .filter(col("rn") == 1)
            .drop("signature", "rn"))
n_dedup = df_dedup.count()
print(f"Total baris (dedup) : {n_dedup}  (dibuang {n_raw - n_dedup} near-duplikat)")

res_dedup = jalankan_pipeline(df_dedup.select("id_artikel", "kategori", "teks_bersih"), "DEDUP")
res_dedup["df_pred"].write.mode("overwrite").parquet(PATH_PREDIKSI_DEDUP)
print(f"  Prediksi DEDUP disimpan: {PATH_PREDIKSI_DEDUP}")
res_dedup["model"].write().overwrite().save(PATH_MODEL)
print(f"  Model DEDUP disimpan   : {PATH_MODEL}")


# ==============================================================================
# 4. PERBANDINGAN RAW vs DEDUP
# ==============================================================================
print("\n=== 4. PERBANDINGAN RAW vs DEDUP ===")
selisih = abs(res_raw["evaluasi"]["accuracy"] - res_dedup["evaluasi"]["accuracy"])
print(f"  RAW   : {res_raw['evaluasi']['accuracy']*100:.2f}%")
print(f"  DEDUP : {res_dedup['evaluasi']['accuracy']*100:.2f}%")
print(f"  Selisih: {selisih*100:.2f} poin")
if selisih < 0.01:
    kesimpulan = ("Selisih < 1%: akurasi TIDAK tergelembung oleh near-duplikat; "
                 "angka DEDUP sah sebagai angka resmi (bebas-leakage).")
else:
    kesimpulan = ("Selisih signifikan: near-duplikat menggelembungkan akurasi; "
                 "gunakan angka DEDUP sebagai angka resmi.")
print(f"  -> {kesimpulan}")

# Interpretasi (dihitung dari skenario DEDUP = angka resmi)
per  = res_dedup["metrik_per_kategori"]
terkuat, terlemah = per[0], per[-1]
mat, labs = res_dedup["confusion_matrix"]["matriks"], res_dedup["confusion_matrix"]["label"]
max_off, max_pair = 0, ("?", "?")
for i in range(len(labs)):
    for j in range(len(labs)):
        if i != j and mat[i][j] > max_off:
            max_off, max_pair = mat[i][j], (labs[i], labs[j])
interpretasi = {
    "ringkasan": (
        "Model Naive Bayes (multinomial) dengan fitur TF-IDF mencapai akurasi "
        f"{res_dedup['evaluasi']['accuracy']*100:.2f}% pada {res_dedup['n_test']} artikel uji "
        f"(skenario DEDUP, bebas-leakage). Kategori '{terkuat['kategori']}' terklasifikasi "
        f"paling baik (F1={terkuat['f1']}), sedangkan '{terlemah['kategori']}' paling menantang "
        f"(F1={terlemah['f1']}). Kesalahan terbesar: berita '{max_pair[0]}' ditebak "
        f"'{max_pair[1]}' ({max_off} artikel), yang masuk akal karena kedua topik sering beririsan."
    ),
    "kategori_terkuat"  : terkuat["kategori"],
    "kategori_terlemah" : terlemah["kategori"],
    "paling_sering_salah": f"{max_pair[0]} -> {max_pair[1]} ({max_off} artikel)"
}


# ==============================================================================
# 5. BANGUN DOKUMEN JSON (raw + dedup + perbandingan + interpretasi)
# ==============================================================================
print("\n=== 5. BANGUN DOKUMEN JSON ===")
keseimbangan = {r["kategori"]: int(r["count"])
                for r in df_dedup.groupBy("kategori").count().collect()}
ringkasan = {
    "proyek": {
        "judul"     : "Klasifikasi Berita Indonesia Menggunakan Naive Bayes pada Spark MLlib",
        "topik"     : "Klasifikasi berita Indonesia (Topik 4)",
        "algoritma" : MODEL_NAME,
        "sumber"    : SUMBER,
        "arsitektur": "HDFS -> Spark -> Spark MLlib -> Evaluasi -> NoSQL (MongoDB)"
    },
    "dataset": {
        "nama_file"           : "berita_antara_final.csv",
        "sumber"              : "ANTARA News",
        "total_baris_raw"     : int(n_raw),
        "total_baris_dedup"   : int(n_dedup),
        "near_duplikat_dibuang": int(n_raw - n_dedup),
        "kategori"            : res_dedup["labels"],
        "keseimbangan_kelas"  : keseimbangan
    },
    "model": {
        "nama"      : "Naive Bayes",
        "tipe"      : "multinomial",
        "parameter" : {"smoothing": 1.0, "modelType": "multinomial"},
        "fitur"     : "TF-IDF: Tokenizer -> HashingTF(100000) -> IDF",
        "fitur_dari": "teks_bersih (judul + isi)",
        "catatan"   : "Model dilatih dari nol (fit); 100% parameter default, tanpa tuning"
    },
    "eksperimen": {
        "skenario": "RAW vs DEDUP (uji kebocoran near-duplikat)",
        "raw": {
            "total_baris" : int(n_raw),
            "split"       : {"train": res_raw["n_train"], "test": res_raw["n_test"], "seed": SEED},
            "evaluasi"    : res_raw["evaluasi"],
            "metrik_per_kategori": res_raw["metrik_per_kategori"],
            "confusion_matrix"   : res_raw["confusion_matrix"]
        },
        "dedup": {
            "total_baris"         : int(n_dedup),
            "near_duplikat_dibuang": int(n_raw - n_dedup),
            "split"               : {"train": res_dedup["n_train"], "test": res_dedup["n_test"], "seed": SEED},
            "evaluasi"            : res_dedup["evaluasi"],
            "metrik_per_kategori" : res_dedup["metrik_per_kategori"],
            "confusion_matrix"    : res_dedup["confusion_matrix"]
        },
        "perbandingan": {
            "selisih_accuracy": round(selisih, 4),
            "kesimpulan"      : kesimpulan
        }
    },
    "interpretasi": interpretasi
}
json_str = json.dumps(ringkasan, indent=2, ensure_ascii=False)
print(json_str)


# ==============================================================================
# 6. SIMPAN (file lokal + HDFS + MongoDB)
# ==============================================================================
print("\n=== 6. SIMPAN ===")

# [1] File lokal (container /data -> host ./data)
os.makedirs(os.path.dirname(PATH_LOCAL_JSON), exist_ok=True)
with open(PATH_LOCAL_JSON, "w", encoding="utf-8") as f:
    f.write(json_str)
print(f"[1] File lokal : {PATH_LOCAL_JSON}")

# [2] HDFS
try:
    jvm  = spark.sparkContext._jvm
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs   = jvm.org.apache.hadoop.fs.FileSystem.get(conf)
    fs.mkdirs(jvm.org.apache.hadoop.fs.Path("/proyek_bigdata/analisis"))
    out  = fs.create(jvm.org.apache.hadoop.fs.Path(PATH_HDFS_JSON), True)
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
    print(f"[3] MongoDB  : {MONGO_DB}.{MONGO_COLL}  ({coll.count_documents({})} dokumen)")
    doc = coll.find_one()
    print(f"    Verifikasi: accuracy_dedup={doc['eksperimen']['dedup']['evaluasi']['accuracy']}, "
          f"selisih={doc['eksperimen']['perbandingan']['selisih_accuracy']}")
    client.close()
except ImportError:
    print("[3] MongoDB  : SKIP (pymongo belum terinstall di container)")
    print("    Install dulu:  docker exec -u root spark-master pip install pymongo")

print("\nSelesai. Eksperimen RAW+DEDUP + JSON ringkasan analisis berhasil.")
spark.stop()
