# ==============================================================================
# BAGIAN C: SPARK MLlib — KLASIFIKASI BERITA INDONESIA DENGAN NAIVE BAYES
# ------------------------------------------------------------------------------
# Alur (sesuai panduan: HDFS -> Spark -> MLlib -> Evaluasi):
#   1. Baca data bersih (Parquet) dari HDFS
#   2. Ekstraksi fitur teks: Tokenizer -> HashingTF -> IDF (TF-IDF)
#   3. Enkoding label: kategori -> StringIndexer (0..4)
#   4. Split train/test 80/20 (seed tetap agar eksperimen bisa diulang)
#   5. Latih model NaiveBayes (multinomial)
#   6. Evaluasi: accuracy, precision, recall, F1 (per kelas + rata-rata)
#      + confusion matrix
#   7. Simpan hasil prediksi & model ke HDFS (titik koneksi ke MongoDB)
# ==============================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SEED = 42  # seed tetap -> hasil split bisa direplikasi (kriteria: keterulangan eksperimen)

PATH_PARQUET  = "hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet"
PATH_PREDIKSI = "hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi.parquet"
PATH_MODEL    = "hdfs://namenode:9000/proyek_bigdata/model/naive_bayes_model"

# ==============================================================================
# 1. INISIALISASI SPARK SESSION
# ==============================================================================
spark = (
    SparkSession.builder.appName("BagianC_MLlib_NaiveBayes_KlasifikasiBerita")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ==============================================================================
# 2. MEMBACA DATA BERSIH DARI HDFS
# ==============================================================================
print("\n=== 1. MEMBACA DATA BERSIH (PARQUET) DARI HDFS ===")
df = spark.read.parquet(PATH_PARQUET)
df.printSchema()
print(f"Total baris: {df.count()}")
print("\nDistribusi label (kategori):")
df.select("kategori").groupBy("kategori").count().orderBy("kategori").show()

# ==============================================================================
# 3. EKSTRAKSI FITUR TEKS (TF-IDF)
#    - Tokenizer : pecah teks_bersih menjadi array kata
#    - HashingTF : array kata -> vektor frekuensi (numFeatures=100000,
#                  tanpa perlu membangun vocabulary eksplisit -> efisien
#                  untuk data teks besar)
#    - IDF       : memberi bobot invers; kata yang umum di semua dokumen
#                  (mis. "berita", "antara") bobotnya ditekan
# ==============================================================================
print("\n=== 2. EKSTRAKSI FITUR: Tokenizer -> HashingTF -> IDF ===")
tokenizer  = Tokenizer(inputCol="teks_bersih", outputCol="kata")
hashing_tf = HashingTF(inputCol="kata", outputCol="tf", numFeatures=100000)
idf        = IDF(inputCol="tf", outputCol="fitur")

df_subset = df.select("id_artikel", "kategori", "teks_bersih")
df_kata   = tokenizer.transform(df_subset)
df_tf     = hashing_tf.transform(df_kata)
idf_model = idf.fit(df_tf)
df_fitur  = idf_model.transform(df_tf)

# ==============================================================================
# 4. ENKODING LABEL (kategori -> indeks numerik 0..4)
# ==============================================================================
print("\n=== 3. ENKODING LABEL: StringIndexer ===")
indexer     = StringIndexer(inputCol="kategori", outputCol="label")
indexer_model = indexer.fit(df_fitur)
df_final = indexer_model.transform(df_fitur)
print("Peta indeks -> kategori:")
for idx, nama in enumerate(indexer_model.labels):
    print(f"  {idx} -> {nama}")

# ==============================================================================
# 5. SPLIT TRAIN/TEST 80/20
# ==============================================================================
print("\n=== 4. SPLIT TRAIN/TEST 80/20 (seed=42) ===")
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=SEED)
print(f"Baris train : {train_df.count()}")
print(f"Baris test  : {test_df.count()}")

# ==============================================================================
# 6. LATIH MODEL NAIVE BAYES (multinomial)
#    Multinomial cocok untuk data frekuensi kata (TF-IDF), sesuai rekomendasi
#    panduan untuk klasifikasi dokumen.
# ==============================================================================
print("\n=== 5. PELATIHAN MODEL: NaiveBayes (multinomial) ===")
nb    = NaiveBayes(featuresCol="fitur", labelCol="label", modelType="multinomial")
model = nb.fit(train_df)

# ==============================================================================
# 7. PREDIKSI PADA DATA TEST
# ==============================================================================
print("\n=== 6. PREDIKSI PADA DATA TEST ===")
prediksi = model.transform(test_df)

# ==============================================================================
# 8. EVALUASI
#    - Accuracy        : proporsi prediksi benar secara keseluruhan
#    - Precision/Recall/F1: per kelas + rata-rata tertimbang (weighted)
#    - Confusion matrix: matriks 5x5 untuk melihat kategori yang sering
#      tertukar
# ==============================================================================
print("\n=== 7. EVALUASI MODEL ===")

# 8a. Accuracy
acc_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_eval.evaluate(prediksi)
print(f"\n[Accuracy]  : {accuracy:.4f}  ({accuracy*100:.2f}%)")

# 8b. Precision / Recall / F1 (rata-rata tertimbang)
#     Catatan: di Spark 3.5 nama metriknya "weightedPrecision" /
#     "weightedRecall" / "weightedFMeasure" (bukan "weighted avg ...")
print("\n[Metrik rata-rata tertimbang (weighted):]")
for metrik, nama_api in [("precision", "weightedPrecision"),
                         ("recall", "weightedRecall"),
                         ("f1", "weightedFMeasure")]:
    ev = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName=nama_api)
    print(f"  {nama_api:18s}: {ev.evaluate(prediksi):.4f}")

# 8c. Precision / Recall / F1 per kelas
#     Di Spark 3.5 metrik per kelas memakai metricName="...ByLabel" +
#     parameter metricLabel (indeks kelas)
print("\n[Metrik per kategori]:")
rows = []
for idx, nama in enumerate(indexer_model.labels):
    p = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                metricName="precisionByLabel", metricLabel=idx).evaluate(prediksi)
    r = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                metricName="recallByLabel", metricLabel=idx).evaluate(prediksi)
    f = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                metricName="fMeasureByLabel", metricLabel=idx).evaluate(prediksi)
    rows.append((nama, p, r, f))
df_per_kelas = (spark.createDataFrame(rows, ["kategori", "precision", "recall", "f1"])
                .withColumn("precision", col("precision").cast("double"))
                .withColumn("recall", col("recall").cast("double"))
                .withColumn("f1", col("f1").cast("double"))
                .orderBy("f1", ascending=False))
df_per_kelas.show()

# 8d. Confusion matrix (baris = label sebenarnya, kolom = prediksi)
#     Catatan: DataFrame.pivot dihapus di Spark 3.5, jadi matriks 5x5
#     dibangun manual dari hasil groupBy (hanya 25 sel, ringan)
print("[Confusion Matrix] (baris = kategori sebenarnya, kolom = prediksi):")
cm = (prediksi.groupBy("label", "prediction").count()
            .orderBy("label", "prediction"))
n = len(indexer_model.labels)
matrix = [[0] * n for _ in range(n)]
for row in cm.collect():
    matrix[int(row["label"])][int(row["prediction"])] = row["count"]
label_header = "sebenarnya/prediksi"
print(f"{label_header:<18s} | " + " | ".join(f"{nama[:10]:>10s}" for nama in indexer_model.labels))
print("-" * 90)
for i, nama in enumerate(indexer_model.labels):
    print(f"{nama:<18s} | " + " | ".join(f"{v:10d}" for v in matrix[i]))

# 8e. Contoh prediksi (5 benar & 5 salah) — untuk interpretasi
print("\n[Contoh prediksi BENAR (5 baris)]:")
prediksi.filter(col("label") == col("prediction")) \
    .select("id_artikel", "kategori", "prediction") \
    .show(5, truncate=False)

print("[Contoh prediksi SALAH (5 baris)]:")
prediksi.filter(col("label") != col("prediction")) \
    .select("id_artikel", "kategori", "prediction") \
    .show(5, truncate=False)

# ==============================================================================
# 9. SIMPAN HASIL PREDIKSI & MODEL KE HDFS
#    - Parquet prediksi: titik koneksi ke MongoDB (dikerjakan oleh teman NoSQL)
#    - Model: bisa dimuat kembali untuk prediksi artikel baru
# ==============================================================================
print("\n=== 8. SIMPAN HASIL KE HDFS ===")
# Catatan: "probability" & "rawPrediction" pada NaiveBayes adalah SparseVector
# (struct), sulit di-ekstrak per kelas via SQL. Kolom confidence sengaja
# tidak disimpan; output inti = id_artikel, kategori (sebenarnya),
# kategori_prediksi. (Confidence per kelas bisa ditambahkan bila diperlukan.)
# pemetaan indeks prediksi -> nama kategori via when() (5 kelas, ringan)
from pyspark.sql.functions import when
expr_kategori = when(col("prediction") == 0, indexer_model.labels[0])
for idx in range(1, len(indexer_model.labels)):
    expr_kategori = expr_kategori.when(col("prediction") == idx, indexer_model.labels[idx])
prediksi_final = (
    prediksi.select("id_artikel", "kategori", "label", "prediction")
    .withColumn("kategori_prediksi", expr_kategori)
    .drop("label", "prediction")
    .orderBy("id_artikel")
)
prediksi_final.write.mode("overwrite").parquet(PATH_PREDIKSI)
print(f"Prediksi test set disimpan ke : {PATH_PREDIKSI}")
print(f"Jumlah baris prediksi disimpan: {prediksi_final.count()}")

model.write().overwrite().save(PATH_MODEL)
print(f"Model NaiveBayes disimpan ke  : {PATH_MODEL}")

# ==============================================================================
# 10. RINGKASAN
# ==============================================================================
print("\n=== 9. RINGKASAN ===")
print(f"Data test        : {test_df.count()} artikel")
print(f"Accuracy         : {accuracy*100:.2f}%")
print(f"Kelas terbanyak  : {df_per_kelas.first().asDict()['kategori']} (F1 tertinggi)")
print("Evaluasi selesai. Hasil siap diinterpretasi & (opsional) dikirim ke MongoDB.")

spark.stop()
