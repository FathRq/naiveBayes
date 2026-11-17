# ==============================================================================
# RE-RUN NAIVE BAYES dengan DATA DEDUPLICATED (split bebas-leakage)
# ------------------------------------------------------------------------------
# Tujuan: menjawab kecurigaan "akurasi terlalu tinggi".
#   1. Hitung signature = 40 kata pertama teks_bersih (~280 karakter)
#   2. Dedup: simpan hanya 1 baris per signature (buang near-duplikat/IHSG)
#   3. Pipeline IDENTIK dengan run sebelumnya (Tokenizer->HashingTF->IDF->
#      StringIndexer->split 80/20 seed=42->NaiveBayes multinomial)
#   4. Bandingkan akurasi: raw (91.69%) vs dedup (X%)
# Jika X ~= raw  -> angka TIDAK tergelembung (bela di laporan)
# ==============================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SEED = 42
PATH_PARQUET = "hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet"

spark = (
    SparkSession.builder.appName("MLlib_NaiveBayes_DEDUP")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------
# 1. BACA + DEDUP
# ------------------------------------------------------------------
print("\n=== 1. MEMBACA + DEDUP (signature = 40 kata pertama) ===")
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

# ------------------------------------------------------------------
# 2. PIPELINE IDENTIK (TF-IDF + StringIndexer)
# ------------------------------------------------------------------
print("\n=== 2. EKSTRAKSI FITUR (identik dengan run sebelumnya) ===")
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
print("Peta indeks -> kategori:")
for idx, nama in enumerate(indexer_model.labels):
    print(f"  {idx} -> {nama}")

# ------------------------------------------------------------------
# 3. SPLIT 80/20 (seed=42) + TRAIN + PREDIKSI
# ------------------------------------------------------------------
print("\n=== 3. SPLIT 80/20 + TRAIN + PREDIKSI ===")
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=SEED)
print(f"Baris train : {train_df.count()}")
print(f"Baris test  : {test_df.count()}")

nb    = NaiveBayes(featuresCol="fitur", labelCol="label", modelType="multinomial")
model = nb.fit(train_df)
prediksi = model.transform(test_df)

# ------------------------------------------------------------------
# 4. EVALUASI
# ------------------------------------------------------------------
print("\n=== 4. EVALUASI (DE-DUPLICATED) ===")
acc_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_eval.evaluate(prediksi)
print(f"\n[Accuracy]  : {accuracy:.4f}  ({accuracy*100:.2f}%)")

for metrik, nama_api in [("precision", "weightedPrecision"),
                         ("recall", "weightedRecall"),
                         ("f1", "weightedFMeasure")]:
    ev = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName=nama_api)
    print(f"  {nama_api:18s}: {ev.evaluate(prediksi):.4f}")

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

# confusion matrix
print("[Confusion Matrix] (baris = sebenarnya, kolom = prediksi):")
cm = prediksi.groupBy("label", "prediction").count().orderBy("label", "prediction")
n = len(indexer_model.labels)
matrix = [[0] * n for _ in range(n)]
for row in cm.collect():
    matrix[int(row["label"])][int(row["prediction"])] = row["count"]
label_header = "sebenarnya/prediksi"
print(f"{label_header:<18s} | " + " | ".join(f"{nama[:10]:>10s}" for nama in indexer_model.labels))
print("-" * 90)
for i, nama in enumerate(indexer_model.labels):
    print(f"{nama:<18s} | " + " | ".join(f"{v:10d}" for v in matrix[i]))

# ------------------------------------------------------------------
# 5. PERBANDINGAN
# ------------------------------------------------------------------
print("\n=== 5. PERBANDINGAN: RAW vs DEDUP ===")
print(f"  RAW   (random split, ada near-dup)  : 91.69%")
print(f"  DEDUP (bebas-leakage)               : {accuracy*100:.2f}%")
selisih = abs(accuracy*100 - 91.69)
print(f"  Selisih                             : {selisih:.2f} poin")
if selisih < 1.0:
    print("  -> SELISIH < 1%: akurasi TIDAK tergelembung, angka 91.69% SAH.")
else:
    print("  -> Selisih signifikan: near-duplikat memang menggelembungkan;")
    print("     gunakan angka DEDUP sebagai angka resmi di laporan.")

spark.stop()
