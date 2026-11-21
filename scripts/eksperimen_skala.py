import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SEED = 42
PATH_PARQUET = "hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet"

spark = (
    SparkSession.builder.appName("Eksperimen_Skala_Data")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n=== MEMBACA & DEDUPLIKASI DATA (SKENARIO DEDUP) ===")
df = spark.read.parquet(PATH_PARQUET)
df_sig = df.withColumn("signature", substring(col("teks_bersih"), 1, 280))
w = Window.partitionBy("signature").orderBy(col("id_artikel"))
df_dedup = (
    df_sig.withColumn("rn", row_number().over(w))
    .filter(col("rn") == 1)
    .drop("signature", "rn")
    .select("id_artikel", "kategori", "teks_bersih")
)
df_dedup.cache()
total_dedup = df_dedup.count()
print(f"Total data DEDUP (100%): {total_dedup} artikel\n")

skala_list = [0.25, 0.50, 0.75, 1.00]
hasil_eksperimen = []

for frac in skala_list:
    persen = int(frac * 100)
    print(f"--- Menguji Skala {persen}% ---")
    
    if frac < 1.0:
        df_sub = df_dedup.sample(withReplacement=False, fraction=frac, seed=SEED)
    else:
        df_sub = df_dedup
        
    jml_artikel = df_sub.count()
    
    # Mulai ukur waktu proses ekstraksi fitur + pelatihan model
    t_start = time.time()
    
    tokenizer = Tokenizer(inputCol="teks_bersih", outputCol="kata")
    hashing_tf = HashingTF(inputCol="kata", outputCol="tf", numFeatures=100000)
    idf = IDF(inputCol="tf", outputCol="fitur")
    
    df_kata = tokenizer.transform(df_sub)
    df_tf = hashing_tf.transform(df_kata)
    idf_model = idf.fit(df_tf)
    df_fitur = idf_model.transform(df_tf)
    
    indexer = StringIndexer(inputCol="kategori", outputCol="label")
    df_final = indexer.fit(df_fitur).transform(df_fitur)
    
    train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=SEED)
    
    nb = NaiveBayes(featuresCol="fitur", labelCol="label", modelType="multinomial")
    model = nb.fit(train_df)
    
    waktu_latih = round(time.time() - t_start, 2)
    
    # Evaluasi
    prediksi = model.transform(test_df)
    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(prediksi)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedFMeasure"
    ).evaluate(prediksi)
    
    hasil_eksperimen.append((persen, jml_artikel, waktu_latih, round(acc * 100, 2), round(f1, 4)))
    print(f"  Artikel: {jml_artikel} | Waktu: {waktu_latih}s | Acc: {acc*100:.2f}% | F1: {f1:.4f}")

print("\n" + "=" * 65)
print("RINGKASAN TABEL VII (SKENARIO DEDUP)")
print("=" * 65)
print(f"{'Skala':<10} {'Jumlah Artikel':<16} {'Waktu (detik)':<16} {'Accuracy':<12} {'F1-Score':<10}")
print("-" * 65)
for h in hasil_eksperimen:
    print(f"{str(h[0])+'%':<10} {h[1]:<16} {h[2]:<16} {str(h[3])+'%':<12} {h[4]:<10}")
print("=" * 65)

spark.stop()
