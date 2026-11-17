# Cek cepat isi file prediksi (verifikasi hasil MLlib)
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("CekPrediksi")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet("hdfs://namenode:9000/proyek_bigdata/prediksi/berita_prediksi.parquet")
df.printSchema()
print(f"Total baris: {df.count()}")
print("\n5 baris sampel:")
df.show(5, truncate=False)
print("\nDistribusi kategori_prediksi:")
df.groupBy("kategori_prediksi").count().orderBy("kategori_prediksi").show()
print("\nAkurasi (kategori == kategori_prediksi):")
from pyspark.sql.functions import col
df.select((col("kategori") == col("kategori_prediksi")).alias("benar")).groupBy("benar").count().show()
spark.stop()
