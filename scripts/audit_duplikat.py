# ==============================================================================
# AUDIT INTEGRITAS DATA: cek duplikat / near-duplikat yang bisa "membocorkan"
# train ke test (penyebab akurasi tergelembung)
# ==============================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, substring, length

spark = (
    SparkSession.builder.appName("AuditIntegritasData")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet("hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet")
total = df.count()
print(f"Total baris: {total}")

# 0. Sumber media (1 sumber atau banyak?)
print("\n=== [0] Distribusi nama_media ===")
df.groupBy("nama_media").count().orderBy(col("count").desc()).show(20, truncate=40)

# 1. Duplikat PERSIS judul
print("=== [1] Duplikat PERSIS judul ===")
dup_judul = df.groupBy("judul").count().filter(col("count") > 1)
n_dup_judul = dup_judul.count()
baris_terpengaruh_judul = dup_judul.withColumnRenamed("count", "c").select("judul", "c").join(
    df.select("judul"), "judul").count() if n_dup_judul > 0 else 0
print(f"Jumlah judul unik yang kembar : {n_dup_judul}")
if n_dup_judul > 0:
    dup_judul.orderBy(col("count").desc()).show(10, truncate=60)

# 2. Duplikat PERSIS isi_artikel
print("=== [2] Duplikat PERSIS isi_artikel ===")
dup_isi = df.groupBy("isi_artikel").count().filter(col("count") > 1)
n_dup_isi = dup_isi.count()
print(f"Jumlah isi yang kembar persis : {n_dup_isi}")
if n_dup_isi > 0:
    dup_isi.orderBy(col("count").desc()).show(5, truncate=60)

# 3. NEAR-DUPLIKAT: 30 kata pertama teks_bersih identik
#    (teks di-truncate ~400 kata, jadi "kepala" artikel yang sama = artikel yang sama)
print("=== [3] NEAR-DUPLIKAT: 30 kata pertama identik ===")
df2 = df.withColumn("head30", concat_ws(" ",
        substring(col("teks_bersih"), 1, 200)))  # aproksimasi ~30 kata
near = df2.groupBy("head30").count().filter(col("count") > 1)
n_near = near.count()
print(f"Jumlah 'kepala' artikel yang kembar: {n_near}")
if n_near > 0:
    # berapa baris total yang terlibat near-duplikat?
    involved = near.select("head30").join(df2.select("head30"), "head30")
    n_involved = involved.count()
    print(f"Baris yang terlibat near-duplikat : {n_involved}  ({n_involved/total*100:.1f}% dari data)")
    near.orderBy(col("count").desc()).show(8, truncate=70)

# 4. Estimasi dampak: berapa % data yang "bocor" ke kedua sisi split
#    (aproksimasi: jika X% baris adalah near-duplikat, peluang bocor ~ tinggi)
print("\n=== [4] Ringkasan risiko ===")
print(f"Jika ada near-duplikat, randomSplit 80/20 akan memisahkan salinannya")
print(f"ke train & test -> model 'menghafal' -> akurasi test tergelembung.")

spark.stop()
