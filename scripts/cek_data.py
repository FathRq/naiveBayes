from pyspark.sql import SparkSession

# 1. Inisialisasi SparkSession yang terhubung ke Spark Master dan HDFS
spark = (
    SparkSession.builder.appName("BacaDataBeritaHDFS")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)

# Mengurangi log terminal yang terlalu ramai
spark.sparkContext.setLogLevel("WARN")

print("=== MEMBACA DATASET DARI HDFS ===")
# Gunakan multiLine=True dan escape='"' karena teks berita sering mengandung enter (\n) dan tanda kutip
df = (
    spark.read.option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .option("quote", '"')
    .option("mode", "PERMISSIVE")
    .csv("hdfs://namenode:9000/proyek_bigdata/raw/dataset.csv")
)

# 2. Cek Struktur & Tipe Data Kolom
print("\n--- 1. SKEMA & NAMA KOLOM ---")
df.printSchema()

# 3. Cek Total Baris Mentah
print(f"\n--- 2. TOTAL BARIS MENTAH: {df.count()} ---")

# 4. Tampilkan 5 Sampel Data Pertama
print("\n--- 3. SAMPEL 5 DATA TERATAS ---")
df.show(5, truncate=50)

spark.stop()