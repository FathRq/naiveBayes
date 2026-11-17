from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    count,
    length,
    lower,
    regexp_replace,
    split,
    size,
    trim,
    when,
    to_timestamp,
    year,
    month,
)

# 1. INISIALISASI SPARK SESSION
spark = (
    SparkSession.builder.appName("BagianB_PengolahanAnalisisBeritaAntara")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ==============================================================================
# POIN 1: MEMBACA DATASET DARI HDFS
# ==============================================================================
print("\n=== 1. MEMBACA DATASET MENTAH DARI HDFS ===")
df_raw = (
    spark.read.option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .option("quote", '"')
    .option("mode", "PERMISSIVE")
    .csv("hdfs://namenode:9000/proyek_bigdata/raw/dataset.csv")
)

# ==============================================================================
# POIN 2: MEMERIKSA STRUKTUR, TIPE DATA, DAN JUMLAH BARIS AWAL
# ==============================================================================
print("\n=== 2. STRUKTUR & JUMLAH BARIS AWAL ===")
df_raw.printSchema()
total_awal = df_raw.count()
print(f"Total baris mentah: {total_awal}")

print("\nJumlah nilai NULL per kolom (Sebelum Cleaning):")
df_raw.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df_raw.columns]
).show()

# ==============================================================================
# POIN 3: MENANGANI DATA KOSONG, DUPLIKASI, DAN KOLOM RUSAK / BERGESER
# ==============================================================================
print("\n=== 3. PEMBERSIHAN DATA (DATA CLEANING) ===")

# a. Hapus baris yang nilai pada kolom utamanya NULL
df_clean = df_raw.dropna(subset=["id_artikel", "judul", "isi_artikel", "kategori"])

# b. Filter ketat untuk membuang baris/kolom yang rusak atau bergeser:
#    - id_artikel harus murni angka (regex ^[0-9]+$)
#    - nama_media harus 'ANTARA News' (memastikan kolom tidak bergeser)
#    - url harus diawali 'http'
#    - kategori tidak kosong dan panjangnya < 30 karakter
#    - isi_artikel minimal 100 karakter
df_clean = df_clean.filter(
    (col("id_artikel").rlike("^[0-9]+$"))
    & (trim(col("nama_media")) == "ANTARA News")
    & (col("url").startswith("http"))
    & (trim(col("kategori")) != "")
    & (length(trim(col("kategori"))) < 30)
    & (length(trim(col("isi_artikel"))) >= 100)
)

# c. Hapus data duplikat berdasarkan id_artikel dan judul
df_clean = df_clean.dropDuplicates(["id_artikel"]).dropDuplicates(["judul"])

total_bersih = df_clean.count()
print(f"Total baris setelah dibersihkan : {total_bersih}")
print(f"Total baris rusak/duplikat dibuang: {total_awal - total_bersih}")

# ==============================================================================
# POIN 4: TRANSFORMASI DATA
# ==============================================================================
print("\n=== 4. TRANSFORMASI DATA ===")

# Hapus teks boilerplate hak cipta di akhir artikel ANTARA
boilerplate_regex = r"Dilarang keras mengambil konten, melakukan crawling atau pengindeksan otomatis untuk AI di situs web ini tanpa izin tertulis dari Kantor Berita ANTARA\.?"

df_transformed = (
    df_clean
    # Casting tipe data id_artikel ke Long dan tanggal_publikasi ke Timestamp
    .withColumn("id_artikel", col("id_artikel").cast("long"))
    .withColumn("waktu_publikasi_ts", to_timestamp(col("tanggal_publikasi")))
    .withColumn("tahun_publikasi", year(col("waktu_publikasi_ts")))
    .withColumn("bulan_publikasi", month(col("waktu_publikasi_ts")))
    # Normalisasi nama kategori
    .withColumn("kategori", trim(lower(col("kategori"))))
    # Bersihkan isi_artikel dari boilerplate hak cipta
    .withColumn("isi_tanpa_copyright", regexp_replace(col("isi_artikel"), boilerplate_regex, ""))
    # Gabungkan judul + isi berita, ubah ke huruf kecil, hapus tanda baca & angka untuk fitur MLlib
    .withColumn(
        "teks_bersih",
        trim(
            regexp_replace(
                lower(concat_ws(" ", col("judul"), col("isi_tanpa_copyright"))),
                r"[^a-z\s]",
                " ",
            )
        ),
    )

    .withColumn("teks_bersih", regexp_replace(col("teks_bersih"), r"\s+", " "))
    # Ekstraksi fitur panjang teks & jumlah kata
    .withColumn("panjang_karakter", length(col("teks_bersih")))
    .withColumn("jumlah_kata", size(split(col("teks_bersih"), " ")))
)

# Filter kategori yang jumlah datanya terlalu sedikit (< 50 berita) agar tidak mengganggu klasifikasi
kategori_valid = (
    df_transformed.groupBy("kategori")
    .count()
    .filter(col("count") >= 50)
    .select("kategori")
)
df_transformed = df_transformed.join(kategori_valid, on="kategori", how="inner")

print(f"Total baris final siap analisis & MLlib: {df_transformed.count()}")
df_transformed.select(
    "id_artikel", "kategori", "tahun_publikasi", "jumlah_kata", "teks_bersih"
).show(5, truncate=40)

# ==============================================================================
# DIAGNOSTIC: verifikasi kolom fitur sebelum disimpan
# (mendeteksi di tahap mana NULL muncul & memastikan teks_bersih bermakna)
# ==============================================================================
print("\n--- DIAGNOSTIC: jumlah NULL per kolom fitur ---")
df_transformed.select(
    [count(when(col(c).isNull(), c)).alias(c)
     for c in ["judul", "isi_artikel", "isi_tanpa_copyright", "teks_bersih"]]
).show()

print("\n--- DIAGNOSTIC: sampel teks_bersih non-NULL (verifikasi isi bermakna) ---")
df_transformed.filter(col("teks_bersih").isNotNull()) \
    .select("id_artikel", "kategori", "jumlah_kata", "teks_bersih") \
    .show(5, truncate=80)

# Simpan hasil bersih ke HDFS dalam format Parquet untuk tahap Spark MLlib
df_transformed.write.mode("overwrite").parquet(
    "hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet"
)
print("Data bersih berhasil disimpan ke HDFS: /proyek_bigdata/clean/berita_clean.parquet")

# ==============================================================================
# POIN 5: MINIMAL TIGA ANALISIS (SPARK SQL & DATAFRAME API)
# ==============================================================================
print("\n=== 5. HASIL ANALISIS DATA (MINIMAL 3 ANALISIS) ===")

df_transformed.createOrReplaceTempView("tabel_berita")

# --- ANALISIS 1: Agregasi & Pengelompokan (Distribusi Berita per Kategori) ---
print("\n[Analisis 1] Distribusi Jumlah Berita per Kategori (DataFrame API):")
analisis_1 = (
    df_transformed.groupBy("kategori")
    .agg(count("*").alias("total_berita"))
    .orderBy(col("total_berita").desc())
)
analisis_1.show(20, truncate=False)

# --- ANALISIS 2: Perhitungan Statistik (Rata-rata, Min, Max Jumlah Kata per Kategori) ---
print("\n[Analisis 2] Statistik Panjang Kata per Kategori (Spark SQL):")
analisis_2 = spark.sql("""
    SELECT 
        kategori,
        COUNT(*) AS jumlah_artikel,
        ROUND(AVG(jumlah_kata), 2) AS rata_rata_kata,
        MIN(jumlah_kata) AS min_kata,
        MAX(jumlah_kata) AS max_kata
    FROM tabel_berita
    GROUP BY kategori
    ORDER BY rata_rata_kata DESC
""")
analisis_2.show(20, truncate=False)

# --- ANALISIS 3: Tren Temporal & Filtering (Jumlah Berita per Tahun & Kategori Terbanyak) ---
print("\n[Analisis 3] Distribusi Publikasi Berita per Tahun (Spark SQL):")
analisis_3 = spark.sql("""
    SELECT 
        tahun_publikasi,
        COUNT(*) AS total_artikel,
        ROUND(AVG(jumlah_kata), 2) AS rata_rata_kata
    FROM tabel_berita
    WHERE tahun_publikasi IS NOT NULL
    GROUP BY tahun_publikasi
    ORDER BY tahun_publikasi DESC
""")
analisis_3.show(20, truncate=False)

# --- ANALISIS 4 (Tambahan): Proporsi Berita Mendalam (> 300 Kata) per Kategori ---
print("\n[Analisis 4] Jumlah Artikel Mendalam (> 300 Kata) per Kategori (Spark SQL):")
analisis_4 = spark.sql("""
    SELECT 
        kategori,
        COUNT(*) AS artikel_mendalam
    FROM tabel_berita
    WHERE jumlah_kata > 300
    GROUP BY kategori
    ORDER BY artikel_mendalam DESC
""")
analisis_4.show(20, truncate=False)

spark.stop()