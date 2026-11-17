# ==============================================================================
# BUKTI: TIDAK ADA DATA LEAKAGE (kolom kategori tidak bocor ke model)
# ------------------------------------------------------------------------------
# Menjawab kecurigaan: "apakah kolom kategori bocor ke dalam model?"
#
# Cara kerja klasifikasi SUPERVISED:
#   - FITUR  (input model) = vektor TF-IDF, dibuat dari TEKS saja (judul+isi)
#   - LABEL  (target)      = kategori  (ini "kunci jawaban")
#   - Model belajar pemetaan:  FITUR  ->  LABEL
#   - Kolom 'kategori' TIDAK pernah masuk ke dalam vektor fitur.
#
# Skrip ini MENAMPILKAN bukti visual:
#   1. Kolom apa saja yang benar-benar masuk ke model
#   2. 1 baris contoh dengan alur lengkap:
#        teks -> kata -> vektor TF-IDF -> label -> prediksi -> benar?
#
# CARA JALANKAN:
#   docker exec spark-master /opt/spark/bin/spark-submit /scripts/bukti_tidak_leakage.py
# ==============================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes

PATH_PARQUET = "hdfs://namenode:9000/proyek_bigdata/clean/berita_clean.parquet"

spark = (
    SparkSession.builder.appName("Bukti_Tidak_Leakage")
    .master("spark://spark-master:7077")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------
# 1. BACA + BANGUN FITUR (TF-IDF dari TEKS SAJA, bukan dari kategori)
# ------------------------------------------------------------------
print("\n=== 1. MEMBACA DATA + BANGUN FITUR ===")
df = spark.read.parquet(PATH_PARQUET)
print(f"Jumlah baris : {df.count()}")

# FITUR hanya dari teks_bersih (judul + isi). TIDAK dari kolom kategori.
tokenizer  = Tokenizer(inputCol="teks_bersih", outputCol="kata")
hashing_tf = HashingTF(inputCol="kata", outputCol="tf", numFeatures=100000)
idf        = IDF(inputCol="tf", outputCol="fitur")

df_kata   = tokenizer.transform(df)
df_tf     = hashing_tf.transform(df_kata)
idf_model = idf.fit(df_tf)
df_fitur  = idf_model.transform(df_tf)

# LABEL = kategori (kunci jawaban), diubah ke indeks numerik
indexer       = StringIndexer(inputCol="kategori", outputCol="label")
indexer_model = indexer.fit(df_fitur)
df_final      = indexer_model.transform(df_fitur)
labels = list(indexer_model.labels)
print("Peta indeks -> kategori:")
for i, nama in enumerate(labels):
    print(f"  {i} -> {nama}")

# ------------------------------------------------------------------
# 2. TRAIN + PREDIKSI (model demonstrasi, dilatih di semua data)
# ------------------------------------------------------------------
print("\n=== 2. TRAIN + PREDIKSI (model demonstrasi) ===")
nb = NaiveBayes(featuresCol="fitur", labelCol="label", modelType="multinomial")
model = nb.fit(df_final)
prediksi = model.transform(df_final)

# ------------------------------------------------------------------
# 3. BUKTI VISUAL
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("BUKTI: APA SAJA YANG DILIHAT MODEL?")
print("=" * 70)

print("\n[1] Kolom yang MASUK ke model NaiveBayes:")
print("      - fitur : vektor TF-IDF (hanya ANGKA, dibuat dari teks)")
print("      - label : indeks kategori (kunci jawaban)")
print("      -> Kolom 'kategori' (string) TIDAK ada di vektor fitur.")

print("\n[2] 1 BARIS CONTOH - alur lengkap dari teks sampai prediksi:")
row = prediksi.select(
    "id_artikel", "kategori", "teks_bersih", "kata", "fitur", "label", "prediction"
).first()

print(f"\n  id_artikel : {row['id_artikel']}")

print(f"\n  (A) TEKS ASLI (teks_bersih, dipotong 120 karakter):")
print(f"        \"{row['teks_bersih'][:120]} ...\"")

print(f"\n  (B) HASIL TOKENIZATION (20 kata pertama):")
print(f"        {row['kata'][:20]}")

# fitur = SparseVector (hanya menyimpan nilai non-nol)
vec = row["fitur"]
print(f"\n  (C) VEKTOR FITUR yang benar-benar dilihat model:")
if hasattr(vec, "indices"):
    idxs  = list(vec.indices)
    vals  = list(vec.values)
    pairs = list(zip(idxs, vals))
    print(f"        vektor SPARSE, {len(pairs)} nilai non-nol (10 pertama):")
    for idx, val in pairs[:10]:
        print(f"          indeks {idx:6d}  ->  bobot TF-IDF {val:.4f}")
    print("        ... semuanya ANGKA. Tidak ada string kategori di sini.")
else:
    print("        vektor DENSE (10 nilai pertama):")
    for i in range(min(10, len(vec))):
        print(f"          indeks {i:6d}  ->  bobot TF-IDF {vec[i]:.4f}")

print(f"\n  (D) LABEL / KUNCI JAWABAN:")
print(f"        indeks label = {int(row['label'])}   ->   kategori = '{row['kategori']}'")

pred_idx  = int(row["prediction"])
pred_nama = labels[pred_idx]
benar     = (pred_idx == int(row["label"]))
print(f"\n  (E) PREDIKSI MODEL (ditebak HANYA dari vektor fitur):")
print(f"        indeks prediksi = {pred_idx}   ->   kategori = '{pred_nama}'")
print(f"        BENAR? = {benar}")

# ------------------------------------------------------------------
# 4. KESIMPULAN
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("KESIMPULAN")
print("=" * 70)
print("Model Naive Bayes HANYA melihat vektor ANGKA (TF-IDF) yang")
print("dihasilkan dari TEKS (judul + isi). Kolom 'kategori' hanya")
print("berfungsi sebagai LABEL (kunci jawaban) saat training & evaluasi.")
print("Kategori TIDAK pernah menjadi fitur  ->  TIDAK ADA data leakage.")
print("")
print("ANALOGI: belajar dari buku (soal + kunci jawaban), lalu ujian")
print("tanpa kunci. Skor ujian = akurasi. Kunci di buku tidak bocor")
print("ke jawabanmu - ia hanya dipakai mengoreksi setelah kamu menjawab.")

spark.stop()
