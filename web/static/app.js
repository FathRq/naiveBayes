// Dashboard Klasifikasi Berita Indonesia — mengambil data dari API backend
// (/api/status, /api/ringkasan, /api/agregasi, /api/prediksi).

const WARNA = {
  hiburan: "#ec4899",
  olahraga: "#22c55e",
  teknologi: "#3b82f6",
  politik: "#f59e0b",
  ekonomi: "#8b5cf6",
};

let ringkasan = null;
let skenarioAktif = "dedup";
let chartSkenario = null;
let halaman = 1;
const PER_PAGE = 15;

const fmt = (n) => new Intl.NumberFormat("id-ID").format(n);
const pct = (x) => (x * 100).toFixed(2).replace(".", ",") + "%";

async function ambilJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

// ---------------------------------------------------------------- status
async function muatStatus() {
  try {
    const s = await ambilJSON("/api/status");
    document.getElementById("status-sumber").textContent =
      s.sumber_ringkasan || "tidak tersedia";
    document.getElementById("status-mongo").textContent = s.mongodb_terhubung
      ? "MongoDB terhubung (" + s.jumlah_dokumen_prediksi + " dokumen prediksi)"
      : "MongoDB belum terhubung — mode fallback file JSON";
    document.getElementById("kpi-mongo").textContent = s.mongodb_terhubung
      ? fmt(s.jumlah_dokumen_prediksi)
      : "0";
    if (!s.mongodb_terhubung) {
      const el = document.getElementById("nosql-offline");
      el.classList.remove("hidden");
      el.textContent =
        "MongoDB sedang tidak terhubung, jadi bagian NoSQL dan explorer " +
        "prediksi belum aktif. Jalankan: docker compose up -d --build, " +
        "lalu spark-submit simpan_mongodb.py. Ringkasan di atas dibaca " +
        "dari file JSON lokal (fallback).";
    }
    return s;
  } catch (e) {
    document.getElementById("status-sumber").textContent = "backend tidak terjangkau";
    return null;
  }
}

// ---------------------------------------------------------------- ringkasan
function renderRingkasan() {
  const d = ringkasan;
  const dedup = d.eksperimen.dedup;

  document.getElementById("kpi-akurasi").textContent = pct(dedup.evaluasi.accuracy);
  document.getElementById("kpi-f1").textContent = dedup.evaluasi.weighted_f1.toFixed(4);
  document.getElementById("kpi-total").textContent = fmt(d.dataset.total_baris_dedup);
  document.getElementById("kpi-dedup-note").textContent =
    fmt(d.dataset.near_duplikat_dibuang) + " near-duplikat dibuang dari " +
    fmt(d.dataset.total_baris_raw) + " baris mentah";

  document.getElementById("dataset-detail").innerHTML =
    "<p><span class='text-slate-500'>Sumber:</span> <b>" + d.dataset.sumber +
    "</b> (scraping etis, patuh robots.txt)</p>" +
    "<p><span class='text-slate-500'>File:</span> <code>" + d.dataset.nama_file + "</code></p>" +
    "<p><span class='text-slate-500'>Baris mentah:</span> " + fmt(d.dataset.total_baris_raw) + "</p>" +
    "<p><span class='text-slate-500'>Setelah dedup:</span> " + fmt(d.dataset.total_baris_dedup) + "</p>" +
    "<p><span class='text-slate-500'>Kategori:</span> " + d.dataset.kategori.join(", ") + "</p>" +
    "<p><span class='text-slate-500'>Split DEDUP:</span> " + fmt(dedup.split.train) +
    " latih / " + fmt(dedup.split.test) + " uji (seed " + dedup.split.seed + ")</p>";

  // Chart keseimbangan kelas
  const labels = Object.keys(d.dataset.keseimbangan_kelas);
  new Chart(document.getElementById("chart-kelas"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Jumlah artikel per kategori (setelah dedup)",
        data: labels.map((k) => d.dataset.keseimbangan_kelas[k]),
        backgroundColor: labels.map((k) => WARNA[k] || "#6366f1"),
      }],
    },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
  });

  // Chart perbandingan RAW vs DEDUP
  chartSkenario = new Chart(document.getElementById("chart-skenario"), {
    type: "bar",
    data: {
      labels: ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"],
      datasets: [
        { label: "RAW", backgroundColor: "#94a3b8",
          data: [d.eksperimen.raw.evaluasi.accuracy, d.eksperimen.raw.evaluasi.weighted_precision,
                 d.eksperimen.raw.evaluasi.weighted_recall, d.eksperimen.raw.evaluasi.weighted_f1] },
        { label: "DEDUP (resmi)", backgroundColor: "#4f46e5",
          data: [dedup.evaluasi.accuracy, dedup.evaluasi.weighted_precision,
                 dedup.evaluasi.weighted_recall, dedup.evaluasi.weighted_f1] },
      ],
    },
    options: { scales: { y: { min: 0.85, max: 1 } } },
  });

  document.getElementById("perbandingan-note").innerHTML =
    "<b>Perbandingan RAW vs DEDUP:</b> selisih akurasi " +
    pct(d.eksperimen.perbandingan.selisih_accuracy) + ". " +
    d.eksperimen.perbandingan.kesimpulan;

  document.getElementById("interpretasi").textContent = d.interpretasi.ringkasan;
  document.getElementById("terkuat").textContent = d.interpretasi.kategori_terkuat;
  document.getElementById("terlemah").textContent = d.interpretasi.kategori_terlemah;
  document.getElementById("sering-salah").textContent = d.interpretasi.paling_sering_salah;

  document.getElementById("model-detail").innerHTML =
    "<p>Algoritma: <b>" + d.model.nama + "</b> (" + d.model.tipe + ")</p>" +
    "<p>Fitur: " + d.model.fitur + "</p>" +
    "<p>Fitur dari: " + d.model.fitur_dari + "</p>" +
    "<p>Parameter: smoothing " + d.model.parameter.smoothing + ", modelType " +
    d.model.parameter.modelType + " (100% default, tanpa tuning)</p>" +
    "<p>" + d.model.catatan + "</p>";

  // Isi pilihan filter explorer
  const kat = d.dataset.kategori;
  for (const id of ["f-kategori", "f-prediksi"]) {
    const sel = document.getElementById(id);
    kat.forEach((k) => {
      const o = document.createElement("option");
      o.value = k; o.textContent = k;
      sel.appendChild(o);
    });
  }

  renderSkenario();
}

function renderSkenario() {
  const d = ringkasan.eksperimen[skenarioAktif];
  document.getElementById("label-skenario").textContent = skenarioAktif.toUpperCase();
  document.getElementById("label-cm").textContent = skenarioAktif.toUpperCase();

  document.querySelectorAll(".btn-skenario").forEach((b) => {
    const aktif = b.dataset.skenario === skenarioAktif;
    b.className = "btn-skenario px-4 py-2 " + (aktif
      ? "bg-indigo-600 text-white"
      : "bg-white text-slate-600");
  });

  // Tabel metrik per kategori
  document.getElementById("tabel-metrik").innerHTML = d.metrik_per_kategori.map((m) =>
    "<tr class='border-b'>" +
    "<td class='py-2 font-medium'><span class='inline-block w-2.5 h-2.5 rounded-full mr-2' style='background:" +
    (WARNA[m.kategori] || "#6366f1") + "'></span>" + m.kategori + "</td>" +
    "<td>" + m.precision.toFixed(4) + "</td>" +
    "<td>" + m.recall.toFixed(4) + "</td>" +
    "<td class='font-semibold'>" + m.f1.toFixed(4) + "</td></tr>"
  ).join("");

  // Confusion matrix (heatmap sederhana)
  const cm = d.confusion_matrix;
  const maks = Math.max(...cm.matriks.flat());
  let html = "<thead><tr><th class='p-2 text-slate-500'>asli \\ prediksi</th>" +
    cm.label.map((l) => "<th class='p-2 text-slate-500'>" + l + "</th>").join("") + "</tr></thead><tbody>";
  cm.matriks.forEach((baris, i) => {
    html += "<tr><th class='p-2 text-left text-slate-500'>" + cm.label[i] + "</th>";
    baris.forEach((v, j) => {
      const diag = i === j;
      const intens = Math.round((v / maks) * 70);
      const bg = diag
        ? "background:rgba(79,70,229," + (0.25 + intens / 100) + ");color:white"
        : v > 0 ? "background:rgba(244,63,94," + (0.08 + intens / 160) + ")" : "";
      html += "<td class='p-2 text-center font-medium' style='" + bg + "'>" + v + "</td>";
    });
    html += "</tr>";
  });
  document.getElementById("tabel-cm").innerHTML = html + "</tbody>";
}

// ---------------------------------------------------------------- agregasi MongoDB
async function muatAgregasi() {
  try {
    const a = await ambilJSON("/api/agregasi");
    if (a.distribusi_prediksi && a.distribusi_prediksi.length) {
      new Chart(document.getElementById("chart-distribusi"), {
        type: "doughnut",
        data: {
          labels: a.distribusi_prediksi.map((x) => x.kategori),
          datasets: [{
            data: a.distribusi_prediksi.map((x) => x.jumlah),
            backgroundColor: a.distribusi_prediksi.map((x) => WARNA[x.kategori] || "#6366f1"),
          }],
        },
      });
      document.getElementById("akurasi-mongo").textContent =
        pct(a.akurasi_dari_mongodb) + " (" + fmt(a.benar) + " benar dari " +
        fmt(a.total_dokumen) + " dokumen)";
      document.getElementById("pasangan-salah").innerHTML =
        a.pasangan_salah_terbanyak.map((p) =>
          "<div class='flex justify-between bg-slate-50 border rounded-lg px-3 py-2'>" +
          "<span>" + p.kategori + " &#8594; <b>" + p.kategori_prediksi + "</b></span>" +
          "<span class='font-semibold'>" + fmt(p.jumlah) + " artikel</span></div>"
        ).join("");
    }
  } catch (e) { /* mode fallback: bagian NoSQL dibiarkan kosong */ }
}

// ---------------------------------------------------------------- explorer prediksi
async function muatPrediksi() {
  const params = new URLSearchParams({
    page: halaman, per_page: PER_PAGE,
    kategori: document.getElementById("f-kategori").value,
    prediksi: document.getElementById("f-prediksi").value,
    benar: document.getElementById("f-benar").value,
  });
  try {
    const r = await ambilJSON("/api/prediksi?" + params.toString());
    const tbody = document.getElementById("tabel-prediksi");
    if (r.error) {
      tbody.innerHTML = "<tr><td colspan='6' class='py-4 text-amber-600'>" + r.error + "</td></tr>";
      document.getElementById("prediksi-total").textContent = "";
      document.getElementById("halaman-info").textContent = "";
      return;
    }
    tbody.innerHTML = r.data.map((d) =>
      "<tr class='border-b'>" +
      "<td class='py-2 font-mono'>" + d.id_artikel + "</td>" +
      "<td>" + d.kategori + "</td>" +
      "<td class='font-medium'>" + d.kategori_prediksi + "</td>" +
      "<td class='text-slate-500'>" + (d.model || "-") + "</td>" +
      "<td class='text-slate-500'>" + (d.sumber || "-") + "</td>" +
      "<td>" + (d.benar
        ? "<span class='bg-emerald-100 text-emerald-700 font-semibold rounded px-2 py-0.5'>Benar</span>"
        : "<span class='bg-rose-100 text-rose-700 font-semibold rounded px-2 py-0.5'>Salah</span>") +
      "</td></tr>"
    ).join("");
    document.getElementById("prediksi-total").textContent = fmt(r.total) + " dokumen";
    document.getElementById("halaman-info").textContent =
      "Halaman " + r.page + " dari " + r.total_halaman;
    document.getElementById("btn-prev").disabled = r.page <= 1;
    document.getElementById("btn-next").disabled = r.page >= r.total_halaman;
  } catch (e) { /* abaikan */ }
}

// ---------------------------------------------------------------- init
document.querySelectorAll(".btn-skenario").forEach((b) =>
  b.addEventListener("click", () => { skenarioAktif = b.dataset.skenario; renderSkenario(); }));
for (const id of ["f-kategori", "f-prediksi", "f-benar"]) {
  document.getElementById(id).addEventListener("change", () => { halaman = 1; muatPrediksi(); });
}
document.getElementById("btn-prev").addEventListener("click", () => { if (halaman > 1) { halaman--; muatPrediksi(); } });
document.getElementById("btn-next").addEventListener("click", () => { halaman++; muatPrediksi(); });

(async () => {
  await muatStatus();
  try {
    const r = await ambilJSON("/api/ringkasan");
    ringkasan = r.data;
    renderRingkasan();
  } catch (e) {
    document.getElementById("interpretasi").textContent =
      "Ringkasan analisis belum tersedia. Jalankan pipeline Spark/MLlib terlebih dahulu.";
  }
  await muatAgregasi();
  await muatPrediksi();
})();
