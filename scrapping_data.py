#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kumpulkan_berita_antara.py (Versi Teroptimasi: Multithreading + lxml + Connection Pooling)
========================================================================================
Pengumpul dataset berita Indonesia (ANTARA News) untuk Proyek Akhir Big Data
(Topik 4: Klasifikasi Berita Indonesia, Naive Bayes, HDFS + Spark MLlib).

Fitur utama
-----------
* Cepat    : Multithreading terkontrol (ThreadPoolExecutor), HTTP connection pooling,
             dan parser lxml (dengan fallback otomatis ke html.parser).
* Etika    : patuh robots.txt (termasuk Crawl-delay), User-Agent jelas + kontak,
             global rate-limiter thread-safe, backoff otomatis saat HTTP 429/503,
             tidak login / tidak melewati paywall, tidak mengambil data pribadi.
* Skala    : target default 50.000 artikel (10.000 per kelas x 5 kelas)
* Aman     : checkpoint CSV bertahap + state.json -> bisa di-RESUME kapan saja
* Keluaran : CSV UTF-8 (tanpa baris baru di dalam sel) siap dibaca Spark

Cara pakai (PowerShell / CMD):
    pip install requests beautifulsoup4 lxml
    python kumpulkan_berita_antara.py --contact emailkamu@mhs.unesa.ac.id --delay 0.5 --workers 4

Hentikan kapan saja dengan Ctrl+C; jalankan perintah yang sama untuk melanjutkan.
Ekspor ulang tanpa mengambil data baru:
    python kumpulkan_berita_antara.py --only-finalize
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

# Deteksi parser tercepat (lxml jika terinstal, fallback ke html.parser)
try:
    import lxml  # noqa: F401
    BS_PARSER = "lxml"
except ImportError:
    BS_PARSER = "html.parser"

# --------------------------------------------------------------------------
# KONFIGURASI SUMBER (label kelas -> kanal resmi ANTARA)
# --------------------------------------------------------------------------
BASE_URL_DEFAULT = "https://www.antaranews.com"

KATEGORI_SUMBER = {
    "Politik": {
        "index": [
            "/politik", "/hukum", "/nasional", "/pemilu",
            "/tag/dpr-ri", "/tag/kpk", "/tag/pilkada", "/tag/pemerintah",
            "/tag/presiden-prabowo", "/tag/kementerian",
        ],
        "rss": ["politik", "hukum", "nasional"],
    },
    "Ekonomi": {
        "index": [
            "/ekonomi", "/ekonomi/finansial", "/ekonomi/bisnis", "/ekonomi/bursa",
            "/ekonomi/global", "/ekonomi/makro", "/ekonomi/properti",
        ],
        "rss": ["ekonomi"],
    },
    "Olahraga": {
        "index": [
            "/olahraga", "/sepakbola", "/olahraga/bulutangkis",
            "/olahraga/bola-basket", "/olahraga/tenis", "/olahraga/balap",
            "/sepakbola/liga-champions", "/sepakbola/liga-liga-dunia",
        ],
        "rss": ["olahraga", "sepakbola"],
    },
    "Teknologi": {
        "index": [
            "/tekno", "/otomotif", "/tekno/gadget", "/tekno/internet", "/tekno/sains",
            "/tag/kecerdasan-buatan", "/tag/teknologi", "/tag/aplikasi",
            "/tag/smartphone", "/tag/kendaraan-listrik",
        ],
        "rss": ["tekno", "otomotif"],
    },
    "Hiburan": {
        "index": [
            "/hiburan", "/lifestyle", "/hiburan/film", "/hiburan/musik",
            "/hiburan/selebriti", "/hiburan/budaya", "/lifestyle/kuliner",
            "/lifestyle/wisata", "/tag/film-indonesia", "/tag/konser-musik",
        ],
        "rss": ["hiburan", "lifestyle"],
    },
}

UA_NAME = "BigDataUNESA-AcademicCollector"
NAMA_MEDIA = "ANTARA News"
FIELDS = [
    "id_artikel", "judul", "ringkasan", "isi_artikel", "tanggal_publikasi",
    "nama_media", "kategori", "url", "waktu_diambil",
]

csv.field_size_limit(10_000_000)
log = logging.getLogger("antara")


# --------------------------------------------------------------------------
# EXCEPTION
# --------------------------------------------------------------------------
class TemporaryFetchError(Exception):
    """Gagal sementara (jaringan/5xx) setelah semua retry habis."""


class AbortCrawl(Exception):
    """Hentikan crawling dengan aman (diduga diblokir / gagal beruntun)."""


# --------------------------------------------------------------------------
# ROBOTS.TXT
# --------------------------------------------------------------------------
class Robots:
    def __init__(self, session, base_url, user_agent):
        self.ua = user_agent
        self.rp = RobotFileParser()
        self._lock = threading.Lock()
        url = base_url + "/robots.txt"
        last_err = None
        for i in range(3):
            try:
                r = session.get(url, timeout=20)
                if r.status_code == 200:
                    self.rp.parse(r.text.splitlines())
                elif 400 <= r.status_code < 500:
                    self.rp.parse([])
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")
                log.info("robots.txt dibaca dari %s (HTTP %s)", url, r.status_code)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(3 * (i + 1))
        raise SystemExit(
            f"Tidak bisa membaca robots.txt ({last_err}). "
            "Berhenti demi keamanan; coba lagi nanti."
        )

    def allowed(self, url):
        with self._lock:
            return self.rp.can_fetch(self.ua, url)

    def crawl_delay(self):
        try:
            return self.rp.crawl_delay(self.ua)
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------
# FETCHER (thread-safe rate limit + retry + backoff adaptif)
# --------------------------------------------------------------------------
class Fetcher:
    def __init__(self, session, base_delay, max_retries=5, max_fail_streak=15):
        self.s = session
        self.base_delay = base_delay
        self.delay = base_delay
        self.max_retries = max_retries
        self.max_fail_streak = max_fail_streak
        self.fail_streak = 0
        self.block_streak = 0
        self._last = 0.0
        self._lock = threading.Lock()

    def _wait(self):
        with self._lock:
            target = self._last + self.delay + random.uniform(0, 0.3 * self.delay)
            now = time.monotonic()
            sleep_for = max(0.0, target - now)
            self._last = now + sleep_for
        if sleep_for > 0:
            time.sleep(sleep_for)

    @staticmethod
    def _retry_after(resp):
        v = resp.headers.get("Retry-After")
        if v and v.strip().isdigit():
            return min(int(v.strip()), 300)
        return None

    def get(self, url):
        """Return bytes (sukses) | None (permanen: 404/410/dll, aman di-skip).
        Raise TemporaryFetchError / AbortCrawl bila perlu."""
        for attempt in range(1, self.max_retries + 1):
            self._wait()
            try:
                r = self.s.get(url, timeout=(10, 25))
            except requests.RequestException as e:
                wait = min(2 ** attempt * 2, 120)
                log.warning(
                    "Kesalahan jaringan (%s) pada %s; coba lagi %ss [%d/%d]",
                    type(e).__name__, url, wait, attempt, self.max_retries,
                )
                time.sleep(wait)
                continue

            code = r.status_code
            if code == 200:
                with self._lock:
                    self.fail_streak = 0
                    self.block_streak = 0
                    self.delay = max(self.base_delay, self.delay * 0.95)
                return r.content
            if code in (429, 503):
                wait = self._retry_after(r) or min(2 ** attempt * 5, 300)
                with self._lock:
                    self.delay = min(self.delay * 2, 60)
                    cur_delay = self.delay
                log.warning(
                    "HTTP %s (dibatasi). Tidur %ss, jeda dasar naik ke %.1fs [%d/%d]",
                    code, wait, cur_delay, attempt, self.max_retries,
                )
                time.sleep(wait)
                continue
            if code in (500, 502, 504, 408):
                wait = min(2 ** attempt * 3, 120)
                log.warning("HTTP %s pada %s; coba lagi %ss", code, url, wait)
                time.sleep(wait)
                continue
            if code in (401, 403):
                with self._lock:
                    self.block_streak += 1
                    streak = self.block_streak
                log.warning("HTTP %s pada %s (skip). Beruntun: %d", code, url, streak)
                if streak >= 8:
                    raise AbortCrawl(
                        "Ditolak (401/403) 8x beruntun: kemungkinan diblokir. "
                        "Berhenti; lanjutkan nanti dengan jeda lebih besar (--delay)."
                    )
                return None
            log.info("HTTP %s pada %s (skip)", code, url)
            return None

        with self._lock:
            self.fail_streak += 1
            f_streak = self.fail_streak
        if f_streak >= self.max_fail_streak:
            raise AbortCrawl(
                f"Gagal beruntun {f_streak}x. Periksa koneksi, "
                "lalu jalankan ulang untuk melanjutkan (resume)."
            )
        raise TemporaryFetchError(url)


# --------------------------------------------------------------------------
# PENYIMPANAN (thread-safe CSV append + checkpoint) & STATE
# --------------------------------------------------------------------------
class Store:
    def __init__(self, raw_csv: Path, skip_file: Path):
        self.raw_csv = raw_csv
        self.skip_file = skip_file
        self.seen = set()
        self.skipped = set()
        self.counts = Counter()
        self.buf = []
        self._lock = threading.Lock()
        self._load()
        new_file = (not raw_csv.exists()) or raw_csv.stat().st_size == 0
        self.f = open(raw_csv, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        if new_file:
            self.w.writeheader()
            self.f.flush()
        self.sf = open(skip_file, "a", encoding="utf-8")

    def _load(self):
        if self.raw_csv.exists():
            with open(self.raw_csv, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    u = row.get("url")
                    k = row.get("kategori")
                    if u and k and u not in self.seen:
                        self.seen.add(u)
                        self.counts[k] += 1
        if self.skip_file.exists():
            with open(self.skip_file, encoding="utf-8") as f:
                self.skipped = {l.strip() for l in f if l.strip()}
        log.info("Resume: %d artikel tersimpan %s", len(self.seen), dict(self.counts))

    def known(self, url):
        with self._lock:
            return url in self.seen or url in self.skipped

    def add(self, row):
        with self._lock:
            if row["url"] in self.seen:
                return False
            self.buf.append(row)
            self.seen.add(row["url"])
            self.counts[row["kategori"]] += 1
            return True

    def skip(self, url):
        with self._lock:
            if url not in self.skipped:
                self.skipped.add(url)
                self.sf.write(url + "\n")

    def flush(self, force_sync=False):
        with self._lock:
            if self.buf:
                self.w.writerows(self.buf)
                self.buf.clear()
            self.f.flush()
            if force_sync:
                os.fsync(self.f.fileno())
            self.sf.flush()

    def close(self):
        self.flush(force_sync=True)
        self.f.close()
        self.sf.close()


def load_state(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("state.json rusak, dibuat ulang (data CSV tetap aman).")
    return {"sources": {}}


def save_state(path: Path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# PARSING
# --------------------------------------------------------------------------
DATELINE_RE = re.compile(r"^[A-Za-z][\w\s,./\-]{1,45}\(ANTARA\)\s*[-–]\s*")
SKIP_PARA_PREFIX = (
    "baca juga", "baca:", "simak", "pewarta", "editor", "copyright",
    "sumber:", "video:", "foto:",
)


def clean(text):
    return " ".join((text or "").split())


def truncate(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + " ..."


def extract_links(content, article_re, base):
    soup = BeautifulSoup(content, BS_PARSER)
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base + "/", a["href"].strip())
        href = href.split("#")[0].split("?")[0]
        if article_re.match(href) and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def parse_article(content, url, kategori, max_body_chars, min_chars=60):
    soup = BeautifulSoup(content, BS_PARSER)

    def meta(key):
        tag = (
            soup.find("meta", attrs={"property": key})
            or soup.find("meta", attrs={"name": key})
        )
        return clean(tag["content"]) if tag and tag.get("content") else ""

    judul = meta("og:title")
    if not judul:
        h1 = soup.find("h1")
        judul = clean(h1.get_text()) if h1 else ""
    judul = re.sub(r"\s*[-|]\s*ANTARA( News)?\s*$", "", judul, flags=re.I)

    ringkasan = meta("og:description") or meta("description")
    tanggal = meta("article:published_time")
    if not tanggal:
        t = soup.find("time")
        tanggal = clean(t.get("datetime") or t.get_text()) if t else ""

    container = (
        soup.select_one("div.post-content")
        or soup.select_one("[itemprop='articleBody']")
        or soup.find("article")
    )
    paras = []
    if container:
        for junk in container.find_all(
            ["script", "style", "aside", "figure", "iframe", "noscript"]
        ):
            junk.decompose()
        for p in container.find_all("p"):
            txt = clean(p.get_text(" "))
            if len(txt) < 25 or txt.lower().startswith(SKIP_PARA_PREFIX):
                continue
            paras.append(txt)
    isi = " ".join(paras)
    isi = DATELINE_RE.sub("", isi)
    isi = truncate(isi, max_body_chars)

    if len(judul) < 10 or len(isi or ringkasan) < min_chars:
        return None

    m = re.search(r"/berita/(\d+)/", url)
    return {
        "id_artikel": m.group(1) if m else "",
        "judul": judul,
        "ringkasan": ringkasan,
        "isi_artikel": isi,
        "tanggal_publikasi": tanggal,
        "nama_media": NAMA_MEDIA,
        "kategori": kategori,
        "url": url,
        "waktu_diambil": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def rss_links(content, article_re):
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    out = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip().split("?")[0]
        if article_re.match(link):
            out.append(link)
    return out


# --------------------------------------------------------------------------
# PROSES CRAWLING (PARALEL PER BATCH URL)
# --------------------------------------------------------------------------
class Ctx:
    pass


def log_progress(ctx):
    c = ctx.store.counts
    log.info(
        "PROGRES | "
        + " | ".join(f"{k}: {c[k]}/{ctx.per_class}" for k in KATEGORI_SUMBER)
        + f" | total {sum(c.values())}"
    )


def checkpoint(ctx, force_sync=False):
    ctx.store.flush(force_sync=force_sync)
    save_state(ctx.state_path, ctx.state)
    log_progress(ctx)


def _fetch_and_parse_worker(kat, u, ctx):
    if ctx.store.known(u):
        return ("known", u, None)
    if not ctx.robots.allowed(u):
        return ("skip", u, None)
    try:
        content = ctx.fetcher.get(u)
    except TemporaryFetchError:
        return ("retry", u, None)
    if content is None:
        return ("skip", u, None)
    row = parse_article(content, u, kat, ctx.max_body_chars)
    if row is None:
        return ("skip", u, None)
    return ("ok", u, row)


def process_urls(kat, urls, ctx):
    # Saring URL baru dan potong sesuai sisa kuota kategori agar tidak berlebih
    remaining = ctx.per_class - ctx.store.counts[kat]
    if remaining <= 0:
        return

    candidates = [u for u in urls if not ctx.store.known(u)]
    if not candidates:
        return

    # Ambil sedikit lebih banyak dari sisa kuota (buffer 20%) untuk antisipasi artikel skip
    batch = candidates[: remaining + max(5, int(remaining * 0.2))]

    with ThreadPoolExecutor(max_workers=ctx.workers) as executor:
        futures = [
            executor.submit(_fetch_and_parse_worker, kat, u, ctx) for u in batch
        ]
        for fut in as_completed(futures):
            # Jika ada worker melempar AbortCrawl, hentikan semua dan teruskan exception
            status, u, row = fut.result()

            if status == "skip":
                ctx.store.skip(u)
            elif status == "ok":
                if ctx.store.counts[kat] < ctx.per_class:
                    ctx.store.add(row)
                    if len(ctx.store.buf) >= ctx.checkpoint_every:
                        checkpoint(ctx)


def crawl_rss(ctx):
    log.info("Fase 1: RSS resmi (berita terbaru)")
    for kat, cfg in KATEGORI_SUMBER.items():
        if ctx.store.counts[kat] >= ctx.per_class:
            continue
        for slug in cfg["rss"]:
            url = f"{ctx.base}/rss/{slug}.xml"
            if not ctx.robots.allowed(url):
                log.info("RSS %s tidak diizinkan robots.txt, dilewati", url)
                continue
            try:
                content = ctx.fetcher.get(url)
            except TemporaryFetchError:
                continue
            if not content:
                log.info("RSS %s tidak tersedia, dilewati", url)
                continue
            links = rss_links(content, ctx.article_re)
            log.info("RSS %s -> %d tautan", slug, len(links))
            process_urls(kat, links, ctx)
    checkpoint(ctx)


def next_source(kat, ctx):
    for path in KATEGORI_SUMBER[kat]["index"]:
        st = ctx.state["sources"].setdefault(
            f"{kat}|{path}", {"next_page": 1, "done": False, "stale": 0}
        )
        if not st["done"]:
            return path, st
    return None, None


def crawl_index_page(kat, path, st, ctx):
    page = st["next_page"]
    url = ctx.base + path + ("" if page == 1 else f"/{page}")
    if not ctx.robots.allowed(url):
        log.info("%s tidak diizinkan robots.txt -> sumber ditutup", url)
        st["done"] = True
        return
    try:
        content = ctx.fetcher.get(url)
    except TemporaryFetchError:
        return
    if content is None:
        log.info("Indeks %s habis (halaman %d tidak ada)", path, page)
        st["done"] = True
        return
    links = extract_links(content, ctx.article_re, ctx.base)
    if not links:
        st["done"] = True
        return
    new = [l for l in links if not ctx.store.known(l)]
    if new:
        st["stale"] = 0
        process_urls(kat, new, ctx)
    else:
        st["stale"] += 1
        if st["stale"] >= ctx.max_stale:
            log.info(
                "Indeks %s: %d halaman tanpa artikel baru -> ditutup",
                path, st["stale"],
            )
            st["done"] = True
    st["next_page"] = page + 1
    if st["next_page"] > ctx.max_pages:
        st["done"] = True


def crawl(ctx):
    if not ctx.no_rss:
        crawl_rss(ctx)
    log.info("Fase 2: indeks kanal (round-robin antar kelas agar seimbang)")
    while True:
        progressed = False
        for kat in KATEGORI_SUMBER:
            if ctx.store.counts[kat] >= ctx.per_class:
                continue
            path, st = next_source(kat, ctx)
            if st is None:
                continue
            progressed = True
            crawl_index_page(kat, path, st, ctx)
            save_state(ctx.state_path, ctx.state)
        if not progressed:
            break
    checkpoint(ctx, force_sync=True)


# --------------------------------------------------------------------------
# FINALISASI (dedup URL + penyeimbangan kelas)
# --------------------------------------------------------------------------
def finalize(out_dir: Path, per_class, balance, args_dict):
    raw = out_dir / "berita_antara_raw.csv"
    final = out_dir / "berita_antara_final.csv"
    if not raw.exists():
        raise SystemExit("File mentah belum ada. Jalankan pengumpulan dulu.")
    seen, groups, n_raw = set(), {k: [] for k in KATEGORI_SUMBER}, 0
    with open(raw, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            n_raw += 1
            u, k = row.get("url"), row.get("kategori")
            if not u or k not in groups or u in seen:
                continue
            if not (row.get("judul") or "").strip():
                continue
            seen.add(u)
            groups[k].append(row)
    sizes = {k: len(v) for k, v in groups.items()}
    cap = per_class
    if balance and all(sizes.values()):
        cap = min(per_class, min(sizes.values()))
    rows = []
    for k, v in groups.items():
        rows.extend(v[:cap])
    random.Random(42).shuffle(rows)
    with open(final, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "sumber": "ANTARA News (antaranews.com) - indeks kanal + RSS publik",
        "baris_mentah": n_raw,
        "baris_unik_url": len(seen),
        "distribusi_unik_per_kelas": sizes,
        "baris_final": len(rows),
        "batas_per_kelas_final": cap,
        "balanced": balance,
        "rentang_tanggal_publikasi": [
            min(
                (r["tanggal_publikasi"] for r in rows if r["tanggal_publikasi"]),
                default="",
            ),
            max(
                (r["tanggal_publikasi"] for r in rows if r["tanggal_publikasi"]),
                default="",
            ),
        ],
        "waktu_finalisasi_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameter": args_dict,
    }
    (out_dir / "metadata_pengumpulan.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("FINAL: %d baris -> %s", len(rows), final)
    log.info("Distribusi unik sebelum penyeimbangan: %s", sizes)
    if cap < per_class:
        log.warning(
            "Kelas terkecil hanya %d (< target %d per kelas). Jalankan lagi "
            "untuk menambah data atau gunakan --no-balance.",
            cap, per_class,
        )


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pengumpul berita ANTARA (paralel + checkpoint + resume)"
    )
    ap.add_argument(
        "--target-total", type=int, default=50000,
        help="Total artikel (dibagi rata ke 5 kelas). Default 50000",
    )
    ap.add_argument(
        "--contact", default="emailkamu@mhs.unesa.ac.id",
        help="Email kontak untuk User-Agent (WAJIB diganti)",
    )
    ap.add_argument("--out-dir", default="data_berita")
    ap.add_argument(
        "--delay", type=float, default=0.6,
        help="Jeda global antar-permintaan (detik). Default 0.6",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Jumlah thread paralel untuk mengunduh artikel (disarankan 3-5). Default 4",
    )
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--max-body-chars", type=int, default=2500)
    ap.add_argument("--max-stale-pages", type=int, default=15)
    ap.add_argument("--max-pages-per-source", type=int, default=5000)
    ap.add_argument("--no-rss", action="store_true")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--only-finalize", action="store_true")
    ap.add_argument("--base-url", default=BASE_URL_DEFAULT, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "collect.log", encoding="utf-8"),
        ],
    )

    per_class = args.target_total // len(KATEGORI_SUMBER)
    if args.only_finalize:
        finalize(out_dir, per_class, not args.no_balance, vars(args))
        return
    if "emailkamu@" in args.contact:
        log.warning(
            "Ganti --contact dengan email asli agar pengelola situs bisa "
            "menghubungi Anda bila ada masalah."
        )

    base = args.base_url.rstrip("/")
    ua = (
        f"{UA_NAME}/1.0 (+akademik; Proyek Akhir Big Data UNESA; "
        f"kontak: {args.contact})"
    )

    # Konfigurasi Session dengan HTTP Connection Pool agar efisien untuk multithreading
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max(10, args.workers * 2),
        pool_maxsize=max(10, args.workers * 2),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": ua,
        "Accept-Language": "id-ID,id;q=0.9",
        "Accept": "text/html,application/xml;q=0.9,*/*;q=0.5",
    })

    robots = Robots(session, base, UA_NAME)
    delay = args.delay
    cd = robots.crawl_delay()
    if cd and cd > delay:
        delay = float(cd)
        log.info("Crawl-delay robots.txt = %ss -> dipakai", cd)

    ctx = Ctx()
    ctx.base = base
    ctx.article_re = re.compile(r"^" + re.escape(base) + r"/berita/\d+/[^/?#]+$")
    ctx.robots = robots
    ctx.fetcher = Fetcher(session, delay)
    ctx.store = Store(out_dir / "berita_antara_raw.csv", out_dir / "skipped_urls.txt")
    ctx.state_path = out_dir / "state.json"
    ctx.state = load_state(ctx.state_path)
    ctx.per_class = per_class
    ctx.workers = max(1, min(args.workers, 8))
    ctx.checkpoint_every = args.checkpoint_every
    ctx.max_body_chars = args.max_body_chars
    ctx.max_stale = args.max_stale_pages
    ctx.max_pages = args.max_pages_per_source
    ctx.no_rss = args.no_rss

    log.info(
        "Mulai | target %d/kelas | workers %d | parser %s | jeda %.2fs | UA: %s",
        per_class, ctx.workers, BS_PARSER, delay, ua,
    )
    try:
        crawl(ctx)
        completed = True
    except KeyboardInterrupt:
        log.warning("Dihentikan pengguna. Menyimpan checkpoint ...")
        completed = False
    except AbortCrawl as e:
        log.error("BERHENTI AMAN: %s", e)
        completed = False
    finally:
        checkpoint(ctx, force_sync=True)
        ctx.store.close()

    if completed:
        finalize(out_dir, per_class, not args.no_balance, vars(args))
    else:
        log.info(
            "Jalankan perintah yang sama untuk melanjutkan (resume) atau "
            "--only-finalize untuk mengekspor data yang sudah ada."
        )


if __name__ == "__main__":
    main()