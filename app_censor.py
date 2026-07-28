import streamlit as st
import PIL.Image
import PIL.ImageDraw
import json
import io
import re
import time
import random
import zipfile
import difflib
import numpy as np
import google.generativeai as genai
import easyocr

# ==========================================
# 0. KONFIGURASI MODEL FALLBACK (biar ga kena rate limit)
# ==========================================
# Urutan prioritas model, dari yang paling murah/cepat ke yang lebih besar.
# Kalau satu model kena limit (429/503), otomatis pindah ke model berikutnya.
MODEL_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

MAX_RETRY_PER_MODEL = 3

# ==========================================
# 1. KONFIGURASI SISTEM & AI
# ==========================================
st.set_page_config(page_title="AI Name Censor", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Auto Censor — Nama Akun Sahaja")
st.markdown("""
Aplikasi ini menggunakan **Gemini AI** untuk mengidentifikasi nama pengguna (username) 
yang dicetak **tebal (bold)** pada komentar, lalu menggunakan **EasyOCR** untuk 
menemukan lokasi pixel yang PAS dari nama tersebut, dan menyensornya secara otomatis.

*Kenapa dipisah? Karena Gemini bagus untuk mengenali TEKS, tapi koordinat bounding box-nya 
sering meleset (drift) pada screenshot yang panjang/tinggi. EasyOCR jauh lebih akurat 
untuk lokasi pixel karena membaca gambar aslinya langsung, bukan menebak.*
""")

# Input API Key di Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan AI & Sensor")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Masukkan Gemini API Key Anda:", type="password", value=default_key)
    st.markdown("[Dapatkan API Key Gratis di sini](https://aistudio.google.com/)")

    st.divider()
    st.subheader("🛠️ Penyesuaian Kotak Sensor")
    st.caption("Kotak sekarang dari EasyOCR (pixel asli), jadi biasanya sudah pas. "
               "Slider ini cuma untuk fine-tune kecil kalau masih kurang pas.")

    offset_y = st.slider("Geser Vertikal (Y) px:", min_value=-20, max_value=20, value=0, step=1)
    pad_ukuran = st.slider("Lebar Ekstra (Padding px):", min_value=0, max_value=20, value=3, step=1)

    st.divider()
    match_threshold = st.slider(
        "Ambang Kecocokan Nama (fuzzy match):",
        min_value=0.5, max_value=1.0, value=0.72, step=0.02,
        help="Kalau ada nama yang gagal tersensor, coba turunkan nilai ini sedikit."
    )

    st.divider()
    st.info("✅ Gemini untuk identifikasi nama · EasyOCR untuk lokasi pixel.")


# ==========================================
# 2. LOADER EASYOCR (di-cache agar tidak reload tiap gambar)
# ==========================================
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


# ==========================================
# 3. FUNGSI GEMINI — IDENTIFIKASI NAMA + FALLBACK GANTI MODEL
# ==========================================
def get_model_fallback_list():
    """Susun daftar model yang benar-benar tersedia di akun ini, sesuai MODEL_PRIORITY."""
    available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    ordered = []
    for key in MODEL_PRIORITY:
        match = next((m for m in available if key in m), None)
        if match and match not in ordered:
            ordered.append(match)
    # Jaga-jaga: kalau tidak ada satupun yang cocok, pakai apa saja yang tersedia
    if not ordered:
        ordered = available[:3]
    return ordered


def get_gemini_response(image_pil, prompt, model_names, status_box, max_retry_per_model=MAX_RETRY_PER_MODEL):
    """
    Kirim gambar ke Gemini. Kalau satu model kena limit/sibuk (429/503/RESOURCE_EXHAUSTED),
    tunggu dengan backoff lalu retry; kalau tetap gagal setelah beberapa percobaan,
    pindah ke model berikutnya di daftar fallback.
    """
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    gambar_part = {'mime_type': 'image/png', 'data': img_bytes}

    last_err = None
    for model_name in model_names:
        model = genai.GenerativeModel(model_name)
        delay = 10
        nama_model_pendek = model_name.split("/")[-1]

        for attempt in range(max_retry_per_model):
            try:
                time.sleep(2)
                response = model.generate_content([prompt, gambar_part])
                return response.text, nama_model_pendek
            except Exception as e:
                err_msg = str(e)
                last_err = e
                if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_time = delay + random.uniform(0, 5)
                    status_box.warning(
                        f"⚠️ Model **{nama_model_pendek}** kena limit/sibuk "
                        f"(percobaan {attempt + 1}/{max_retry_per_model}). Menunggu {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    status_box.warning(f"⚠️ Model **{nama_model_pendek}** error: {err_msg[:150]}")
                    break
        status_box.info(f"➡️ Pindah dari model **{nama_model_pendek}** ke model berikutnya...")

    raise Exception(f"Semua model gagal dicoba. Error terakhir: {last_err}")


AIM_PROMPT = """
Analisis gambar tangkapan layar komentar media sosial ini.
Fokus HANYA pada NAMA PENGGUNA (username/account name) yang posisinya selalu ada di baris pertama dan dicetak TEBAL (BOLD).
Jangan mengambil nama yang hanya disebut di dalam isi komentar (misal dibalas/mention).
Urutkan nama sesuai urutan kemunculannya dari atas ke bawah gambar.

Kembalikan hasilnya dalam format JSON murni, tanpa teks penjelasan, tanpa markdown.
JSON harus berupa array of string. Contoh:
["See Toh Kwai Leng", "Pengkok Lim", "Mat Ken"]
"""


# ==========================================
# 4. FUNGSI EASYOCR — UNTUK LOKASI PIXEL YANG AKURAT
# ==========================================
def run_ocr(image_pil, reader):
    """Menjalankan EasyOCR dan mengembalikan list of dict: {text, x1, y1, x2, y2}."""
    img_np = np.array(image_pil)
    raw_results = reader.readtext(img_np, detail=1, paragraph=False)

    results = []
    for bbox, text, conf in raw_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        results.append({
            "text": text,
            "x1": min(xs), "y1": min(ys),
            "x2": max(xs), "y2": max(ys),
        })
    # Urutkan dari atas ke bawah, lalu kiri ke kanan (memudahkan penggabungan baris)
    results.sort(key=lambda r: (round(r["y1"] / 10), r["x1"]))
    return results


def _normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def find_name_bbox(ocr_results, target_name, used_indices, threshold=0.72):
    """
    Mencari bounding box pixel yang paling cocok untuk sebuah nama, dengan menggabungkan
    box OCR yang berdekatan (kata per kata) di baris yang sama sampai teksnya paling mirip
    dengan nama target. Mengembalikan (x1, y1, x2, y2) atau None kalau tidak ketemu.
    """
    target_norm = _normalize(target_name)
    if not target_norm:
        return None

    best_score = 0
    best_box = None
    best_indices = []

    n = len(ocr_results)
    for i in range(n):
        if i in used_indices:
            continue
        combined_text = ""
        combined_indices = []
        x1 = y1 = float("inf")
        x2 = y2 = float("-inf")
        base_y = (ocr_results[i]["y1"] + ocr_results[i]["y2"]) / 2
        base_height = ocr_results[i]["y2"] - ocr_results[i]["y1"]

        # Coba gabungkan sampai maksimal 6 box berturutan yang ada di baris (y) yang sama
        for j in range(i, min(i + 6, n)):
            if j in used_indices:
                continue
            r = ocr_results[j]
            r_mid_y = (r["y1"] + r["y2"]) / 2
            # Harus di baris yang kurang lebih sama secara vertikal
            if abs(r_mid_y - base_y) > max(base_height, 12) * 0.7:
                break

            combined_text += r["text"]
            combined_indices.append(j)
            x1, y1 = min(x1, r["x1"]), min(y1, r["y1"])
            x2, y2 = max(x2, r["x2"]), max(y2, r["y2"])

            score = difflib.SequenceMatcher(None, _normalize(combined_text), target_norm).ratio()
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
                best_indices = list(combined_indices)

    if best_score >= threshold:
        for idx in best_indices:
            used_indices.add(idx)
        return best_box
    return None


def apply_censor_pixel_boxes(image_pil, boxes, pad_ukuran, offset_y):
    """Menerapkan blok hitam langsung dari koordinat pixel (bukan normalized)."""
    censored_image = image_pil.copy()
    draw = PIL.ImageDraw.Draw(censored_image)
    for (x1, y1, x2, y2) in boxes:
        y1_adj = y1 + offset_y
        y2_adj = y2 + offset_y
        draw.rectangle(
            [x1 - pad_ukuran, y1_adj - pad_ukuran, x2 + pad_ukuran, y2_adj + pad_ukuran],
            fill="black"
        )
    return censored_image


# ==========================================
# 5. ALUR KERJA UI STREAMLIT
# ==========================================

uploaded_utama = st.file_uploader(
    "1️⃣ Unggah Gambar Utama Postingan (opsional — TIDAK disensor, cuma ikut disimpan di ZIP)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploader_utama"
)

uploaded_komentar = st.file_uploader(
    "2️⃣ Unggah Gambar Komentar (nama akun akan disensor otomatis)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploader_komentar"
)

uploaded_files = uploaded_komentar  # dipertahankan agar sisa alur kerja di bawah tidak berubah

if uploaded_utama or uploaded_komentar:
    if uploaded_komentar and not api_key:
        st.error("❌ Ada gambar komentar yang diupload, tapi Gemini API Key belum diisi di sidebar.")
        st.stop()

    if uploaded_komentar:
        reader = load_ocr_reader()
        genai.configure(api_key=api_key)
        model_fallback_list = get_model_fallback_list()

    st.divider()
    total_gambar = len(uploaded_utama) + len(uploaded_komentar)
    st.subheader(f"🔄 Memproses {total_gambar} Gambar...")
    if uploaded_komentar:
        st.caption(f"Urutan model fallback: {', '.join(m.split('/')[-1] for m in model_fallback_list)}")

    download_placeholder = st.empty()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:

        # ---------- 0. GAMBAR UTAMA: langsung disimpan, tanpa sensor ----------
        if uploaded_utama:
            st.markdown("#### 1️⃣ Gambar Utama (tidak disensor)")
            for uploaded_file in uploaded_utama:
                filename = uploaded_file.name
                file_bytes = uploaded_file.getvalue()
                st.image(file_bytes, caption=filename, width=250)
                zip_file.writestr(f"Utama/{filename}", file_bytes)

        progress_bar = st.progress(0) if uploaded_komentar else None

        if uploaded_komentar:
            st.markdown("#### 2️⃣ Gambar Komentar (disensor otomatis)")

        for index, uploaded_file in enumerate(uploaded_komentar):
            filename = uploaded_file.name
            response_text = ""
            try:
                # 1. Baca gambar
                image = PIL.Image.open(uploaded_file).convert("RGB")

                # 2. Gemini: identifikasi NAMA saja (tanpa koordinat), dengan fallback ganti model
                status_box = st.empty()
                with st.spinner(f"Gemini membaca nama di {filename}..."):
                    response_text, model_dipakai = get_gemini_response(
                        image, AIM_PROMPT, model_fallback_list, status_box
                    )
                status_box.caption(f"✅ Berhasil pakai model: **{model_dipakai}**")

                json_clean = response_text.replace("```json", "").replace("```", "").strip()
                name_list = json.loads(json_clean)

                # 3. EasyOCR: baca semua teks + lokasi pixel aslinya
                with st.spinner(f"EasyOCR mencari lokasi pixel di {filename}..."):
                    ocr_results = run_ocr(image, reader)

                # 4. Cocokkan tiap nama dari Gemini ke box OCR yang paling mirip
                used_indices = set()
                matched_boxes = []
                unmatched_names = []
                for name in name_list:
                    box = find_name_bbox(ocr_results, name, used_indices, threshold=match_threshold)
                    if box:
                        matched_boxes.append(box)
                    else:
                        unmatched_names.append(name)

                # 5. Terapkan sensor pakai box pixel asli dari OCR
                censored_image = apply_censor_pixel_boxes(image, matched_boxes, pad_ukuran, offset_y)

                # 6. Tampilkan Hasil
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    col1.image(image, caption=f"Asli: {filename}", use_container_width=True)
                    col2.image(censored_image, caption=f"Disensor: {len(matched_boxes)}/{len(name_list)} Nama", use_container_width=True)

                    if unmatched_names:
                        st.warning(f"⚠️ Nama berikut tidak berhasil dicocokkan ke posisi pixel, coba turunkan 'Ambang Kecocokan Nama': {unmatched_names}")

                    with st.expander("🛠️ Lihat Data (Nama dari Gemini & Box dari OCR)"):
                        st.json({"nama_dari_gemini": name_list, "box_pixel_terpakai": matched_boxes})

                # 7. Simpan ke ZIP dengan NAMA FILE ASLI (tanpa prefix)
                img_byte_arr = io.BytesIO()
                censored_image.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"Komentar_Disensor/{filename}", img_byte_arr.getvalue())

            except json.JSONDecodeError:
                st.error(f"❌ Gagal membaca data JSON dari Gemini untuk {filename}. Respon AI:\n\n{response_text}")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan pada {filename}: {str(e)}")

            progress_bar.progress((index + 1) / len(uploaded_komentar))

        zip_buffer.seek(0)
        download_placeholder.download_button(
            label="📦 Unduh Semua Gambar (.zip)",
            data=zip_buffer,
            file_name="hasil_gns_censor.zip",
            mime="application/zip",
            use_container_width=True
        )
        st.success("✅ Semua gambar selesai diproses. Kalau ada nama yang tidak tersensor, cek peringatan di atas dan coba turunkan Ambang Kecocokan Nama.")

else:
    st.info("👆 Silakan unggah Gambar Utama dan/atau Gambar Komentar untuk memulai.")
