import streamlit as st
import PIL.Image
import PIL.ImageDraw
import json
import io
import zipfile
import google.generativeai as genai

# ==========================================
# 1. KONFIGURASI SISTEM & AI
# ==========================================
st.set_page_config(page_title="AI Name Censor", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Auto Censor — Nama Akun Sahaja")
st.markdown("""
Aplikasi ini menggunakan **Gemini AI Vision** untuk membaca teks pada tangkapan layar, 
mengidentifikasi nama pengguna yang dicetak **tebal (bold)** pada komentar, 
dan menyensornya secara otomatis dengan kotak hitam solid.
""")

# Input API Key di Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan AI")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Masukkan Gemini API Key Anda:", type="password", value=default_key)
    st.markdown("[Dapatkan API Key Gratis di sini](https://aistudio.google.com/)")
    
    pad_ukuran = st.slider("Lebar Ekstra Sensor (Padding px):", min_value=0, max_value=20, value=2, step=1)
    st.divider()
    st.info("✅ Menggunakan model Gemini 1.5 Flash untuk performa vision terbaik.")

# ==========================================
# 2. FUNGSI INTI (DETEKSI & SENSOR)
# ==========================================

def get_gemini_response(image_pil, prompt, api_key):
    """Mengirim gambar ke Gemini dengan deteksi model dinamis."""
    genai.configure(api_key=api_key)
    
    # 1. Cek model apa saja yang aktif dan tersedia
    available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    
    # 2. Cari model yang ada kata "flash"
    chosen_model = "gemini-1.5-flash-latest" # Fallback default
    for m_name in available_models:
        if "flash" in m_name:
            chosen_model = m_name.replace("models/", "")
            break
            
    model = genai.GenerativeModel(chosen_model)
    
    # Konversi PIL ke bytes
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Kirim ke model
    response = model.generate_content([
        prompt, 
        {'mime_type': 'image/png', 'data': img_bytes}
    ])
    return response.text

def apply_censor_to_coordinates(image_pil, boxes, pad_ukuran):
    """Menerapkan Blok Hitam pada area koordinat yang diberikan."""
    width, height = image_pil.size
    censored_image = image_pil.copy()
    draw = PIL.ImageDraw.Draw(censored_image)
    
    for box in boxes:
        if len(box) == 4:
            # Deteksi apakah koordinatnya dinormalisasi (0-1000) atau piksel asli
            is_normalized = all(v <= 1000 for v in box)
            
            if is_normalized:
                ymin, xmin, ymax, xmax = box
                left = (xmin / 1000) * width
                right = (xmax / 1000) * width
                top = (ymin / 1000) * height
                bottom = (ymax / 1000) * height
            else:
                ymin, xmin, ymax, xmax = box
                left, right = xmin, xmax
                top, bottom = ymin, ymax

            # PENGAMAN: Urutkan agar kotak tidak terbalik (lebar 0 piksel)
            x1, x2 = sorted([left, right])
            y1, y2 = sorted([top, bottom])
            
            # Gambar Kotak Hitam Solid dengan ekstra padding
            draw.rectangle([x1 - pad_ukuran, y1 - pad_ukuran, x2 + pad_ukuran, y2 + pad_ukuran], fill="black")
            
    return censored_image

# ==========================================
# 3. PROMPT KHUSUS UNTUK GEMINI
# ==========================================
AIM_PROMPT = """
Analisis gambar tangkapan layar komentar media sosial ini.
Identifikasi setiap nama pengguna (username/account name) yang dicetak TEBAL (BOLD) pada bagian komentar.
Jangan identifikasi teks komentar biasa, jangan identifikasi timestamp. Hanya nama yang ditebalkan.

Kembalikan hasilnya dalam format JSON murni, tanpa teks penjelasan lain di luar JSON.
JSON harus berupa array dari objek. Setiap objek memiliki kunci 'nama' (teks nama) dan kunci 'kotak' (array koordinat yang dinormalisasi [ymin, xmin, ymax, xmax]).

Contoh Output:
[
  {"nama": "UserA", "kotak": [120, 50, 150, 200]},
  {"nama": "UserB", "kotak": [300, 50, 330, 220]}
]
"""

# ==========================================
# 4. ALUR KERJA UI STREAMLIT
# ==========================================

uploaded_files = st.file_uploader(
    "Unggah tangkapan layar komentar (PNG/JPG)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    if not api_key:
        st.error("❌ Silakan masukkan Gemini API Key Anda di sidebar untuk melanjutkan.")
        st.stop()
        
    st.divider()
    st.subheader(f"🔄 Memproses {len(uploaded_files)} Gambar...")
    
    download_placeholder = st.empty()
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        progress_bar = st.progress(0)
        
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                # 1. Baca gambar
                image = PIL.Image.open(uploaded_file).convert("RGB")
                filename = uploaded_file.name
                
                # 2. Panggil AI
                with st.spinner(f"AI sedang membaca {filename}..."):
                    response_text = get_gemini_response(image, AIM_PROMPT, api_key)
                
                # 3. Parse JSON
                json_clean = response_text.replace("```json", "").replace("```", "").strip()
                name_data = json.loads(json_clean)
                
                # 4. Ekstrak koordinat
                coordinates = [item['kotak'] for item in name_data if 'kotak' in item]
                
                # 5. Terapkan Sensor Kotak Hitam
                censored_image = apply_censor_to_coordinates(image, coordinates, pad_ukuran)
                
                # 6. Tampilkan Hasil
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    col1.image(image, caption=f"Asli: {filename}", use_container_width=True)
                    col2.image(censored_image, caption=f"Disensor: {len(coordinates)} Nama", use_container_width=True)
                    
                    # Debugger: Lihat hasil JSON AI
                    with st.expander("🛠️ Lihat Data JSON (Koordinat AI)"):
                        st.json(name_data)
                
                # 7. Simpan ke ZIP
                img_byte_arr = io.BytesIO()
                censored_image.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"censored_{filename}", img_byte_arr.getvalue())
                
            except json.JSONDecodeError:
                st.error(f"❌ Gagal membaca data JSON dari AI untuk {filename}. Respon AI: {response_text}")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan pada {filename}: {str(e)}")
                
            progress_bar.progress((index + 1) / len(uploaded_files))
            
        zip_buffer.seek(0)
        download_placeholder.download_button(
            label="📦 Unduh Semua Gambar Disensor (.zip)",
            data=zip_buffer,
            file_name="semua_gambar_disensor.zip",
            mime="application/zip",
            use_container_width=True
        )
        st.success("✅ Semua gambar selesai diproses.")

elif not uploaded_files:
    st.info("👆 Silakan unggah satu atau beberapa gambar untuk memulai.")
