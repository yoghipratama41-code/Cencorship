import streamlit as st
import PIL.Image
import PIL.ImageDraw
import json
import io
import zipfile
import time
import google.generativeai as genai

# ==========================================
# 1. KONFIGURASI SISTEM & AI
# ==========================================
st.set_page_config(page_title="AI Name Censor", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Auto Censor — Nama Akun Sahaja")
st.markdown("""
Aplikasi ini menggunakan **Gemini AI Vision** untuk membaca teks pada tangkapan layar, 
mengidentifikasi nama pengguna yang dicetak **tebal (bold)** pada komentar, 
dan menyensornya secara otomatis.
""")

# Input API Key di Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan AI & Sensor")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Masukkan Gemini API Key Anda:", type="password", value=default_key)
    st.markdown("[Dapatkan API Key Gratis di sini](https://aistudio.google.com/)")
    
    st.divider()
    st.subheader("🛠️ Penyesuaian Posisi")
    st.caption("Karena pakai model Flash (Free), gunakan ini jika kotak meleset ke bawah/atas.")
    
    # Offset Y diset default -20 karena Flash biasanya meleset ke bawah
    offset_y = st.slider("Geser Vertikal (Y):", min_value=-50, max_value=50, value=-20, step=1, 
                         help="Geser ke kiri (minus) untuk menaikkan kotak. Geser ke kanan (plus) untuk menurunkan.")
    
    pad_ukuran = st.slider("Lebar Ekstra (Padding px):", min_value=0, max_value=20, value=2, step=1)
    
    st.divider()
    st.info("✅ Dioptimalkan untuk Free Plan (Gemini 1.5 Flash + Anti-Ratelimit).")

# ==========================================
# 2. FUNGSI INTI (DETEKSI & SENSOR)
# ==========================================

def get_gemini_response(image_pil, prompt, api_key):
    """Mengirim gambar ke Gemini (Fokus Flash untuk Free Plan)."""
    genai.configure(api_key=api_key)
    
    available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    
    # PRIORITAS: Gemini Flash untuk stabilitas Free Plan
    chosen_model = "gemini-1.5-flash-latest" 
    
    for m_name in available_models:
        if "flash" in m_name:
            chosen_model = m_name.replace("models/", "")
            break
            
    model = genai.GenerativeModel(chosen_model)
    
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    response = model.generate_content([
        prompt, 
        {'mime_type': 'image/png', 'data': img_bytes}
    ])
    return response.text

def apply_censor_to_coordinates(image_pil, boxes, pad_ukuran, offset_y):
    """Menerapkan Blok Hitam dengan kemampuan pergeseran vertikal (Offset Y)."""
    width, height = image_pil.size
    censored_image = image_pil.copy()
    draw = PIL.ImageDraw.Draw(censored_image)
    
    for box in boxes:
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            
            if all(v <= 1.0 for v in box):
                left = xmin * width
                right = xmax * width
                top = ymin * height
                bottom = ymax * height
            elif all(v <= 1000 for v in box):
                left = (xmin / 1000) * width
                right = (xmax / 1000) * width
                top = (ymin / 1000) * height
                bottom = (ymax / 1000) * height
            else:
                left, right = xmin, xmax
                top, bottom = ymin, ymax

            x1, x2 = sorted([left, right])
            y1, y2 = sorted([top, bottom])
            
            y1_adjusted = y1 + offset_y
            y2_adjusted = y2 + offset_y
            
            draw.rectangle([x1 - pad_ukuran, y1_adjusted - pad_ukuran, x2 + pad_ukuran, y2_adjusted + pad_ukuran], fill="black")
            
    return censored_image

# ==========================================
# 3. PROMPT KHUSUS UNTUK GEMINI
# ==========================================
AIM_PROMPT = """
Analisis gambar tangkapan layar komentar media sosial ini.
Fokus HANYA pada NAMA PENGGUNA (username/account name) yang posisinya selalu ada di baris pertama dan dicetak TEBAL (BOLD).
Jangan mengambil bounding box untuk isi teks komentar biasa di bawah nama tersebut.
Bounding box [ymin, xmin, ymax, xmax] HARUS ketat dan presisi mengelilingi nama tersebut, jangan meluber ke baris bawahnya.

Kembalikan hasilnya dalam format JSON murni, tanpa teks penjelasan.
JSON harus berupa array dari objek. Setiap objek memiliki kunci 'nama' dan kunci 'kotak' (array koordinat dinormalisasi [ymin, xmin, ymax, xmax] dengan presisi desimal).

Contoh:
[
  {"nama": "UserA", "kotak": [0.120, 0.050, 0.135, 0.200]}
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
        total_files = len(uploaded_files)
        
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                # 1. Baca gambar
                image = PIL.Image.open(uploaded_file).convert("RGB")
                filename = uploaded_file.name
                
                # 2. Panggil AI
                with st.spinner(f"AI (Flash) sedang membaca {filename}..."):
                    response_text = get_gemini_response(image, AIM_PROMPT, api_key)
                
                # 3. Parse JSON
                json_clean = response_text.replace("```json", "").replace("```", "").strip()
                name_data = json.loads(json_clean)
                
                # 4. Ekstrak koordinat
                coordinates = [item['kotak'] for item in name_data if 'kotak' in item]
                
                # 5. Terapkan Sensor dengan Offset
                censored_image = apply_censor_to_coordinates(image, coordinates, pad_ukuran, offset_y)
                
                # 6. Tampilkan Hasil
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    col1.image(image, caption=f"Asli: {filename}", use_container_width=True)
                    col2.image(censored_image, caption=f"Disensor: {len(coordinates)} Nama", use_container_width=True)
                    
                    with st.expander("🛠️ Lihat Data JSON (Koordinat AI)"):
                        st.json(name_data)
                
                # 7. Simpan ke ZIP
                img_byte_arr = io.BytesIO()
                censored_image.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"censored_{filename}", img_byte_arr.getvalue())
                
                # 8. ANTI-RATELIMIT UNTUK FREE PLAN (Jeda 4 detik antar gambar)
                if index < total_files - 1:
                    with st.spinner("⏳ Menunggu 4 detik untuk mencegah limit API Google..."):
                        time.sleep(4)
                
            except json.JSONDecodeError:
                st.error(f"❌ Gagal membaca data JSON dari AI untuk {filename}. Respon AI:\n\n{response_text}")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan pada {filename}: {str(e)}")
                
            progress_bar.progress((index + 1) / total_files)
            
        zip_buffer.seek(0)
        download_placeholder.download_button(
            label="📦 Unduh Semua Gambar Disensor (.zip)",
            data=zip_buffer,
            file_name="semua_gambar_disensor.zip",
            mime="application/zip",
            use_container_width=True
        )
        st.success("✅ Selesai! Atur slider 'Geser Vertikal' jika kotak hitam kurang pas.")

elif not uploaded_files:
    st.info("👆 Silakan unggah satu atau beberapa gambar untuk memulai.")
