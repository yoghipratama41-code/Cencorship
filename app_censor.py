import streamlit as st
import PIL.Image
import PIL.ImageDraw
import PIL.Imagefilter
import json
import io
import zipfile
import time
from google.generativeai import GenerativeModel
import google.generativeai as genai

# ==========================================
# 1. KONFIGURASI SISTEM & AI
# ==========================================
st.set_page_config(page_title="AI Name Censor", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Auto Censor — Nama Akun Sahaja")
st.markdown("""
Aplikasi ini menggunakan **Gemini AI Vision** untuk membaca teks pada tangkapan layar, 
mengidentifikasi nama pengguna yang dicetak **tebal (bold)** pada komentar, 
dan menyensornya secara otomatis. *Profile picture* tetap utuh.
""")

# Input API Key di Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan AI")
    api_key = st.text_input("Masukkan Gemini API Key Anda:", type="password")
    st.markdown("[Dapatkan API Key Gratis di sini](https://aistudio.google.com/)")
    
    blur_radius = st.slider("Intensitas Blur:", min_value=5, max_value=50, value=25, step=5)
    st.divider()
    st.info("✅ 100% Menggunakan AI Vision. Tanpa teknik geometri OpenCV.")

# ==========================================
# 2. FUNGSI INTI (DETEKSI & SENSOR)
# ==========================================

def get_gemini_response(image_pil, prompt, api_key):
    """Mengirim gambar ke Gemini dan mendapatkan respon teks (JSON)."""
    genai.configure(api_key=api_key)
    model = GenerativeModel('gemini-1.5-flash') # Model flash lebih cepat & murah untuk vision
    
    # Konversi PIL ke bytes untuk API
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Kirim ke model
    response = model.generate_content([
        prompt, 
        {'mime_type': 'image/png', 'data': img_bytes}
    ])
    return response.text

def apply_censor_to_coordinates(image_pil, normalized_boxes, blur_radius):
    """Menerapkan Gaussian Blur pada area koordinat yang diberikan."""
    width, height = image_pil.size
    censored_image = image_pil.copy()
    draw = PIL.ImageDraw.Draw(censored_image)
    
    for box in normalized_boxes:
        # Gemini mengembalikan koordinat normalisasi (0-1000).
        # Kita perlu mengonversinya kembali ke piksel asli gambar[cite: 1.1.2].
        # Format Gemini biasanya: [ymin, xmin, ymax, xmax][cite: 1.1.2]
        ymin, xmin, ymax, xmax = box
        
        # Konversi ke piksel asli
        left = xmin * width / 1000
        top = ymin * height / 1000
        right = xmax * width / 1000
        bottom = ymax * height / 1000
        
        # Buat kotak koordinat (x1, y1, x2, y2)
        target_box = (left, top, right, bottom)
        
        # 1. Potong area nama
        face_crop = image_pil.crop(target_box)
        
        # 2. Terapkan Gaussian Blur pada potongan tersebut
        blurred_face = face_crop.filter(PIL.Imagefilter.GaussianBlur(radius=blur_radius))
        
        # 3. Tempelkan kembali potongan yang buram ke gambar asli
        censored_image.paste(blurred_face, (int(left), int(top)))
        
    return censored_image

# ==========================================
# 3. PROMPT KHUSUS UNTUK GEMINI
# ==========================================
# Prompt ini sangat krusial agar Gemini mengembalikan JSON murni dengan koordinat[cite: 1.1.2].
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

# Kolom Unggah Gambar
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
    
    # Placeholder untuk tombol unduh massal
    download_placeholder = st.empty()
    
    # Siapkan ZIP di memori untuk unduhan massal
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        progress_bar = st.progress(0)
        
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                # 1. Baca gambar sebagai PIL
                image = PIL.Image.open(uploaded_file).convert("RGB")
                filename = uploaded_file.name
                
                # 2. Panggil AI untuk mendapatkan koordinat
                with st.spinner(f"AI sedang membaca {filename}..."):
                    response_text = get_gemini_response(image, AIM_PROMPT, api_key)
                
                # 3. Parse JSON dari respon AI
                # Terkadang AI menambahkan markdown ```json ... ```, kita hapus jika ada.
                json_clean = response_text.replace("```json", "").replace("```", "").strip()
                name_data = json.loads(json_clean)
                
                # 4. Ekstrak hanya koordinat kotak
                coordinates = [item['kotak'] for item in name_data if 'kotak' in item]
                
                # 5. Terapkan Sensor Pillow
                censored_image = apply_censor_to_coordinates(image, coordinates, blur_radius)
                
                # 6. Tampilkan Hasil
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    col1.image(image, caption=f"Asli: {filename}", use_container_width=True)
                    col2.image(censored_image, caption=f"Disensor: {len(coordinates)} Nama", use_container_width=True)
                
                # 7. Simpan ke ZIP
                img_byte_arr = io.BytesIO()
                censored_image.save(img_byte_arr, format='PNG')
                zip_file.writestr(f"censored_{filename}", img_byte_arr.getvalue())
                
            except json.JSONDecodeError:
                st.error(f"❌ Gagal membaca data JSON dari AI untuk {filename}. Respon AI: {response_text}")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan pada {filename}: {str(e)}")
                
            progress_bar.progress((index + 1) / len(uploaded_files))
            
        # Tombol Unduh Massal
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
