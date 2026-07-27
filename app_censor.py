import streamlit as st
import io
import zipfile
import time
import random
import re
import json

import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFilter

# ============== CONFIG ==============
st.set_page_config(page_title="Auto Censor - Facebook Screenshot", page_icon="🛡️", layout="centered")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
MODEL_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
MAX_RETRY = 3


# ============== HELPER: MODEL FALLBACK ==============
def get_model_fallback_list():
    genai.configure(api_key=GEMINI_API_KEY)
    available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    ordered = []
    for key in MODEL_PRIORITY:
        match = next((m for m in available if key in m), None)
        if match and match not in ordered:
            ordered.append(match)
    return ordered


def detect_regions(image, status_text=None):
    """Detect profile pics and names using Gemini Vision with model fallback."""
    models = get_model_fallback_list()
    if not models:
        raise Exception("No Gemini models available with generateContent support")

    prompt = """
    Analyze this Facebook group screenshot. Identify ALL profile pictures (circular avatars) and account names/usernames visible in the image.

    Rules:
    - Profile pictures: circular avatars on the left side of each post/comment entry.
    - Account names: text immediately to the right of each profile picture (e.g., "Mike Lim", "Dash8889", "KnowledgeableHamster7601", "林财家").
    - Include EVERY visible profile picture and name: main post author (if visible at top) and ALL commenters.
    - If a name has an "Author" badge, verification badge, or role label next to it, include that badge inside the name bounding box.
    - Do NOT include comment body text, timestamps, reaction buttons, "View more comments", or "Reply / Share" buttons.
    - Return coordinates as [left, top, right, bottom] in pixels, relative to the original image.

    Output ONLY a valid JSON array. No markdown code fences, no explanation:
    [
      {"type": "profile_pic", "bbox": [x1, y1, x2, y2]},
      {"type": "name", "bbox": [x1, y1, x2, y2]}
    ]
    """

    last_err = None
    for model_name in models:
        model = genai.GenerativeModel(model_name)
        delay = 10
        short_name = model_name.split("/")[-1]

        for attempt in range(MAX_RETRY):
            try:
                time.sleep(1)
                response = model.generate_content([prompt, image])
                text = response.text.strip()

                # Strip markdown if present
                if text.startswith("```"):
                    text = re.sub(r"^```json\s*", "", text)
                    text = re.sub(r"^```\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                boxes = json.loads(text)
                return boxes

            except Exception as e:
                last_err = e
                err_msg = str(e)
                if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_time = delay + random.uniform(0, 5)
                    if status_text is not None:
                        status_text.info(
                            f"⏳ Model **{short_name}** busy (attempt {attempt+1}/{MAX_RETRY}). "
                            f"Waiting {wait_time:.1f}s..."
                        )
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    if status_text is not None:
                        status_text.warning(f"⚠️ Model **{short_name}** error: {err_msg[:120]}")
                    break

        if status_text is not None:
            status_text.info(f"➡️ Switching from **{short_name}** to next model...")

    raise Exception(f"All models failed. Last error: {last_err}")


def apply_censor(image, boxes):
    """Apply Gaussian blur + semi-transparent black overlay to bounding boxes."""
    if not boxes:
        return image.convert("RGB")

    img = image.convert("RGBA")
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box.get("bbox", [0, 0, 0, 0]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 > x1 and y2 > y1:
                # Blur region
                region = img.crop((x1, y1, x2, y2))
                blurred = region.filter(ImageFilter.GaussianBlur(radius=18))
                img.paste(blurred, (x1, y1))
                # Dark overlay for extra opacity
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 110))
        except Exception:
            continue

    return Image.alpha_composite(img, overlay).convert("RGB")


def process_batch(images, status_container, progress_bar):
    """Process a list of uploaded files and return [(filename, bytes)]."""
    results = []
    total = len(images)

    for i, uploaded_file in enumerate(images):
        status_container.info(f"🔍 [{i+1}/{total}] Detecting: **{uploaded_file.name}**")

        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        # AI detection
        boxes = detect_regions(image, status_container)
        detected_count = len([b for b in boxes if b.get("type") in ("profile_pic", "name")])

        status_container.info(
            f"🛡️ [{i+1}/{total}] Censoring: **{uploaded_file.name}** — {detected_count} region(s) found"
        )

        # Apply censor
        censored = apply_censor(image, boxes)

        # Export to bytes (keep original filename exactly)
        buf = io.BytesIO()
        censored.save(buf, format="PNG")
        buf.seek(0)

        results.append((uploaded_file.name, buf.getvalue()))
        progress_bar.progress((i + 1) / total)

        # Small delay to avoid rate limits between images
        if i < total - 1:
            time.sleep(2)

    return results


def build_zip(processed_images):
    """Build an in-memory ZIP archive preserving original filenames."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in processed_images:
            zf.writestr(filename, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============== UI ==============
st.title("🛡️ Auto Censor — Facebook Screenshot")
st.caption("Upload Facebook group screenshots → AI detects profile pics & account names → Auto blur → Download ZIP")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets.")
    st.markdown("Please create `.streamlit/secrets.toml` with:")
    st.code('GEMINI_API_KEY = "your-api-key-here"', language="toml")
    st.stop()

# Session state for download persistence
if "main_zip" not in st.session_state:
    st.session_state.main_zip = None
if "comment_zip" not in st.session_state:
    st.session_state.comment_zip = None
if "preview_main" not in st.session_state:
    st.session_state.preview_main = None

st.divider()

# ---- MAIN IMAGES ----
st.subheader("📷 Main Images")
main_files = st.file_uploader(
    "Upload main post screenshots (post + top comments)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="main_uploader",
)

if main_files:
    c1, c2 = st.columns([1, 1])
    run_main = c1.button("🚀 Censor Main Images", type="primary", key="btn_main")
    clear_main = c2.button("🗑️ Clear", key="clr_main")

    if clear_main:
        st.session_state.main_zip = None
        st.session_state.preview_main = None
        st.rerun()

    if run_main:
        status = st.empty()
        progress = st.progress(0)

        with st.spinner("Processing main images with AI..."):
            try:
                processed = process_batch(main_files, status, progress)
                st.session_state.main_zip = build_zip(processed)

                # Save first image for preview
                first_name, first_data = processed[0]
                st.session_state.preview_main = first_data

                st.success(f"✅ {len(processed)} main image(s) censored!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.main_zip = None

    if st.session_state.preview_main:
        st.image(st.session_state.preview_main, caption="Preview: first censored main image", use_container_width=True)

    if st.session_state.main_zip:
        st.download_button(
            label="📥 Download main.zip",
            data=st.session_state.main_zip,
            file_name="main.zip",
            mime="application/zip",
            key="dl_main",
        )

st.divider()

# ---- COMMENT IMAGES ----
st.subheader("💬 Comment Images")
comment_files = st.file_uploader(
    "Upload comment thread screenshots",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="comment_uploader",
)

if comment_files:
    c1, c2 = st.columns([1, 1])
    run_comment = c1.button("🚀 Censor Comment Images", type="primary", key="btn_comment")
    clear_comment = c2.button("🗑️ Clear", key="clr_comment")

    if clear_comment:
        st.session_state.comment_zip = None
        st.rerun()

    if run_comment:
        status = st.empty()
        progress = st.progress(0)

        with st.spinner("Processing comment images with AI..."):
            try:
                processed = process_batch(comment_files, status, progress)
                st.session_state.comment_zip = build_zip(processed)
                st.success(f"✅ {len(processed)} comment image(s) censored!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.comment_zip = None

    if st.session_state.comment_zip:
        st.download_button(
            label="📥 Download comment.zip",
            data=st.session_state.comment_zip,
            file_name="comment.zip",
            mime="application/zip",
            key="dl_comment",
        )

st.divider()
st.caption("💡 Filenames inside ZIP are preserved exactly as uploaded. Images are exported as PNG to maintain quality after blurring.")
