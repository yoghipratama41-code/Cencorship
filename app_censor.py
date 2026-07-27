import streamlit as st
import io
import zipfile
import math
import time
import random
import re
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# Optional Gemini import
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ============== CONFIG ==============
st.set_page_config(page_title="Name Censor Only v3", page_icon="🛡️", layout="wide")

# ============== CENSOR ENGINE ==============
def apply_censor(image, boxes):
    """Apply Gaussian blur + dark overlay to bounding boxes (NAMES ONLY)."""
    if not boxes:
        return image.convert("RGB")

    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        # Extra safety: Ensure we only censor names
        if box.get("type") != "name":
            continue
            
        try:
            x1, y1, x2, y2 = map(int, box["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)

            if x2 > x1 and y2 > y1:
                region = img.crop((x1, y1, x2, y2))
                blurred = region.filter(ImageFilter.GaussianBlur(radius=22))
                img.paste(blurred, (x1, y1))
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 120))
        except Exception:
            continue

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_overlay(image, boxes):
    """Draw colored boxes on image for preview (NAMES ONLY)."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        if box.get("type") != "name":
            continue
            
        try:
            x1, y1, x2, y2 = map(int, box["bbox"])
            color = (50, 150, 255, 200) # Blue for names
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label = "Name"
            draw.text((x1 + 2, max(0, y1 - 18)), label, fill=color)
        except Exception:
            continue

    return Image.alpha_composite(img, overlay).convert("RGB")


# ============== DETECTION: SMART CV (OpenCV) ==============
def detect_opencv(image, sensitivity=1.0, name_width=280):
    """
    Detect names ONLY. Uses profile pictures (circles) as anchors, 
    but only returns the coordinates for the text to the right.
    """
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    # Scale down very large images for faster processing
    scale = 1.0
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_array, (new_w, new_h))
    else:
        img_resized = img_array

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 150)

    # Dilate to close small gaps in circle edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = int(180 * sensitivity)
    max_area = int(4000 / sensitivity)
    min_circularity = 0.70 * sensitivity

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        if circularity > min_circularity:
            x, y, bw, bh = cv2.boundingRect(cnt)

            if not (0.55 < bw / bh < 1.8):
                continue

            if scale < 1.0:
                x, y, bw, bh = int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)

            center_x = x + bw // 2
            center_y = y + bh // 2
            radius = max(bw, bh) // 2

            # Anchor must be on the left
            if center_x > w * 0.40:
                continue

            # SKIP appending the profile pic box.
            # Directly calculate the Name box: immediately to the right of the avatar
            r = int(radius * 1.2)
            nx1 = center_x + r + 5
            ny1 = max(0, center_y - r - 5)
            nx2 = min(w, center_x + r + name_width)
            ny2 = min(h, center_y + r + 10)

            name_box = {
                "type": "name",
                "bbox": [nx1, ny1, nx2, ny2],
            }
            boxes.append(name_box)

    boxes = deduplicate_boxes(boxes, min_distance=30)
    return boxes


def deduplicate_boxes(boxes, min_distance=30):
    filtered = []
    for box in boxes:
        cx = (box["bbox"][0] + box["bbox"][2]) / 2
        cy = (box["bbox"][1] + box["bbox"][3]) / 2
        too_close = False
        for existing in filtered:
            ex = (existing["bbox"][0] + existing["bbox"][2]) / 2
            ey = (existing["bbox"][1] + existing["bbox"][3]) / 2
            if math.dist((cx, cy), (ex, ey)) < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(box)
    return filtered


# ============== DETECTION: GEMINI AI VISION ==============
def get_gemini_models():
    available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    priority = [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-3-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ]
    ordered = []
    for key in priority:
        match = next((m for m in available if key in m), None)
        if match and match not in ordered:
            ordered.append(match)
    return ordered


def detect_gemini(image, api_key, status_box=None):
    """Detect using Gemini Vision, strictly filtering for names."""
    genai.configure(api_key=api_key)
    models = get_gemini_models()
    if not models:
        raise Exception("No Gemini models available with generateContent support")

    prompt = """Analyze this Facebook group screenshot. Locate the account names/usernames visible in the image.

Rules:
- Account names: text immediately to the right of each profile picture.
- Include EVERY visible name: main post author and ALL commenters.
- Do NOT include the profile picture coordinates. 
- Ensure the bounding box covers ONLY the name and not overlapping with other comment text below it.
- Return coordinates as [left, top, right, bottom] in pixels, relative to the original image.

Output ONLY a valid JSON array:
[
  {"type": "name", "bbox": [x1, y1, x2, y2]}
]
"""

    last_err = None
    for model_name in models:
        model = genai.GenerativeModel(model_name)
        delay = 10
        short_name = model_name.split("/")[-1]

        for attempt in range(3):
            try:
                time.sleep(1)
                response = model.generate_content([prompt, image])
                text = response.text.strip()

                if text.startswith("```"):
                    text = re.sub(r"^```json\s*", "", text)
                    text = re.sub(r"^```\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                data = json.loads(text)
                
                # Filter strictly for names just in case the AI hallucinates a profile_pic
                return [b for b in data if b.get("type") == "name"]

            except Exception as e:
                last_err = e
                err_msg = str(e)
                if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_time = delay + random.uniform(0, 5)
                    if status_box is not None:
                        status_box.info(
                            f"⏳ Model **{short_name}** busy (attempt {attempt+1}/3). "
                            f"Waiting {wait_time:.1f}s..."
                        )
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    if status_box is not None:
                        status_box.warning(f"⚠️ Model **{short_name}** error: {err_msg[:120]}")
                    break

        if status_box is not None:
            status_box.info(f"➡️ Switching from **{short_name}** to next model...")

    raise Exception(f"All models failed. Last error: {last_err}")


# ============== DETECTION: ROW MODE ==============
def detect_rows(image, row_count, row_y_positions=None):
    """Generate name boxes based on user-specified comment rows."""
    w, h = image.size
    boxes = []

    if row_y_positions and len(row_y_positions) == row_count:
        y_centers = [int(y) for y in row_y_positions]
    else:
        margin = int(h * 0.10)
        usable_h = h - 2 * margin
        step = usable_h / max(row_count, 1)
        y_centers = [int(margin + step * (i + 0.5)) for i in range(row_count)]

    avatar_r = int(h * 0.038)
    avatar_r = max(18, min(avatar_r, 32))

    for y in y_centers:
        # SKIP profile pic box
        # ONLY add Name: rectangle to the right of avatar
        boxes.append({
            "type": "name",
            "bbox": [
                2 + avatar_r * 2 + 4,
                y - avatar_r - 2,
                min(w, 2 + avatar_r * 2 + 300),
                y + avatar_r + 6
            ]
        })

    return boxes


# ============== ZIP BUILDER ==============
def build_zip(processed):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in processed:
            zf.writestr(filename, data)
    buf.seek(0)
    return buf.getvalue()


# ============== UI ==============
st.title("🛡️ Auto Censor v3 — Names Only")
st.caption("Blur account names and usernames. Profile pictures remain untouched.")

with st.sidebar:
    st.header("⚙️ Settings")

    method = st.radio(
        "Detection Method",
        options=["Smart CV (Free, Local)", "AI Vision (Gemini)", "Row Mode (Manual)"],
        index=0,
    )

    if method == "Smart CV (Free, Local)":
        sensitivity = st.slider("Sensitivity", 0.5, 1.5, 1.0, 0.1)
        name_width = st.slider("Name box width (px)", 150, 400, 280, 10,
                               help="Width of the blur area to the right of each avatar.")
        st.success("✅ No API calls. Runs entirely on your machine.")

    elif method == "AI Vision (Gemini)":
        default_key = ""
        try:
            default_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
        gemini_key = st.text_input("Gemini API Key", type="password", value=default_key)
        st.info("💡 Uses Gemini Flash free tier (1,500 requests/day).")

    else:
        st.info("📏 Define comment rows manually.")

    show_overlay = st.checkbox("Show detection overlay in preview", value=True)

    st.divider()
    st.markdown("""
    **Overlay Legend:**
    - 🔵 **Blue** = Name / Username
    """)

st.divider()
st.subheader("📷 Upload Images")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Main Images**")
    main_files = st.file_uploader(
        "Upload main screenshots",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="main",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**Comment Images**")
    comment_files = st.file_uploader(
        "Upload comment screenshots",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="comment",
        label_visibility="collapsed",
    )

all_files = []
if main_files:
    all_files.extend([("main", f) for f in main_files])
if comment_files:
    all_files.extend([("comment", f) for f in comment_files])

if not all_files:
    st.info("👆 Upload images above to get started.")
    st.stop()

row_mode_data = None
if method == "Row Mode (Manual)":
    st.divider()
    st.subheader("📏 Row Mode Setup")
    row_count = st.number_input("Number of comment rows per image", min_value=1, max_value=50, value=3)
    use_custom_y = st.checkbox("Set custom Y positions (comma-separated)", value=False)

    if use_custom_y:
        y_input = st.text_input("Y positions (e.g., 120, 200, 280)", value="")
        if y_input.strip():
            try:
                row_mode_data = [int(y.strip()) for y in y_input.split(",")]
                if len(row_mode_data) != row_count:
                    st.warning(f"You specified {row_count} rows but provided {len(row_mode_data)} Y positions. Using evenly-spaced fallback.")
                    row_mode_data = None
            except ValueError:
                st.error("Invalid Y positions. Use numbers separated by commas.")
                st.stop()
        else:
            row_mode_data = None

st.divider()
process_btn = st.button("🚀 Process & Censor All Images", type="primary", use_container_width=True)

if process_btn:
    if method == "AI Vision (Gemini)" and not gemini_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    main_results = []
    comment_results = []
    total = len(all_files)

    for i, (batch_type, uploaded_file) in enumerate(all_files):
        status.info(f"🔍 [{i+1}/{total}] Processing **{uploaded_file.name}**...")

        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            if method == "Smart CV (Free, Local)":
                boxes = detect_opencv(image, sensitivity, name_width)
            elif method == "AI Vision (Gemini)":
                boxes = detect_gemini(image, gemini_key, status)
            else:
                boxes = detect_rows(image, row_count, row_mode_data)

            censored = apply_censor(image, boxes)

            buf = io.BytesIO()
            censored.save(buf, format="PNG")
            buf.seek(0)

            item = {
                "filename": uploaded_file.name,
                "original": image,
                "censored": censored,
                "overlay": draw_overlay(image, boxes) if show_overlay else None,
                "boxes": boxes,
                "bytes": buf.getvalue(),
            }

            if batch_type == "main":
                main_results.append(item)
            else:
                comment_results.append(item)

            if method == "AI Vision (Gemini)" and i < total - 1:
                time.sleep(1.5)

        except Exception as e:
            st.error(f"❌ Failed on **{uploaded_file.name}**: {e}")

        progress.progress((i + 1) / total)

    status.empty()
    progress.empty()

    if main_results or comment_results:
        st.divider()
        st.subheader("📋 Results Preview")

        all_results = main_results + comment_results
        total_regions = sum(len(r["boxes"]) for r in all_results)
        st.success(
            f"✅ Processed {len(main_results)} main + {len(comment_results)} comment images. "
            f"**{total_regions}** name(s) censored."
        )

        for item in all_results:
            with st.container(border=True):
                st.markdown(f"**{item['filename']}** — `{len(item['boxes'])}` name(s) detected")

                if len(item["boxes"]) == 0:
                    st.warning("⚠️ No regions detected. Try increasing Sensitivity (Smart CV) or switching to Row Mode.")

                cols = st.columns([1, 1])

                if show_overlay and item["overlay"] is not None:
                    cols[0].image(item["overlay"], caption="Detection Overlay", use_container_width=True)
                else:
                    cols[0].image(item["original"], caption="Original", use_container_width=True)

                cols[1].image(item["censored"], caption="Censored Result", use_container_width=True)

        st.divider()
        st.subheader("📥 Download")
        dl_cols = st.columns(2)

        if main_results:
            main_zip = build_zip([(r["filename"], r["bytes"]) for r in main_results])
            dl_cols[0].download_button(
                "📦 Download main.zip",
                data=main_zip,
                file_name="main.zip",
                mime="application/zip",
                use_container_width=True,
            )

        if comment_results:
            comment_zip = build_zip([(r["filename"], r["bytes"]) for r in comment_results])
            dl_cols[1].download_button(
                "📦 Download comment.zip",
                data=comment_zip,
                file_name="comment.zip",
                mime="application/zip",
                use_container_width=True,
            )
