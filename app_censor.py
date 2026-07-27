import streamlit as st
import io
import zipfile
import time
import random
import re
import json
import math

import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFilter

# ============== CONFIG ==============
st.set_page_config(page_title="Auto Censor — Names Only", page_icon="🛡️", layout="wide")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
MODEL_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
MAX_RETRY = 3


# ============== GEMINI: DETECT NAMES ONLY ==============
def get_model_fallback_list():
    genai.configure(api_key=GEMINI_API_KEY)
    available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    ordered = []
    for key in MODEL_PRIORITY:
        match = next((m for m in available if key in m), None)
        if match and match not in ordered:
            ordered.append(match)
    return ordered


def detect_names_only(image, status_box=None):
    """
    Use Gemini Vision to detect ONLY account names/usernames.
    Returns list of {"bbox": [x1,y1,x2,y2]}.
    """
    models = get_model_fallback_list()
    if not models:
        raise Exception("No Gemini models available with generateContent support")

    prompt = """Analyze this Facebook group screenshot. Identify ONLY the account names / usernames.

What is a "name":
- The text label immediately to the RIGHT of each circular profile picture avatar.
- Examples: "CaringHedgehog3125", "Mike Oxlong", "Kelvin Lim", "Dash8889", "KnowledgeableHamster7601", "林财家", "Wang Eddie"
- Usually 1–4 words, located at the top of each comment/post entry.
- Some names have a small verification badge, "Author" tag, or role label (e.g., "All-star contributor") immediately next to them — include that badge/label inside the same bounding box.

What is NOT a name (DO NOT include):
- Profile pictures / circular avatars themselves.
- Comment body text (the long paragraph below the name).
- Timestamps like "10h", "2d", "1d".
- Reaction counts like "5", "7", "28", "32".
- Buttons: "Reply", "Share", "View more comments", "View 2 replies".
- Any other UI text or icons.

Precision rules:
1. The bounding box must be TIGHT around the name text only. Do not make it wide enough to cover comment body text below.
2. The bottom edge of the name box should NOT extend past the baseline of the name text.
3. The right edge should stop shortly after the name ends (or after its badge/label).
4. If a timestamp like "· 10h" appears immediately after the name, do NOT include the timestamp in the box.
5. Names are always in the upper portion of each comment row, never in the middle of a paragraph.

Output ONLY a valid JSON array. No markdown code fences, no explanation text:
[
  {"bbox": [left, top, right, bottom]},
  {"bbox": [left, top, right, bottom]}
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

                # Strip markdown fences
                if text.startswith("```"):
                    text = re.sub(r"^```json\s*", "", text)
                    text = re.sub(r"^```\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                raw_boxes = json.loads(text)
                # Normalize to our format
                boxes = []
                for item in raw_boxes:
                    if "bbox" in item and len(item["bbox"]) == 4:
                        boxes.append({"bbox": [int(v) for v in item["bbox"]]})
                return boxes

            except Exception as e:
                last_err = e
                err_msg = str(e)
                if "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    wait_time = delay + random.uniform(0, 5)
                    if status_box is not None:
                        status_box.info(
                            f"⏳ Model **{short_name}** busy (attempt {attempt+1}/{MAX_RETRY}). "
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


# ============== VALIDATION & CLEANUP ==============
def validate_name_boxes(image, boxes):
    """
    Clean up Gemini boxes to ensure they only cover names.
    Filters out boxes that are too large, too wide, or in wrong positions.
    """
    w, h = image.size
    cleaned = []

    for box in boxes:
        try:
            x1, y1, x2, y2 = box["bbox"]
            bw = x2 - x1
            bh = y2 - y1

            # Rule 1: Must be in left half (names are always left-aligned in Facebook comments)
            if x1 > w * 0.65:
                continue

            # Rule 2: Height sanity — name text is typically 15–60px tall
            if bh < 12 or bh > 80:
                continue

            # Rule 3: Width sanity — names are typically 80–450px wide
            if bw < 50 or bw > 500:
                continue

            # Rule 4: Aspect ratio — names are wide and short
            if bw / bh < 1.5 or bw / bh > 25:
                continue

            # Rule 5: Must be in upper portion of image (names are at top of each row)
            # Actually names can be anywhere vertically, but let's be lenient
            # Just ensure it's not at the very bottom edge
            if y1 > h * 0.95:
                continue

            # Rule 6: Add small padding for safety (but not too much)
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.15)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            cleaned.append({"bbox": [x1, y1, x2, y2]})

        except Exception:
            continue

    # Deduplicate overlapping boxes (keep the smaller/tighter one)
    cleaned = deduplicate_boxes(cleaned)
    return cleaned


def deduplicate_boxes(boxes, iou_threshold=0.5):
    """Remove boxes with high overlap, keeping the smaller one (tighter fit)."""
    if not boxes:
        return boxes

    # Sort by area (smaller first)
    boxes_sorted = sorted(boxes, key=lambda b: area(b))
    filtered = []

    for box in boxes_sorted:
        overlap = False
        for kept in filtered:
            if iou(box["bbox"], kept["bbox"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            filtered.append(box)

    return filtered


def area(box):
    x1, y1, x2, y2 = box["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ============== CENSOR ENGINE ==============
def apply_censor(image, boxes):
    """Blur + dark overlay on name boxes only."""
    if not boxes:
        return image.convert("RGB")

    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)

            if x2 > x1 and y2 > y1:
                region = img.crop((x1, y1, x2, y2))
                blurred = region.filter(ImageFilter.GaussianBlur(radius=20))
                img.paste(blurred, (x1, y1))
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 100))
        except Exception:
            continue

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_overlay(image, boxes):
    """Draw blue boxes for preview."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box["bbox"])
            draw.rectangle([x1, y1, x2, y2], outline=(50, 150, 255, 220), width=4)
            draw.text((x1 + 2, max(0, y1 - 18)), "NAME", fill=(50, 150, 255, 220))
        except Exception:
            continue

    return Image.alpha_composite(img, overlay).convert("RGB")


# ============== ZIP BUILDER ==============
def build_zip(processed):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in processed:
            zf.writestr(filename, data)
    buf.seek(0)
    return buf.getvalue()


# ============== UI ==============
st.title("🛡️ Auto Censor — Names Only")
st.caption("AI detects account names → validates coordinates → blurs only names. Profile pictures are left untouched.")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets.")
    st.markdown("Create `.streamlit/secrets.toml` with:")
    st.code('GEMINI_API_KEY = "your-api-key-here"', language="toml")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    show_overlay = st.checkbox("Show detection overlay", value=True)
    blur_radius = st.slider("Blur intensity", 10, 40, 20, 5)

    st.divider()
    st.markdown("""
    **How it works:**
    1. 🔍 Gemini AI scans image for names
    2. ✅ Validator filters out bad boxes (too big, wrong position, etc.)
    3. 🛡️ Only validated name boxes get blurred
    """)

    st.info("💡 Uses Gemini Flash free tier (1,500 req/day).")

# Upload area
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
    st.info("👆 Upload images to start.")
    st.stop()

# Process
st.divider()
process_btn = st.button("🚀 Detect & Censor Names Only", type="primary", use_container_width=True)

if process_btn:
    status = st.empty()
    progress = st.progress(0)

    main_results = []
    comment_results = []
    total = len(all_files)

    for i, (batch_type, uploaded_file) in enumerate(all_files):
        status.info(f"🔍 [{i+1}/{total}] AI detecting names in **{uploaded_file.name}**...")

        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            # Step 1: AI detects names
            raw_boxes = detect_names_only(image, status)

            # Step 2: Validate & clean
            status.info(f"✅ [{i+1}/{total}] Validating {len(raw_boxes)} raw detection(s)...")
            valid_boxes = validate_name_boxes(image, raw_boxes)

            # Step 3: Censor
            censored = apply_censor(image, valid_boxes)

            # Export
            buf = io.BytesIO()
            censored.save(buf, format="PNG")
            buf.seek(0)

            item = {
                "filename": uploaded_file.name,
                "original": image,
                "censored": censored,
                "overlay": draw_overlay(image, valid_boxes) if show_overlay else None,
                "boxes": valid_boxes,
                "raw_count": len(raw_boxes),
                "valid_count": len(valid_boxes),
                "bytes": buf.getvalue(),
            }

            if batch_type == "main":
                main_results.append(item)
            else:
                comment_results.append(item)

            # Rate limit spacing
            if i < total - 1:
                time.sleep(1.5)

        except Exception as e:
            st.error(f"❌ Failed on **{uploaded_file.name}**: {e}")

        progress.progress((i + 1) / total)

    status.empty()
    progress.empty()

    # Display results
    if main_results or comment_results:
        st.divider()
        st.subheader("📋 Results")

        all_results = main_results + comment_results
        total_raw = sum(r["raw_count"] for r in all_results)
        total_valid = sum(r["valid_count"] for r in all_results)
        st.success(
            f"✅ Processed {len(main_results)} main + {len(comment_results)} comment images. "
            f"AI found **{total_raw}** raw → **{total_valid}** validated name box(es)."
        )

        for item in all_results:
            with st.container(border=True):
                st.markdown(
                    f"**{item['filename']}** — "
                    f"Raw: `{item['raw_count']}` | Valid: `{item['valid_count']}`"
                )

                if item["valid_count"] == 0:
                    st.warning("⚠️ No names detected. The AI may have missed them — try re-running.")

                cols = st.columns([1, 1])

                if show_overlay and item["overlay"] is not None:
                    cols[0].image(item["overlay"], caption="Detection Overlay (blue = name)", use_container_width=True)
                else:
                    cols[0].image(item["original"], caption="Original", use_container_width=True)

                cols[1].image(item["censored"], caption="Censored (names only)", use_container_width=True)

        # Download
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

        st.caption("Filenames preserved exactly as uploaded. Exported as PNG.")
