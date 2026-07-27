import streamlit as st
import io
import zipfile
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# ============== CONFIG ==============
st.set_page_config(page_title="Auto Censor — Names Only (Avatar-Guided)", page_icon="🛡️", layout="wide")


# ============== DETECT AVATARS (OpenCV) ==============
def detect_avatars(image, sensitivity=1.0):
    """
    Detect circular profile-picture avatars on the left side.
    Returns list of {"cx", "cy", "r", "bbox"} for each avatar.
    """
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    # Resize very large images for speed
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

    # Close small gaps in circle edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    avatars = []
    min_area = int(180 * sensitivity)
    max_area = int(4000 / sensitivity)
    min_circularity = 0.68 * sensitivity

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if not (0.55 < bw / bh < 1.8):
            continue

        # Scale back to original coordinates
        if scale < 1.0:
            x, y, bw, bh = int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)

        cx = x + bw // 2
        cy = y + bh // 2
        radius = max(bw, bh) // 2

        # Avatars are always on the left side
        if cx > w * 0.40:
            continue

        avatars.append({
            "cx": cx,
            "cy": cy,
            "r": int(radius * 1.15),  # slight padding
            "bbox": [max(0, cx - radius - 2), max(0, cy - radius - 2),
                     min(w, cx + radius + 2), min(h, cy + radius + 2)]
        })

    # Deduplicate by center proximity
    avatars = deduplicate_avatars(avatars)
    return avatars


def deduplicate_avatars(avatars, min_dist=35):
    """Remove duplicate avatar detections."""
    filtered = []
    for a in avatars:
        too_close = False
        for b in filtered:
            if math.dist((a["cx"], a["cy"]), (b["cx"], b["cy"])) < min_dist:
                too_close = True
                break
        if not too_close:
            filtered.append(a)
    return filtered


# ============== DERIVE NAME BOXES FROM AVATARS ==============
def derive_name_boxes(avatars, image_width, image_height, name_width=300, name_height_factor=0.65):
    """
    For each detected avatar, create a name box immediately to its right.
    """
    boxes = []
    for av in avatars:
        cx, cy, r = av["cx"], av["cy"], av["r"]

        # Name box starts just right of avatar
        x1 = cx + r + 4
        y1 = max(0, cy - int(r * name_height_factor))
        x2 = min(image_width, x1 + name_width)
        y2 = min(image_height, cy + int(r * name_height_factor) + 4)

        # Sanity: name box must be reasonably sized
        if (x2 - x1) < 60 or (y2 - y1) < 12:
            continue

        boxes.append({"bbox": [x1, y1, x2, y2]})

    return boxes


# ============== CENSOR ENGINE ==============
def apply_censor(image, boxes, blur_radius=22):
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
                blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                img.paste(blurred, (x1, y1))
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 100))
        except Exception:
            continue

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_overlay(image, avatars, name_boxes):
    """Preview: green = avatar (detected, NOT blurred), blue = name (will be blurred)."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Avatars in green outline (these stay visible)
    for av in avatars:
        x1, y1, x2, y2 = av["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(50, 220, 50, 200), width=3)
        draw.text((x1, max(0, y1 - 16)), "AVATAR", fill=(50, 220, 50, 200))

    # Names in blue outline (these get blurred)
    for box in name_boxes:
        x1, y1, x2, y2 = map(int, box["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline=(50, 150, 255, 220), width=4)
        draw.text((x1 + 2, max(0, y1 - 18)), "NAME", fill=(50, 150, 255, 220))

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
st.title("🛡️ Auto Censor — Names Only (Avatar-Guided)")
st.caption("Detects circular avatars with OpenCV → creates name boxes to the right → blurs ONLY names. Profile pictures stay visible. 100% free, no API.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")

    sensitivity = st.slider(
        "Avatar detection sensitivity",
        0.5, 1.5, 1.0, 0.1,
        help="Higher = detects more circles (may include false positives). Lower = stricter."
    )

    name_width = st.slider(
        "Name box width (px)",
        150, 450, 300, 10,
        help="How far right from each avatar to blur. Adjust if names are cut off or too much comment text is blurred."
    )

    name_height = st.slider(
        "Name box height factor",
        0.4, 1.0, 0.65, 0.05,
        help="Name box height relative to avatar radius. Lower = tighter around name only."
    )

    blur_radius = st.slider(
        "Blur intensity",
        10, 40, 22, 2,
        help="Gaussian blur radius. Higher = more obscured."
    )

    show_overlay = st.checkbox("Show detection overlay", value=True)

    st.divider()
    st.success("✅ No API calls. Runs entirely on your machine.")
    st.markdown("""
    **Overlay Legend:**
    - 🟢 **Green** = Avatar detected (stays visible)
    - 🔵 **Blue** = Name box (gets blurred)
    """)

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

# Process button
st.divider()
process_btn = st.button("🚀 Detect & Censor Names Only", type="primary", use_container_width=True)

if process_btn:
    status = st.empty()
    progress = st.progress(0)

    main_results = []
    comment_results = []
    total = len(all_files)

    for i, (batch_type, uploaded_file) in enumerate(all_files):
        status.info(f"🔍 [{i+1}/{total}] Detecting avatars in **{uploaded_file.name}**...")

        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")
            w, h = image.size

            # Step 1: Detect avatars (OpenCV, free, local)
            avatars = detect_avatars(image, sensitivity)

            # Step 2: Derive name boxes from avatar positions
            name_boxes = derive_name_boxes(avatars, w, h, name_width, name_height)

            # Step 3: Censor only names
            censored = apply_censor(image, name_boxes, blur_radius)

            # Export
            buf = io.BytesIO()
            censored.save(buf, format="PNG")
            buf.seek(0)

            item = {
                "filename": uploaded_file.name,
                "original": image,
                "censored": censored,
                "overlay": draw_overlay(image, avatars, name_boxes) if show_overlay else None,
                "avatars": len(avatars),
                "names": len(name_boxes),
                "bytes": buf.getvalue(),
            }

            if batch_type == "main":
                main_results.append(item)
            else:
                comment_results.append(item)

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
        total_avatars = sum(r["avatars"] for r in all_results)
        total_names = sum(r["names"] for r in all_results)
        st.success(
            f"✅ Processed {len(main_results)} main + {len(comment_results)} comment images. "
            f"Detected **{total_avatars}** avatar(s) → **{total_names}** name box(es)."
        )

        for item in all_results:
            with st.container(border=True):
                st.markdown(
                    f"**{item['filename']}** — "
                    f"Avatars: `{item['avatars']}` | Names: `{item['names']}`"
                )

                if item["names"] == 0:
                    st.warning("⚠️ No avatars detected. Try increasing Sensitivity in the sidebar.")

                cols = st.columns([1, 1])

                if show_overlay and item["overlay"] is not None:
                    cols[0].image(item["overlay"], caption="Overlay: 🟢 Avatar | 🔵 Name", use_container_width=True)
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
