import streamlit as st
import io
import zipfile
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# ============== CONFIG ==============
st.set_page_config(page_title="Auto Censor — Names Only (Text Detect)", page_icon="🛡️", layout="wide")


# ============== NAME DETECTION (OpenCV Text Lines) ==============
def detect_names(image, cfg):
    """
    Detect name text regions using morphological line detection.
    Names = short horizontal text lines at the top of each comment row,
    positioned to the right of the avatar margin.
    """
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold: text becomes white on black
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing with wide horizontal kernel → connects letters into lines
    kernel_w = max(10, int(w * cfg["kernel_ratio"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, cfg["kernel_h"]))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Optional: dilate vertically a bit to catch bold names
    if cfg["dilate_v"] > 0:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, cfg["dilate_v"]))
        closed = cv2.dilate(closed, v_kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Position: must be to the right of avatar space
        if x < cfg["min_x"]:
            continue
        if x > w * cfg["max_x_ratio"]:
            continue

        # Size filters
        if bh < cfg["min_h"] or bh > cfg["max_h"]:
            continue
        if bw < cfg["min_w"] or bw > cfg["max_w"]:
            continue

        # Aspect ratio: names are wide and short
        aspect = bw / bh if bh > 0 else 0
        if aspect < cfg["min_aspect"] or aspect > cfg["max_aspect"]:
            continue

        # Text density check: names should have decent ink coverage
        roi = binary[y:y+bh, x:x+bw]
        fill = cv2.countNonZero(roi) / (bw * bh)
        if fill < cfg["min_fill"]:
            continue

        candidates.append({
            "bbox": [x, y, x+bw, y+bh],
            "cy": y + bh // 2,
            "top": y,
        })

    if not candidates:
        return []

    # Sort by vertical position (top to bottom)
    candidates.sort(key=lambda c: c["top"])

    # Cluster: group lines that are close vertically (same comment row)
    # Then keep only the TOPMOST line in each cluster → that's the name
    merged = []
    cluster = [candidates[0]]
    threshold = cfg["cluster_threshold"]

    for c in candidates[1:]:
        if abs(c["top"] - cluster[-1]["top"]) < threshold:
            cluster.append(c)
        else:
            # Keep topmost in cluster
            cluster.sort(key=lambda x: x["top"])
            merged.append(cluster[0])
            cluster = [c]

    # Last cluster
    cluster.sort(key=lambda x: x["top"])
    merged.append(cluster[0])

    return [{"bbox": m["bbox"]} for m in merged]


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


def draw_overlay(image, boxes):
    """Blue boxes for preview."""
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
st.title("🛡️ Auto Censor — Names Only (Text Detect)")
st.caption("Detects name text lines with OpenCV → blurs only names. No avatars. 100% free, local processing.")

# Sidebar: detection parameters
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    st.caption("Tune these if names are missed or wrong text gets blurred.")

    # Position
    min_x = st.slider("Left margin (px)", 20, 150, 60, 5,
                      help="Ignore text left of this X. Names start after avatar space.")
    max_x_ratio = st.slider("Max X ratio", 0.3, 0.9, 0.65, 0.05,
                            help="Ignore text beyond this % of image width.")

    # Size
    min_h = st.slider("Min text height (px)", 8, 30, 15, 1,
                      help="Names must be at least this tall.")
    max_h = st.slider("Max text height (px)", 25, 80, 42, 1,
                      help="Names must be no taller than this.")
    min_w = st.slider("Min text width (px)", 40, 150, 80, 5,
                      help="Names must be at least this wide.")
    max_w = st.slider("Max text width (px)", 200, 600, 380, 10,
                      help="Names must be no wider than this.")

    # Shape
    min_aspect = st.slider("Min aspect ratio (w/h)", 1.0, 5.0, 2.0, 0.1,
                           help="Names are wide & short. Lower = more lenient.")
    max_aspect = st.slider("Max aspect ratio (w/h)", 8.0, 30.0, 18.0, 1.0,
                           help="Upper limit. Prevents catching super-wide UI bars.")

    # Morphology
    kernel_ratio = st.slider("Connect kernel width (% of image)", 0.005, 0.03, 0.012, 0.001,
                             help="Wider = connects letters into longer lines. Narrower = more fragmented.")
    kernel_h = st.slider("Connect kernel height", 1, 8, 3, 1,
                         help="Taller = catches bold/thicker text.")
    dilate_v = st.slider("Vertical dilation", 0, 5, 1,
                         help="Extra vertical expansion after connecting. Helps catch bold names.")

    # Quality
    min_fill = st.slider("Min text fill ratio", 0.02, 0.25, 0.07, 0.01,
                         help="Minimum black-pixel density in the box. Filters out empty regions.")
    cluster_threshold = st.slider("Row cluster threshold (px)", 20, 100, 50, 5,
                                  help="Lines within this vertical distance are grouped as one comment row. Only the topmost is kept as the name.")

    # Censor
    blur_radius = st.slider("Blur radius", 10, 40, 22, 2)
    show_overlay = st.checkbox("Show detection overlay", value=True)

    st.divider()
    st.success("✅ No API. No avatars. Just text detection.")

    # Preset buttons
    st.markdown("**Quick Presets:**")
    pc1, pc2 = st.columns(2)
    if pc1.button("Standard", use_container_width=True):
        st.session_state.update({"min_x": 60, "max_x_ratio": 0.65, "min_h": 15, "max_h": 42,
                                 "min_w": 80, "max_w": 380, "min_aspect": 2.0, "max_aspect": 18.0,
                                 "kernel_ratio": 0.012, "kernel_h": 3, "dilate_v": 1,
                                 "min_fill": 0.07, "cluster_threshold": 50, "blur_radius": 22})
        st.rerun()
    if pc2.button("Tight (small names)", use_container_width=True):
        st.session_state.update({"min_x": 55, "max_x_ratio": 0.60, "min_h": 12, "max_h": 35,
                                 "min_w": 70, "max_w": 320, "min_aspect": 1.8, "max_aspect": 15.0,
                                 "kernel_ratio": 0.010, "kernel_h": 2, "dilate_v": 0,
                                 "min_fill": 0.08, "cluster_threshold": 45, "blur_radius": 20})
        st.rerun()

# Build config dict
cfg = {
    "min_x": min_x,
    "max_x_ratio": max_x_ratio,
    "min_h": min_h,
    "max_h": max_h,
    "min_w": min_w,
    "max_w": max_w,
    "min_aspect": min_aspect,
    "max_aspect": max_aspect,
    "kernel_ratio": kernel_ratio,
    "kernel_h": kernel_h,
    "dilate_v": dilate_v,
    "min_fill": min_fill,
    "cluster_threshold": cluster_threshold,
}

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
        status.info(f"🔍 [{i+1}/{total}] Detecting names in **{uploaded_file.name}**...")

        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            # Detect names
            name_boxes = detect_names(image, cfg)

            # Censor
            censored = apply_censor(image, name_boxes, blur_radius)

            # Export
            buf = io.BytesIO()
            censored.save(buf, format="PNG")
            buf.seek(0)

            item = {
                "filename": uploaded_file.name,
                "original": image,
                "censored": censored,
                "overlay": draw_overlay(image, name_boxes) if show_overlay else None,
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

    # Display
    if main_results or comment_results:
        st.divider()
        st.subheader("📋 Results")

        all_results = main_results + comment_results
        total_names = sum(r["names"] for r in all_results)
        st.success(
            f"✅ Processed {len(main_results)} main + {len(comment_results)} comment images. "
            f"**{total_names}** name box(es) detected & censored."
        )

        for item in all_results:
            with st.container(border=True):
                st.markdown(f"**{item['filename']}** — `{item['names']}` name(s)")

                if item["names"] == 0:
                    st.warning("⚠️ No names detected. Try adjusting parameters in the sidebar (lower Min Height, widen Max Width, etc.)")

                cols = st.columns([1, 1])

                if show_overlay and item["overlay"] is not None:
                    cols[0].image(item["overlay"], caption="Overlay (blue = name)", use_container_width=True)
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
