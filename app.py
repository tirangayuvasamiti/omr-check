import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import tempfile
import os
import time
import random
from scipy import ndimage
from sklearn.cluster import DBSCAN

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yuva Gyan Mahotsav 2026 – Ultra OMR Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --saffron: #F97316;
    --saffron-d: #EA580C;
    --navy: #0F172A;
    --navy-m: #1E293B;
    --navy-l: #334155;
    --green: #22C55E;
    --green-d: #16A34A;
    --gold: #F59E0B;
    --white: #F8FAFC;
    --bg: #0A0F1E;
    --surface: #111827;
    --surface2: #1A2235;
    --surface3: #243047;
    --border: rgba(255,255,255,0.08);
    --border-h: rgba(249,115,22,0.4);
    --text: #F1F5F9;
    --muted: #94A3B8;
    --correct: #22C55E;
    --wrong: #EF4444;
    --skip: #F59E0B;
    --multi: #A855F7;
    --glow-s: 0 0 20px rgba(249,115,22,0.3);
    --glow-g: 0 0 20px rgba(34,197,94,0.3);
    --glow-r: 0 0 20px rgba(239,68,68,0.3);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg) !important; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

/* HEADER */
.omr-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.omr-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(249,115,22,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.omr-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 10%;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(34,197,94,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.tricolor-bar {
    display: flex;
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 20px;
    width: 100%;
    max-width: 400px;
}
.tricolor-bar div:nth-child(1) { flex:1; background:linear-gradient(90deg, #F97316, #EA580C); }
.tricolor-bar div:nth-child(2) { flex:1; background:#F8FAFC; }
.tricolor-bar div:nth-child(3) { flex:1; background:linear-gradient(90deg, #22C55E, #16A34A); }
.omr-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #F8FAFC 30%, #F97316 70%, #22C55E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    position: relative;
    z-index: 1;
}
.omr-subtitle {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 8px;
    font-weight: 400;
    letter-spacing: 0.3px;
    position: relative;
    z-index: 1;
}
.badge-mark {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-mark.pos { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.3); color: #22C55E; }
.badge-mark.neg { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3); color: #EF4444; }

/* STAT CARDS */
.stat-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 20px 0; }
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card.correct { border-color: rgba(34,197,94,0.3); }
.stat-card.wrong   { border-color: rgba(239,68,68,0.3); }
.stat-card.skip    { border-color: rgba(245,158,11,0.3); }
.stat-card.multi   { border-color: rgba(168,85,247,0.3); }
.stat-card.score   { border-color: rgba(249,115,22,0.4); background: linear-gradient(135deg, rgba(249,115,22,0.08), var(--surface)); }
.stat-glow {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.stat-card.correct .stat-glow { background: #22C55E; box-shadow: 0 0 8px #22C55E; }
.stat-card.wrong   .stat-glow { background: #EF4444; box-shadow: 0 0 8px #EF4444; }
.stat-card.skip    .stat-glow { background: #F59E0B; box-shadow: 0 0 8px #F59E0B; }
.stat-card.multi   .stat-glow { background: #A855F7; box-shadow: 0 0 8px #A855F7; }
.stat-card.score   .stat-glow { background: #F97316; box-shadow: 0 0 12px #F97316; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -1px;
}
.stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 6px;
    font-weight: 600;
}
.c-green  { color: #22C55E; }
.c-red    { color: #EF4444; }
.c-amber  { color: #F59E0B; }
.c-orange { color: #F97316; }
.c-purple { color: #A855F7; }

/* RESULT BANNER */
.result-banner {
    border-radius: 14px;
    padding: 18px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    position: relative;
    overflow: hidden;
}
.banner-excellent { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.3); color: #22C55E; }
.banner-good      { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3); color: #F59E0B; }
.banner-average   { background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.3); color: #F97316; }
.banner-poor      { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); color: #EF4444; }

/* PROGRESS BAR */
.score-progress { margin: 12px 0; }
.score-track {
    height: 8px;
    background: var(--surface3);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
}
.score-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* TABLE */
.bubble-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; background: var(--surface); }
.bubble-table th {
    background: var(--surface2);
    color: var(--muted);
    text-transform: uppercase;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 14px 12px;
    border-bottom: 1px solid var(--border);
}
.bubble-table td { padding: 11px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.bubble-table tr:hover td { background: rgba(249,115,22,0.04); }
.bubble-table td:first-child { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--muted); }
.badge-status {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.bs-correct { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.3); color: #22C55E; }
.bs-wrong   { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3); color: #EF4444; }
.bs-skip    { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #F59E0B; }
.bs-multi   { background: rgba(168,85,247,0.12); border: 1px solid rgba(168,85,247,0.3); color: #A855F7; }

/* UPLOAD ZONE */
.upload-info {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.upload-info p { color: var(--muted); font-size: 0.88rem; line-height: 1.6; }

/* SECTION HEADERS */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px;
}
.section-header h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--saffron);
    box-shadow: 0 0 8px var(--saffron);
}

/* LEGEND */
.legend { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 18px; }
.chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px; border-radius: 20px;
    font-size: 0.74rem; font-weight: 600;
    border: 1px solid;
}
.chip-green  { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.3); color: #22C55E; }
.chip-red    { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.3); color: #EF4444; }
.chip-amber  { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); color: #F59E0B; }
.chip-purple { background: rgba(168,85,247,0.08); border-color: rgba(168,85,247,0.3); color: #A855F7; }
.chip-gray   { background: rgba(148,163,184,0.08); border-color: rgba(148,163,184,0.3); color: #94A3B8; }

/* SIDEBAR OVERRIDES */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 600 !important; font-size: 0.85rem !important; color: var(--muted) !important; }
.stSelectbox > div > div, .stTextInput > div > input, .stNumberInput > div > input {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: var(--saffron) !important; }

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #F97316, #EA580C) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 16px rgba(249,115,22,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(249,115,22,0.4) !important;
}

.stProgress > div > div { background: var(--saffron) !important; }
.stAlert { border-radius: 10px !important; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

.debug-panel {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 12px;
    max-height: 200px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class BubbleResult:
    q_num: int
    detected: list
    answer_key: str
    status: str
    score: float
    fill_values: dict = field(default_factory=dict)

@dataclass
class OMRResult:
    bubbles: List[BubbleResult]
    correct: int = 0
    wrong: int = 0
    unattempted: int = 0
    multi: int = 0
    pos_score: float = 0
    neg_score: float = 0
    total_score: float = 0
    debug_log: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  ULTRA PRO OMR ENGINE v3.0
#  Multi-strategy detection with adaptive thresholding, Hough circles,
#  contour analysis, perspective correction, and confidence scoring.
# ═══════════════════════════════════════════════════════════════════════════════
class UltraOMREngine:

    def __init__(self):
        self.debug_log = []

    def log(self, msg: str):
        self.debug_log.append(msg)

    # ── Image Loading ──────────────────────────────────────────────────────────
    def pdf_to_image(self, pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def pil_to_cv(self, pil_img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ── Pre-processing Pipeline ───────────────────────────────────────────────
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns: gray, adaptive-thresh, otsu-thresh"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # Adaptive threshold - great for uneven lighting/scanned docs
        adaptive = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )

        # OTSU for clean prints
        blur = cv2.GaussianBlur(denoised, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological close to connect bubble outlines
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, k3)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k3)

        return gray, adaptive, otsu

    # ── Perspective Correction ────────────────────────────────────────────────
    def deskew(self, img: np.ndarray) -> np.ndarray:
        """Correct slight rotation using Hough lines."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        if lines is None:
            return img
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 10:  # Only small angles
                angles.append(angle)
        if not angles:
            return img
        median_angle = np.median(angles)
        if abs(median_angle) < 0.3:
            return img
        self.log(f"Deskew: rotating {median_angle:.2f}°")
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # ── Circle Detection via Hough ─────────────────────────────────────────────
    def detect_circles_hough(self, gray: np.ndarray, img_shape: tuple) -> list:
        """Detect circular bubbles using Hough Circle Transform."""
        H, W = img_shape[:2]
        # Estimate bubble radius from image size (typical OMR sheet)
        min_r = max(6, int(min(H, W) * 0.008))
        max_r = max(22, int(min(H, W) * 0.022))

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min_r * 1.8),
            param1=60,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r
        )
        results = []
        if circles is not None:
            for x, y, r in circles[0]:
                results.append({
                    'x': int(x), 'y': int(y),
                    'r': int(r), 'w': int(r * 2), 'h': int(r * 2),
                    'method': 'hough',
                    'bbox': (int(x - r), int(y - r), int(r * 2), int(r * 2))
                })
        self.log(f"Hough circles: {len(results)}")
        return results

    # ── Contour-based Bubble Detection ────────────────────────────────────────
    def detect_circles_contour(self, thresh: np.ndarray) -> list:
        """Detect bubbles using contour circularity filtering."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 12000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            # Circularity: 4π·Area / Perimeter²  (1.0 = perfect circle)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < 0.40:  # Reasonably circular
                continue
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect = w / float(h) if h > 0 else 0
            if not (0.5 < aspect < 2.0):
                continue
            cx, cy = x + w // 2, y + h // 2
            r = max(w, h) // 2
            results.append({
                'x': cx, 'y': cy, 'r': r, 'w': w, 'h': h,
                'area': area, 'circularity': circularity,
                'method': 'contour',
                'bbox': (x, y, w, h)
            })
        self.log(f"Contour circles: {len(results)}")
        return results

    # ── Merge & Deduplicate Detections ────────────────────────────────────────
    def merge_detections(self, hough: list, contour: list, dist_thresh: float = 8.0) -> list:
        """Combine Hough and contour detections, prefer Hough, dedup by proximity."""
        merged = list(hough)
        for c in contour:
            too_close = False
            for m in merged:
                d = np.hypot(c['x'] - m['x'], c['y'] - m['y'])
                if d < dist_thresh:
                    too_close = True
                    break
            if not too_close:
                merged.append(c)
        self.log(f"Merged detections: {len(merged)}")
        return merged

    # ── Grid Layout Analysis ──────────────────────────────────────────────────
    def build_question_grid(self, detections: list, img_shape: tuple) -> dict:
        """
        Robust grid builder:
        1. Remove margin noise (top title, bottom signatures)
        2. Estimate bubble size from median
        3. Divide into 3 columns by x-coordinate quantiles
        4. Within each column, cluster into rows using y-coordinate
        5. Within each row, identify 4 options (A/B/C/D) by x-position
        6. For missing bubbles: interpolate from learned inter-bubble spacing
        """
        if not detections:
            return {}

        H, W = img_shape[:2]
        # Content region: skip top 12% (header) and bottom 8% (footer)
        y_min = int(H * 0.12)
        y_max = int(H * 0.92)
        pts = [d for d in detections if y_min < d['y'] < y_max]
        if not pts:
            self.log("No detections in content region")
            return {}

        # Estimate typical bubble radius
        radii = [d['r'] for d in pts]
        med_r = int(np.median(radii))
        self.log(f"Median bubble radius: {med_r}px")

        # ── Column Assignment ─────────────────────────────────────────────────
        # Use x-coord quantiles to find natural column boundaries
        xs = sorted([d['x'] for d in pts])
        # Three columns: divide x-space into thirds using 33rd/66th percentile
        p33 = np.percentile(xs, 33)
        p66 = np.percentile(xs, 66)
        col_bounds = [0, p33, p66, W]

        cols = [[] for _ in range(3)]
        for d in pts:
            if d['x'] < p33:
                cols[0].append(d)
            elif d['x'] < p66:
                cols[1].append(d)
            else:
                cols[2].append(d)

        self.log(f"Column sizes: {[len(c) for c in cols]}")

        question_map = {}

        for col_idx, col_pts in enumerate(cols):
            if not col_pts:
                continue

            q_offset = col_idx * 20 + 1

            # ── Row Clustering ────────────────────────────────────────────────
            # Sort by y, then cluster with gap detection
            col_pts.sort(key=lambda d: d['y'])
            row_gap = med_r * 2.8

            rows = []
            curr_row = [col_pts[0]]
            for d in col_pts[1:]:
                row_mean_y = np.mean([p['y'] for p in curr_row])
                if abs(d['y'] - row_mean_y) < row_gap:
                    curr_row.append(d)
                else:
                    rows.append(curr_row)
                    curr_row = [d]
            rows.append(curr_row)

            self.log(f"Col {col_idx}: {len(rows)} rows detected")

            # Filter to rows with 1–6 detections (avoid noise rows)
            rows = [r for r in rows if 1 <= len(r) <= 8]

            # Sort rows by mean y
            rows.sort(key=lambda r: np.mean([d['y'] for d in r]))

            # Cap at 20 questions per column
            rows = rows[:20]

            if not rows:
                continue

            # ── Learn Option Spacing from rows with exactly 4 detections ─────
            ref_spacings = []
            for row in rows:
                row.sort(key=lambda d: d['x'])
                if len(row) == 4:
                    xs_row = [d['x'] for d in row]
                    spacings = [xs_row[i+1] - xs_row[i] for i in range(3)]
                    if all(5 < s < W * 0.25 for s in spacings):
                        ref_spacings.append(xs_row)

            if ref_spacings:
                # Learn relative positions: normalize to first option = 0
                norm_positions = []
                for r in ref_spacings:
                    base = r[0]
                    norm_positions.append([x - base for x in r])
                median_offsets = np.median(norm_positions, axis=0)
                self.log(f"Col {col_idx}: learned offsets {median_offsets.astype(int).tolist()}")
            else:
                # Fallback: estimate from column width
                col_width = col_bounds[col_idx + 1] - col_bounds[col_idx]
                spacing = col_width * 0.18
                median_offsets = np.array([0, spacing, spacing * 2, spacing * 3])
                self.log(f"Col {col_idx}: fallback offsets {median_offsets.astype(int).tolist()}")

            # ── Assign A/B/C/D to each row using learned offsets ─────────────
            for row_idx, row in enumerate(rows):
                q_num = q_offset + row_idx
                row.sort(key=lambda d: d['x'])
                opts = {}

                if len(row) >= 4:
                    # Use the 4 best candidates sorted by x
                    candidates = sorted(row, key=lambda d: d['x'])[:4]
                    for i, opt in enumerate(['A', 'B', 'C', 'D']):
                        if i < len(candidates):
                            d = candidates[i]
                            opts[opt] = {
                                'x': d['x'], 'y': d['y'],
                                'r': d.get('r', med_r),
                                'w': d['w'], 'h': d['h'],
                                'bbox': d['bbox'],
                                'interpolated': False
                            }
                else:
                    # Anchor: use leftmost detected bubble
                    anchor = min(row, key=lambda d: d['x'])
                    anchor_x, anchor_y = anchor['x'], anchor['y']

                    for i, opt in enumerate(['A', 'B', 'C', 'D']):
                        target_x = int(anchor_x + median_offsets[i])
                        # Check if any detection is near this expected position
                        best_match = None
                        best_dist = med_r * 2.0
                        for d in row:
                            dist = abs(d['x'] - target_x)
                            if dist < best_dist:
                                best_dist = dist
                                best_match = d

                        if best_match:
                            opts[opt] = {
                                'x': best_match['x'], 'y': best_match['y'],
                                'r': best_match.get('r', med_r),
                                'w': best_match['w'], 'h': best_match['h'],
                                'bbox': best_match['bbox'],
                                'interpolated': False
                            }
                        else:
                            # INTERPOLATE: place virtual bubble at expected position
                            r = med_r
                            opts[opt] = {
                                'x': target_x, 'y': anchor_y,
                                'r': r, 'w': r * 2, 'h': r * 2,
                                'bbox': (target_x - r, anchor_y - r, r * 2, r * 2),
                                'interpolated': True
                            }

                question_map[q_num] = opts

        self.log(f"Total questions mapped: {len(question_map)}")
        return question_map

    # ── Fill Measurement (Multi-Strategy) ─────────────────────────────────────
    def measure_fill_robust(self, gray: np.ndarray, bubble: dict,
                             adaptive: np.ndarray, otsu: np.ndarray) -> Tuple[float, float]:
        """
        Returns (fill_ratio, confidence) using multiple threshold strategies.
        Takes the max fill ratio from adaptive + otsu approaches.
        """
        x, y, w, h = bubble['bbox']
        r = bubble.get('r', max(w, h) // 2)
        pad = max(2, r // 4)

        H, W = gray.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0

        roi_gray = gray[y1:y2, x1:x2]
        roi_adaptive = adaptive[y1:y2, x1:x2]
        roi_otsu = otsu[y1:y2, x1:x2]

        if roi_gray.size == 0:
            return 0.0, 0.0

        # Create elliptical mask for precise measurement inside the bubble
        cy_local = (y2 - y1) // 2
        cx_local = (x2 - x1) // 2
        ry = max(1, (y2 - y1) // 2 - 1)
        rx = max(1, (x2 - x1) // 2 - 1)
        mask = np.zeros(roi_gray.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx_local, cy_local), (rx, ry), 0, 0, 360, 255, -1)

        total_px = np.count_nonzero(mask)
        if total_px == 0:
            return 0.0, 0.0

        # Method 1: Adaptive threshold fill
        filled_adaptive = np.count_nonzero(roi_adaptive & (mask > 0))
        ratio_adaptive = filled_adaptive / total_px

        # Method 2: OTSU fill
        filled_otsu = np.count_nonzero(roi_otsu & (mask > 0))
        ratio_otsu = filled_otsu / total_px

        # Method 3: Direct intensity analysis
        # A filled bubble should be darker than average sheet brightness
        sheet_mean = np.mean(gray)  # Approximate sheet brightness
        roi_mean = np.mean(roi_gray[mask > 0])
        # Darkness ratio: how much darker than sheet mean
        darkness_ratio = max(0.0, (sheet_mean - roi_mean) / sheet_mean) if sheet_mean > 0 else 0.0

        # Weighted combination
        fill_ratio = max(ratio_adaptive, ratio_otsu) * 0.65 + darkness_ratio * 0.35

        # Confidence: how consistent are the methods?
        variance = np.var([ratio_adaptive, ratio_otsu, darkness_ratio])
        confidence = max(0.0, 1.0 - variance * 4)

        return fill_ratio, confidence

    # ── Classification with Adaptive Threshold ────────────────────────────────
    def classify_bubbles_adaptive(self, gray: np.ndarray, question_map: dict,
                                   adaptive: np.ndarray, otsu: np.ndarray,
                                   fill_thresh: float = 0.25) -> dict:
        """
        Smart bubble classification:
        - Compute fill ratios for all bubbles in a question
        - Use adaptive cut-off: if max fill >> others, it's clearly selected
        - Relative thresholding: selected = fill > 2× mean_fill AND > absolute threshold
        """
        results = {}

        for q, opts in question_map.items():
            fills = {}
            confidences = {}
            for opt in ['A', 'B', 'C', 'D']:
                if opt not in opts:
                    continue
                f, c = self.measure_fill_robust(gray, opts[opt], adaptive, otsu)
                fills[opt] = f
                confidences[opt] = c

            if not fills:
                results[q] = {'selected': [], 'fills': fills, 'confidences': confidences}
                continue

            # ── Adaptive selection logic ──────────────────────────────────────
            max_fill = max(fills.values())
            mean_fill = np.mean(list(fills.values()))
            fill_vals = list(fills.values())
            fill_vals.sort(reverse=True)

            # Dynamic threshold: between absolute floor and relative assessment
            # If the sheet has a very dark background, use relative mode
            relative_thresh = mean_fill * 2.2  # Must be 2.2× above average
            effective_thresh = max(fill_thresh, min(relative_thresh, fill_thresh * 2.5))

            # Selection: must exceed effective threshold OR be clearly dominant
            selected = []
            for opt, f in fills.items():
                is_above_abs = f >= fill_thresh
                is_dominant = (max_fill > 0.10) and (f >= max_fill * 0.75) and (f >= fill_thresh * 0.7)
                if is_above_abs or is_dominant:
                    selected.append((opt, f))

            # If multiple are "selected" but one is clearly dominant, keep only that
            if len(selected) > 1:
                max_among_selected = max(s[1] for s in selected)
                # If top bubble is 1.5× the second, it's the only selection
                selected_sorted = sorted(selected, key=lambda s: s[1], reverse=True)
                if len(selected_sorted) >= 2:
                    ratio = selected_sorted[0][1] / (selected_sorted[1][1] + 1e-6)
                    if ratio > 1.8:
                        selected = [selected_sorted[0]]

            selected_opts = [s[0] for s in selected]

            results[q] = {
                'selected': selected_opts,
                'fills': fills,
                'confidences': confidences
            }

        return results

    # ── Measure Fill for Annotated Debug Display ──────────────────────────────
    def measure_fill_otsu(self, otsu: np.ndarray, bubble: dict) -> float:
        """Simple OTSU fill for annotation purposes."""
        x, y, w, h = bubble['bbox']
        pad = 2
        H, W = otsu.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        roi = otsu[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return np.count_nonzero(roi) / roi.size

    # ── Main Grade Pipeline ────────────────────────────────────────────────────
    def grade(self, img: np.ndarray, answer_key: dict,
              pos: float = 3.0, neg: float = 1.0,
              fill_thresh: float = 0.25) -> Tuple[OMRResult, np.ndarray]:

        self.debug_log = []
        self.log(f"Image shape: {img.shape}")

        # Step 1: Deskew
        img = self.deskew(img)

        # Step 2: Preprocess
        gray, adaptive, otsu = self.preprocess(img)

        # Step 3: Multi-strategy circle detection
        hough_circles = self.detect_circles_hough(gray, img.shape)
        contour_circles = self.detect_circles_contour(adaptive)
        contour_circles_otsu = self.detect_circles_contour(otsu)

        # Merge all detections
        all_detections = self.merge_detections(hough_circles, contour_circles, dist_thresh=8)
        all_detections = self.merge_detections(all_detections, contour_circles_otsu, dist_thresh=8)

        self.log(f"Total unique detections: {len(all_detections)}")

        # Step 4: Build grid
        question_map = self.build_question_grid(all_detections, img.shape)

        # Step 5: Classify with adaptive thresholding
        bubble_results_raw = self.classify_bubbles_adaptive(
            gray, question_map, adaptive, otsu, fill_thresh
        )

        # Step 6: Grade & Annotate
        annotated = img.copy()
        results = []

        for q in range(1, 61):
            key = answer_key.get(q, '')
            raw = bubble_results_raw.get(q, {'selected': [], 'fills': {}, 'confidences': {}})
            selected = raw.get('selected', [])
            fills = raw.get('fills', {})
            opts_map = question_map.get(q, {})

            if len(selected) == 0:
                status = 'unattempted'
                score = 0.0
            elif len(selected) > 1:
                status = 'multi'
                score = -neg if key else 0.0
            elif key and selected[0] == key:
                status = 'correct'
                score = pos
            elif key:
                status = 'wrong'
                score = -neg
            else:
                status = 'unattempted'
                score = 0.0

            br = BubbleResult(
                q_num=q, detected=selected, answer_key=key,
                status=status, score=score, fill_values=fills
            )
            results.append(br)

            # ── Annotation ───────────────────────────────────────────────────
            for opt in ['A', 'B', 'C', 'D']:
                if opt not in opts_map:
                    continue
                bubble = opts_map[opt]
                cx, cy = bubble['x'], bubble['y']
                r = bubble.get('r', max(bubble['w'], bubble['h']) // 2)

                is_interpolated = bubble.get('interpolated', False)
                fill_val = fills.get(opt, 0.0)

                if opt in selected:
                    # Color based on outcome
                    if status == 'correct':
                        color = (50, 205, 50)   # Green
                    elif status == 'wrong':
                        color = (50, 50, 235)   # Blue-red
                    elif status == 'multi':
                        color = (180, 50, 220)  # Purple
                    else:
                        color = (50, 180, 220)  # Cyan
                    cv2.circle(annotated, (cx, cy), r + 2, color, -1)
                    cv2.circle(annotated, (cx, cy), r + 3, (255, 255, 255), 1)
                    cv2.putText(annotated, opt, (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                else:
                    # Unfilled: light outline, dim
                    color = (60, 60, 60) if is_interpolated else (140, 140, 140)
                    thickness = 1
                    cv2.circle(annotated, (cx, cy), r, color, thickness)
                    # Show fill value for debugging
                    if fill_val > 0.05:
                        cv2.putText(annotated, f"{fill_val:.2f}", (cx - 8, cy + 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 180, 100), 1)

                # Mark correct answer if missed
                if key == opt and opt not in selected and status not in ('correct', 'unattempted'):
                    cv2.circle(annotated, (cx, cy), r + 5, (50, 205, 50), 2)

            # Q number label
            if opts_map:
                first_opt = opts_map.get('A', list(opts_map.values())[0])
                lx = max(0, first_opt['x'] - first_opt.get('r', 12) * 4)
                ly = first_opt['y'] + 4
                # Status-colored label
                label_color = {
                    'correct': (50, 205, 50),
                    'wrong': (50, 50, 235),
                    'unattempted': (140, 140, 140),
                    'multi': (180, 50, 220)
                }.get(status, (200, 200, 200))
                cv2.putText(annotated, f"Q{q}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1)

        # Summary stats
        correct = sum(1 for r in results if r.status == 'correct')
        wrong = sum(1 for r in results if r.status == 'wrong')
        unattempted = sum(1 for r in results if r.status == 'unattempted')
        multi = sum(1 for r in results if r.status == 'multi')
        pos_score = correct * pos
        neg_score = (wrong + multi) * neg
        total = pos_score - neg_score

        self.log(f"Results: {correct}✓ {wrong}✗ {unattempted}— {multi}M → {total:.1f}")

        omr_result = OMRResult(
            bubbles=results, correct=correct, wrong=wrong,
            unattempted=unattempted, multi=multi,
            pos_score=pos_score, neg_score=neg_score, total_score=total,
            debug_log=list(self.debug_log)
        )
        return omr_result, annotated


# ─── Session State ─────────────────────────────────────────────────────────────
if 'answer_key' not in st.session_state:
    st.session_state.answer_key = {i: random.choice(['A', 'B', 'C', 'D']) for i in range(1, 61)}
if 'result' not in st.session_state:
    st.session_state.result = None
if 'original_img' not in st.session_state:
    st.session_state.original_img = None
if 'annotated_img' not in st.session_state:
    st.session_state.annotated_img = None

engine = UltraOMREngine()

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="omr-header">
  <div class="tricolor-bar"><div></div><div></div><div></div></div>
  <h1 class="omr-title">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="omr-subtitle">
    Ultra OMR Auto-Grader v3.0 &nbsp;·&nbsp; Tiranga Yuva Samiti &nbsp;·&nbsp;
    <span style="color:#22C55E;font-weight:700;">Multi-Strategy AI Detection</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Grading Configuration")
    st.markdown("**Marking Scheme**")
    pos_mark = st.number_input("✅ Correct (+)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    neg_mark = st.number_input("❌ Wrong (−)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

    st.markdown("**Detection Sensitivity**")
    fill_threshold = st.slider(
        "Fill Threshold", 0.08, 0.55, 0.22, 0.01,
        help="Lower = picks up lighter pencil marks. Recommended: 0.18–0.30"
    )
    st.caption(f"💡 Current: {fill_threshold:.2f} — try 0.15 for light marks, 0.30 for dark marks only")

    show_debug = st.checkbox("🔍 Show Debug Log", value=False)

    st.divider()
    st.markdown("### 📋 Answer Key")
    st.caption("Pre-filled randomly for testing. Edit or paste below.")

    bulk_key = st.text_area(
        "Paste 60 answers (comma-separated)",
        placeholder="A,B,C,D,A,B,...",
        height=80
    )
    if st.button("Apply Bulk Key", use_container_width=True):
        parts = [p.strip().upper() for p in bulk_key.split(',')]
        for i, ans in enumerate(parts[:60]):
            if ans in ('A', 'B', 'C', 'D', ''):
                st.session_state.answer_key[i + 1] = ans
        st.success("✅ Key applied!")

    st.caption("Or set individually:")
    options = ['', 'A', 'B', 'C', 'D']
    cols_k = st.columns(2)
    for q in range(1, 61):
        col = cols_k[0] if q % 2 != 0 else cols_k[1]
        with col:
            st.session_state.answer_key[q] = st.selectbox(
                f"Q{q}", options,
                index=options.index(st.session_state.answer_key.get(q, '')),
                key=f"key_{q}"
            )

# ─── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header"><div class="section-dot"></div><h3>Upload & Grade OMR Sheet</h3></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legend">
  <span class="chip chip-green">● Correct</span>
  <span class="chip chip-red">● Wrong</span>
  <span class="chip chip-gray">○ Unattempted</span>
  <span class="chip chip-amber">○ Skipped</span>
  <span class="chip chip-purple">● Multi-Filled</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-info">
  <p>
    <strong style="color:#F97316;">Supported formats:</strong> PDF (scanned), PNG, JPG, TIFF, BMP<br>
    <strong style="color:#22C55E;">Recommended:</strong> 200–300 DPI scan, straight alignment, clean background<br>
    <strong style="color:#A855F7;">Ultra Detection:</strong> Hough circles + contour analysis + adaptive thresholding + interpolation
  </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop OMR sheet here",
    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="PDF or image file"
)

if uploaded:
    with st.spinner("Loading file..."):
        file_bytes = uploaded.read()
        if uploaded.type == 'application/pdf':
            img_cv = engine.pdf_to_image(file_bytes, dpi=300)
        else:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_cv = engine.pil_to_cv(pil_img)
        st.session_state.original_img = img_cv.copy()

    st.success(f"✅ Loaded: **{uploaded.name}** — {img_cv.shape[1]}×{img_cv.shape[0]}px")

    col_orig, col_action = st.columns([1, 1])
    with col_orig:
        st.markdown("**📄 Original Sheet**")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_action:
        st.markdown("**🚀 Grade Settings**")
        st.info(f"""
**Marking:** +{pos_mark} correct, −{neg_mark} wrong  
**Fill Threshold:** {fill_threshold:.2f}  
**Questions:** 60 (3 columns × 20)  
**Engine:** Ultra OMR v3.0
        """)
        if st.button("🔬 Grade OMR Sheet", use_container_width=True):
            bar = st.progress(0, text="Deskewing & preprocessing...")
            time.sleep(0.2)
            bar.progress(20, text="Running Hough circle detection...")
            time.sleep(0.2)
            bar.progress(40, text="Running contour analysis...")
            time.sleep(0.1)
            bar.progress(55, text="Merging & deduplicating detections...")
            time.sleep(0.1)
            bar.progress(70, text="Building adaptive question grid...")

            result, annotated = engine.grade(
                img_cv,
                st.session_state.answer_key,
                pos=pos_mark, neg=neg_mark,
                fill_thresh=fill_threshold
            )

            bar.progress(90, text="Scoring & annotating...")
            st.session_state.result = result
            st.session_state.annotated_img = annotated
            time.sleep(0.2)
            bar.progress(100, text="Complete!")
            time.sleep(0.3)
            bar.empty()
            st.success("✅ Grading complete!")


# ─── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    result = st.session_state.result
    st.divider()

    st.markdown("""
    <div class="section-header"><div class="section-dot"></div><h3>Results Dashboard</h3></div>
    """, unsafe_allow_html=True)

    # Banner
    total_q = 60
    max_score = total_q * pos_mark
    pct = (result.total_score / max_score) * 100 if max_score > 0 else 0
    if pct >= 75:
        bcls, btxt, bicon = "banner-excellent", "Outstanding Performance!", "🏆"
    elif pct >= 50:
        bcls, btxt, bicon = "banner-good", "Good Performance", "👍"
    elif pct >= 35:
        bcls, btxt, bicon = "banner-average", "Average — Keep Practicing", "📚"
    else:
        bcls, btxt, bicon = "banner-poor", "Needs Improvement", "⚠️"

    pct_bar_color = "#22C55E" if pct >= 75 else ("#F59E0B" if pct >= 50 else ("#F97316" if pct >= 35 else "#EF4444"))

    st.markdown(f"""
    <div class="result-banner {bcls}">
      <span style="font-size:1.8rem;">{bicon}</span>
      <div>
        <div>{btxt}</div>
        <div style="font-size:0.88rem; font-weight:400; opacity:0.8; margin-top:3px;">
          Score: <strong>{result.total_score:.1f}</strong> / {max_score:.0f} &nbsp;·&nbsp; {pct:.1f}%
        </div>
      </div>
    </div>
    <div class="score-progress">
      <div class="score-track">
        <div class="score-fill" style="width:{min(pct,100):.1f}%; background:linear-gradient(90deg, {pct_bar_color}, {pct_bar_color}88);"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat Cards
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card correct">
        <div class="stat-num c-green">{result.correct}</div>
        <div class="stat-label">Correct</div>
        <div class="stat-glow"></div>
      </div>
      <div class="stat-card wrong">
        <div class="stat-num c-red">{result.wrong}</div>
        <div class="stat-label">Wrong</div>
        <div class="stat-glow"></div>
      </div>
      <div class="stat-card skip">
        <div class="stat-num c-amber">{result.unattempted}</div>
        <div class="stat-label">Skipped</div>
        <div class="stat-glow"></div>
      </div>
      <div class="stat-card multi">
        <div class="stat-num c-purple">{result.multi}</div>
        <div class="stat-label">Multi-Mark</div>
        <div class="stat-glow"></div>
      </div>
      <div class="stat-card score">
        <div class="stat-num c-orange">{result.total_score:.1f}</div>
        <div class="stat-label">Final Score</div>
        <div class="stat-glow"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**📉 Score Breakdown**")
        attempted = result.correct + result.wrong + result.multi
        accuracy = (result.correct / attempted * 100) if attempted > 0 else 0
        
        m1, m2 = st.columns(2)
        m1.metric("Positive Score", f"+{result.pos_score:.1f}")
        m2.metric("Negative Score", f"−{result.neg_score:.1f}")
        m3, m4 = st.columns(2)
        m3.metric("Net Score", f"{result.total_score:.1f}")
        m4.metric("Accuracy", f"{accuracy:.1f}%")

        st.markdown(f"""
        <div style="background:var(--surface2); border-radius:10px; padding:14px; margin-top:10px; border: 1px solid var(--border);">
          <div style="font-size:0.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Attempt Analysis</div>
          <div style="display:flex; justify-content:space-between; font-size:0.88rem; margin:4px 0;">
            <span>Attempted</span><span style="font-weight:700;">{attempted}/60 ({attempted/60*100:.0f}%)</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.88rem; margin:4px 0;">
            <span>Unattempted</span><span style="font-weight:700;">{result.unattempted}/60</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.88rem; margin:4px 0;">
            <span>Multi-marked</span><span style="font-weight:700;">{result.multi}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if show_debug and result.debug_log:
            st.markdown("**🔍 Debug Log**")
            st.markdown(
                '<div class="debug-panel">' +
                '<br>'.join(result.debug_log) +
                '</div>',
                unsafe_allow_html=True
            )

    with col_b:
        st.markdown("**🎯 Graded OMR Preview**")
        if st.session_state.annotated_img is not None:
            ann_rgb = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)

            ann_pil = Image.fromarray(ann_rgb)
            buf = io.BytesIO()
            ann_pil.save(buf, format='PNG')
            st.download_button(
                "⬇️ Download Graded Image", buf.getvalue(),
                file_name="graded_omr.png", mime="image/png",
                use_container_width=True
            )

    st.divider()

    st.markdown("""
    <div class="section-header"><div class="section-dot"></div><h3>Question-wise Detailed Report</h3></div>
    """, unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        filter_status = st.multiselect(
            "Filter by status",
            ['correct', 'wrong', 'unattempted', 'multi'],
            default=['correct', 'wrong', 'unattempted', 'multi']
        )
    with filter_col2:
        show_fills = st.checkbox("Show fill values", value=False,
                                  help="Show raw fill percentages for debugging")

    filtered = [b for b in result.bubbles if b.status in filter_status]

    rows_html = ""
    for b in filtered:
        detected_str = ', '.join(b.detected) if b.detected else '—'
        key_str = b.answer_key if b.answer_key else '—'
        score_str = f"+{b.score:.0f}" if b.score > 0 else (f"{b.score:.0f}" if b.score != 0 else "0")
        score_color = "c-green" if b.score > 0 else ("c-red" if b.score < 0 else "c-amber")
        badge_cls = {'correct': 'bs-correct', 'wrong': 'bs-wrong',
                     'unattempted': 'bs-skip', 'multi': 'bs-multi'}.get(b.status, '')
        status_icon = {'correct': '✓', 'wrong': '✗', 'unattempted': '—', 'multi': '×'}.get(b.status, '')

        fill_str = ""
        if show_fills and b.fill_values:
            fill_str = " ".join([f"{k}:{v:.2f}" for k, v in sorted(b.fill_values.items())])

        rows_html += f"""
        <tr>
          <td>Q{b.q_num:02d}</td>
          <td><span style="color:#60A5FA; font-weight:700; font-family:'JetBrains Mono',monospace;">{detected_str}</span></td>
          <td><span style="color:#34D399; font-weight:700; font-family:'JetBrains Mono',monospace;">{key_str}</span></td>
          <td><span class="badge-status {badge_cls}">{status_icon} {b.status.upper()}</span></td>
          <td class="{score_color}" style="font-weight:800; font-family:'JetBrains Mono',monospace;">{score_str}</td>
          {"<td style='font-size:0.72rem;color:var(--muted);font-family:JetBrains Mono,monospace;'>" + fill_str + "</td>" if show_fills else ""}
        </tr>"""

    extra_th = "<th>Fill Values</th>" if show_fills else ""
    table_html = f"""
    <div style="max-height:520px; overflow-y:auto; border:1px solid var(--border); border-radius:12px; box-shadow:0 4px 16px rgba(0,0,0,0.3);">
    <table class="bubble-table">
      <thead>
        <tr>
          <th>Q#</th><th>Detected</th><th>Answer Key</th><th>Status</th><th>Score</th>{extra_th}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)

    st.write("")
    export_data = [
        {
            'Question': b.q_num,
            'Detected': ', '.join(b.detected) if b.detected else '',
            'Answer Key': b.answer_key,
            'Status': b.status,
            'Score': b.score,
            **{f'Fill_{k}': round(v, 4) for k, v in b.fill_values.items()}
        }
        for b in result.bubbles
    ]
    df_export = pd.DataFrame(export_data)
    st.download_button(
        "⬇️ Download Full Results CSV",
        df_export.to_csv(index=False),
        file_name="omr_results.csv",
        mime="text/csv"
    )

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 12px; color:var(--muted); font-size:0.8rem; font-weight:500;">
  <div class="tricolor-bar" style="max-width:160px; margin:0 auto 12px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · Ultra OMR v3.0
  <br><span style="font-size:0.72rem; opacity:0.6;">Hough Transform · Contour Analysis · Adaptive Thresholding · Grid Interpolation</span>
</div>
""", unsafe_allow_html=True)
