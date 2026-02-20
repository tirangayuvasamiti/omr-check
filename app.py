"""
YUVA GYAN MAHOTSAV 2026 — PRECISION OMR GRADER v5.0
=====================================================
Auto-Crop + Perspective Warp + Hardcoded Exact Grid

VERIFIED BUBBLE POSITIONS (from blank sheet analysis at 300 DPI):
  Col 1 (Q01-Q20): A=380, B=502, C=620, D=744
  Col 2 (Q21-Q40): A=1114, B=1236, C=1354, D=1476
  Col 3 (Q41-Q60): A=1848, B=1966, C=2088, D=2210
  Row 1 Y = 720,  Row spacing = 109 px,  Bubble radius = 23 px

AUTO-CROP PIPELINE:
  1. Convert to grayscale → binary threshold
  2. Find 4 corner anchor squares (■) of the answer grid
  3. Compute homography → warp to canonical 2479×3508 frame
  4. Sample bubbles at exact canonical coordinates
  5. Adaptive + darkness + dark-pixel fill scoring
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pdf2image import convert_from_bytes
import time
import random

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YGM 2026 OMR Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --saffron:#F97316;--saffron-d:#EA580C;--navy:#0F172A;--navy-m:#1E293B;
    --green:#22C55E;--green-d:#16A34A;--gold:#F59E0B;
    --bg:#080D1A;--surface:#0F1923;--surface2:#162032;--surface3:#1D2D45;
    --border:rgba(255,255,255,0.07);--text:#E2E8F0;--muted:#64748B;
    --correct:#22C55E;--wrong:#EF4444;--skip:#F59E0B;--multi:#A855F7;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:var(--bg);color:var(--text);}
.stApp{background:var(--bg)!important;}
.main .block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1440px;}

.omr-header{background:linear-gradient(135deg,#0A1628,#0F1F3D 50%,#0A1628);
    border:1px solid rgba(249,115,22,.2);border-radius:18px;padding:28px 36px 24px;
    margin-bottom:24px;position:relative;overflow:hidden;}
.omr-header::before{content:'';position:absolute;top:-80px;right:-80px;width:260px;height:260px;
    background:radial-gradient(circle,rgba(249,115,22,.12),transparent 65%);border-radius:50%;}
.omr-header::after{content:'';position:absolute;bottom:-50px;left:8%;width:180px;height:180px;
    background:radial-gradient(circle,rgba(34,197,94,.08),transparent 65%);border-radius:50%;}
.tricolor{display:flex;height:4px;border-radius:2px;overflow:hidden;margin-bottom:16px;max-width:360px;}
.tricolor div:nth-child(1){flex:1;background:linear-gradient(90deg,#F97316,#EA580C);}
.tricolor div:nth-child(2){flex:1;background:#E2E8F0;}
.tricolor div:nth-child(3){flex:1;background:linear-gradient(90deg,#22C55E,#16A34A);}
.omr-title{font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;
    background:linear-gradient(130deg,#F1F5F9 20%,#F97316 60%,#22C55E 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1.1;position:relative;z-index:1;}
.omr-sub{font-size:.87rem;color:var(--muted);margin-top:7px;z-index:1;position:relative;}
.tech-pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;
    font-size:.73rem;font-weight:700;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);
    color:#22C55E;font-family:'JetBrains Mono',monospace;margin-top:8px;}
.tech-pill-o{background:rgba(249,115,22,.1);border-color:rgba(249,115,22,.25);color:#F97316;}

.stat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:18px 0;}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
    padding:18px 14px;text-align:center;position:relative;overflow:hidden;transition:transform .2s;}
.stat-card:hover{transform:translateY(-2px);}
.stat-card.c{border-color:rgba(34,197,94,.3);}.stat-card.w{border-color:rgba(239,68,68,.3);}
.stat-card.s{border-color:rgba(245,158,11,.3);}.stat-card.m{border-color:rgba(168,85,247,.3);}
.stat-card.sc{border-color:rgba(249,115,22,.4);background:linear-gradient(135deg,rgba(249,115,22,.06),var(--surface));}
.glow{position:absolute;bottom:0;left:0;right:0;height:2px;}
.stat-card.c .glow{background:#22C55E;box-shadow:0 0 8px #22C55E;}
.stat-card.w .glow{background:#EF4444;box-shadow:0 0 8px #EF4444;}
.stat-card.s .glow{background:#F59E0B;box-shadow:0 0 8px #F59E0B;}
.stat-card.m .glow{background:#A855F7;box-shadow:0 0 8px #A855F7;}
.stat-card.sc .glow{background:#F97316;box-shadow:0 0 12px #F97316;}
.sn{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;line-height:1;letter-spacing:-1px;}
.sl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-top:5px;font-weight:600;}
.cg{color:#22C55E;}.cr{color:#EF4444;}.ca{color:#F59E0B;}.co{color:#F97316;}.cp{color:#A855F7;}

.rbanner{border-radius:14px;padding:18px 22px;margin:14px 0;display:flex;align-items:center;
    gap:14px;font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;}
.rb-ex{background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.3);color:#22C55E;}
.rb-gd{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.3);color:#F59E0B;}
.rb-av{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.3);color:#F97316;}
.rb-pr{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.3);color:#EF4444;}
.score-track{height:7px;background:var(--surface3);border-radius:4px;overflow:hidden;margin-top:6px;}
.score-fill{height:100%;border-radius:4px;}

.warp-status{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
    padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--muted);margin-bottom:12px;}
.warp-ok{border-color:rgba(34,197,94,.3);}.warp-warn{border-color:rgba(245,158,11,.3);}
.warp-err{border-color:rgba(239,68,68,.3);}

.btable{width:100%;border-collapse:collapse;font-size:.84rem;background:var(--surface);}
.btable th{background:var(--surface2);color:var(--muted);text-transform:uppercase;font-size:.69rem;
    font-weight:700;letter-spacing:1.5px;padding:13px 12px;border-bottom:1px solid var(--border);}
.btable td{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--text);}
.btable tr:hover td{background:rgba(249,115,22,.04);}
.btable td:first-child{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--muted);}
.bs{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;
    font-size:.71rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;}
.bs-c{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:#22C55E;}
.bs-w{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#EF4444;}
.bs-s{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:#F59E0B;}
.bs-m{background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.3);color:#A855F7;}

.legend{display:flex;gap:9px;flex-wrap:wrap;margin:10px 0 16px;}
.chip{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:20px;font-size:.73rem;font-weight:600;border:1px solid;}
.cg-c{background:rgba(34,197,94,.08);border-color:rgba(34,197,94,.3);color:#22C55E;}
.cg-r{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.3);color:#EF4444;}
.cg-a{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.3);color:#F59E0B;}
.cg-p{background:rgba(168,85,247,.08);border-color:rgba(168,85,247,.3);color:#A855F7;}
.cg-g{background:rgba(100,116,139,.08);border-color:rgba(100,116,139,.3);color:#64748B;}
.cg-b{background:rgba(96,165,250,.08);border-color:rgba(96,165,250,.3);color:#60A5FA;}

.sh{display:flex;align-items:center;gap:9px;margin:22px 0 12px;}
.sh h3{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--saffron);box-shadow:0 0 8px var(--saffron);}
.dlog{background:var(--surface2);border:1px solid var(--border);border-radius:9px;padding:14px;
    font-family:'JetBrains Mono',monospace;font-size:.74rem;color:var(--muted);margin-top:10px;
    max-height:220px;overflow-y:auto;line-height:1.7;}
.dlog .ok{color:#22C55E;}.dlog .warn{color:#F59E0B;}.dlog .info{color:#60A5FA;}.dlog .err{color:#EF4444;}

section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
section[data-testid="stSidebar"] label{font-weight:600!important;font-size:.83rem!important;}
.stSelectbox>div>div,.stTextInput>div>input,.stNumberInput>div>input{
    background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
.stButton>button{background:linear-gradient(135deg,#F97316,#EA580C)!important;color:#fff!important;
    border:none!important;border-radius:10px!important;padding:11px 26px!important;
    font-weight:700!important;font-family:'Syne',sans-serif!important;font-size:1rem!important;
    box-shadow:0 4px 16px rgba(249,115,22,.25)!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(249,115,22,.4)!important;}
.stProgress>div>div{background:var(--saffron)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CANONICAL GRID — HARDCODED FROM ACTUAL SHEET ANALYSIS
#  These positions are VERIFIED from the blank official OMR sheet at 300 DPI.
# ═══════════════════════════════════════════════════════════════════════════════

# Exact bubble center X per option per column (300 DPI canonical)
BUBBLE_CX = {
    'A': [380,  1114, 1848],
    'B': [502,  1236, 1966],
    'C': [620,  1354, 2088],
    'D': [744,  1476, 2210],
}
ROW_Y_START  = 720    # Y center of first answer row
ROW_SPACING  = 109    # Pixels between consecutive row centers
BUBBLE_R     = 23     # Outer circle radius (px)
INNER_R      = 17     # Inner measurement radius (avoiding outline artifacts)
CANONICAL_W  = 2479   # Canonical image width
CANONICAL_H  = 3508   # Canonical image height

# Corner anchor squares for homography (solid ■ markers)
# TL: first square row1, TR: last square row1, BL: first square row20, BR: last square row20
ANCHOR_TEMPLATE = np.float32([
    [270,  720],   # TL
    [2283, 720],   # TR
    [270,  2792],  # BL
    [2283, 2792],  # BR
])


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
    correct: int = 0; wrong: int = 0; unattempted: int = 0; multi: int = 0
    pos_score: float = 0; neg_score: float = 0; total_score: float = 0
    debug_log: List[str] = field(default_factory=list)
    warp_quality: str = "unknown"   # 'good', 'approx', 'none'
    warp_error_px: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  OMR ENGINE v5 — AUTO-CROP + PRECISE GRID
# ═══════════════════════════════════════════════════════════════════════════════
class OMREngine:

    def __init__(self):
        self.debug_log = []

    def log(self, msg: str, level: str = 'info'):
        prefix = {'info': '[INFO]', 'ok': '[OK]', 'warn': '[WARN]', 'err': '[ERR]'}.get(level, '[INFO]')
        self.debug_log.append(f"{prefix} {msg}")

    # ── Image Loading ──────────────────────────────────────────────────────────
    def load_pdf(self, pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        self.log(f"PDF loaded at {dpi} DPI → {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    def load_image(self, pil_img: Image.Image) -> np.ndarray:
        img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.log(f"Image loaded: {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    # ── Step 1: Find Solid Black Anchor Squares (■) ──────────────────────────
    def find_squares(self, gray: np.ndarray) -> List[dict]:
        """Detect solid filled black squares — the ■ timing marks on the sheet."""
        H, W = gray.shape[:2]
        scale = W / CANONICAL_W   # Normalize for any scan resolution

        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(50, int(200 * scale * scale))
        max_area = max(500, int(5000 * scale * scale))

        squares = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < int(8 * scale) or h < int(8 * scale):
                continue
            if not (0.45 < w / float(h) < 2.2):
                continue
            roi = binary[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            fill = np.count_nonzero(roi) / roi.size
            if fill < 0.65:
                continue
            if y < H * 0.10 or y > H * 0.95:
                continue
            squares.append({'x': x, 'y': y, 'w': w, 'h': h,
                             'cx': x+w//2, 'cy': y+h//2, 'area': area})

        self.log(f"Solid squares detected: {len(squares)}", 'info')
        return squares

    # ── Step 2: Auto-Crop via Homography ─────────────────────────────────────
    def find_warp(self, squares: List[dict], img_shape: tuple) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Find the 4 corner anchor squares and compute a homography to warp
        the scanned image to canonical coordinates.

        Returns: (M, quality_string, avg_reprojection_error_px)
        quality: 'good' | 'approx' | 'none'
        """
        H, W = img_shape[:2]
        scale = W / CANONICAL_W

        if not squares:
            self.log("No squares found — cannot warp", 'err')
            return None, 'none', 999.0

        # Cluster squares into rows by Y
        squares_sorted = sorted(squares, key=lambda s: s['cy'])
        row_gap = max(30, int(20 * scale))

        rows = []
        curr = [squares_sorted[0]]
        for s in squares_sorted[1:]:
            if abs(s['cy'] - curr[-1]['cy']) < row_gap:
                curr.append(s)
            else:
                rows.append(curr)
                curr = [s]
        rows.append(curr)

        # Filter to rows that have ~9 squares (answer rows)
        answer_rows = [r for r in rows if 6 <= len(r) <= 12]
        self.log(f"Answer rows found: {len(answer_rows)} (need 20)", 'info')

        if len(answer_rows) < 4:
            self.log("Too few answer rows — using fallback scale warp", 'warn')
            return self._scale_warp(img_shape), 'approx', 50.0

        # Corner squares: TL = leftmost of first row, TR = rightmost of first row
        #                 BL = leftmost of last row,  BR = rightmost of last row
        first_row = sorted(answer_rows[0], key=lambda s: s['cx'])
        last_row  = sorted(answer_rows[-1], key=lambda s: s['cx'])

        tl = first_row[0];  tr = first_row[-1]
        bl = last_row[0];   br = last_row[-1]

        src = np.float32([[tl['cx'], tl['cy']], [tr['cx'], tr['cy']],
                          [bl['cx'], bl['cy']], [br['cx'], br['cy']]])
        dst = ANCHOR_TEMPLATE.copy()

        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is None:
            self.log("Homography failed — using scale warp", 'warn')
            return self._scale_warp(img_shape), 'approx', 50.0

        # Compute reprojection error
        src_h = np.array([[p] for p in src], dtype=np.float32)
        projected = cv2.perspectiveTransform(src_h, M).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst, axis=1)
        avg_err = float(np.mean(errors))

        quality = 'good' if avg_err < 10 else 'approx'
        self.log(f"Homography computed: avg reprojection error = {avg_err:.1f}px → {quality}", 'ok')
        return M, quality, avg_err

    def _scale_warp(self, img_shape: tuple) -> np.ndarray:
        """Fallback: simple scale matrix when corner anchors not found."""
        H, W = img_shape[:2]
        sx = CANONICAL_W / W
        sy = CANONICAL_H / H
        M = np.float32([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return M

    def apply_warp(self, img: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Warp image to canonical frame."""
        return cv2.warpPerspective(img, M, (CANONICAL_W, CANONICAL_H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))

    # ── Step 3: Preprocess for Fill Measurement ───────────────────────────────
    def preprocess(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (adaptive_thresh, otsu_thresh)."""
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8
        )
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return adaptive, otsu

    # ── Step 4: Measure Fill ──────────────────────────────────────────────────
    def measure_fill(self, gray: np.ndarray, adaptive: np.ndarray,
                     otsu: np.ndarray, cx: int, cy: int) -> float:
        """
        3-method fill score inside an elliptical mask at (cx, cy) with INNER_R:
          1. Adaptive threshold pixel ratio
          2. OTSU threshold pixel ratio
          3. Darkness vs sheet background
        Returns combined [0, 1] fill score.
        """
        r = INNER_R
        H, W = gray.shape[:2]
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(W, cx + r), min(H, cy + r)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        h_roi, w_roi = y2 - y1, x2 - x1
        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        lx, ly = cx - x1, cy - y1
        cv2.circle(mask, (lx, ly), r - 1, 255, -1)

        total_px = np.count_nonzero(mask)
        if total_px == 0:
            return 0.0

        # Method 1: Adaptive threshold
        roi_a = adaptive[y1:y2, x1:x2]
        ratio_a = np.count_nonzero(roi_a & (mask > 0)) / total_px

        # Method 2: OTSU
        roi_o = otsu[y1:y2, x1:x2]
        ratio_o = np.count_nonzero(roi_o & (mask > 0)) / total_px

        # Method 3: Intensity darkness
        roi_g = gray[y1:y2, x1:x2]
        vals = roi_g[mask > 0]
        if len(vals) == 0:
            return max(ratio_a, ratio_o)
        sheet_bg = np.percentile(gray, 90)
        mean_intensity = np.mean(vals)
        darkness = max(0.0, (sheet_bg - mean_intensity) / max(sheet_bg, 1.0))

        # Weighted combination
        fill = ratio_a * 0.40 + ratio_o * 0.30 + darkness * 0.30
        return float(min(1.0, max(0.0, fill)))

    # ── Step 5: Classify Selection ────────────────────────────────────────────
    def classify(self, fills: Dict[str, float], thresh: float) -> List[str]:
        """
        Classify which bubble(s) are selected.
        Uses absolute threshold + dominance check to reduce false multi-marks.
        """
        if not fills:
            return []
        max_fill = max(fills.values())
        if max_fill < thresh * 0.35:
            return []   # All empty

        sorted_f = sorted(fills.items(), key=lambda x: x[1], reverse=True)

        # Dominance: if top is ≥1.8× second, only top selected
        if len(sorted_f) >= 2 and sorted_f[0][1] >= thresh * 0.6:
            dominance = sorted_f[0][1] / (sorted_f[1][1] + 1e-6)
            if dominance >= 1.8:
                return [sorted_f[0][0]]

        # Otherwise: all above threshold
        above = [opt for opt, f in fills.items() if f >= thresh]
        if above:
            return above

        # Soft fallback
        if sorted_f[0][1] >= thresh * 0.55:
            return [sorted_f[0][0]]
        return []

    # ── Main Grade Pipeline ───────────────────────────────────────────────────
    def grade(self, img: np.ndarray, answer_key: dict,
              pos: float = 3.0, neg: float = 1.0,
              fill_thresh: float = 0.22) -> Tuple[OMRResult, np.ndarray, np.ndarray]:
        """
        Returns: (OMRResult, warped_canonical_img, annotated_img)
        """
        self.debug_log = []
        H_orig, W_orig = img.shape[:2]
        self.log(f"Input: {W_orig}×{H_orig}px", 'info')

        # ── Auto-crop via homography ──────────────────────────────────────────
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squares = self.find_squares(gray_orig)
        M, warp_quality, warp_err = self.find_warp(squares, img.shape)

        if M is not None:
            warped = self.apply_warp(img, M)
            self.log(f"Warp applied → {CANONICAL_W}×{CANONICAL_H}px", 'ok')
        else:
            # No warp — scale to canonical size
            warped = cv2.resize(img, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_LINEAR)
            warp_quality = 'none'
            self.log("No warp — resized to canonical", 'warn')

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        adaptive, otsu = self.preprocess(gray)

        # ── Sample all 60 × 4 bubbles ─────────────────────────────────────────
        annotated = warped.copy()
        results = []

        for col_idx in range(3):
            for row_idx in range(20):
                q = col_idx * 20 + row_idx + 1
                cy = ROW_Y_START + row_idx * ROW_SPACING
                key = answer_key.get(q, '')

                fills = {}
                for opt in ['A', 'B', 'C', 'D']:
                    cx = BUBBLE_CX[opt][col_idx]
                    fills[opt] = self.measure_fill(gray, adaptive, otsu, cx, cy)

                selected = self.classify(fills, fill_thresh)

                # Grade
                if len(selected) == 0:
                    status = 'unattempted'; score = 0.0
                elif len(selected) > 1:
                    status = 'multi'; score = -neg if key else 0.0
                elif key and selected[0] == key:
                    status = 'correct'; score = pos
                elif key:
                    status = 'wrong'; score = -neg
                else:
                    status = 'unattempted'; score = 0.0

                results.append(BubbleResult(q_num=q, detected=selected, answer_key=key,
                                            status=status, score=score, fill_values=fills))

                # ── Annotate ─────────────────────────────────────────────────
                STATUS_COLORS = {
                    'correct':    (50, 210, 50),
                    'wrong':      (50, 50, 230),
                    'multi':      (200, 50, 220),
                    'unattempted': (110, 110, 110),
                }
                for opt in ['A', 'B', 'C', 'D']:
                    cx = BUBBLE_CX[opt][col_idx]
                    # Detection zone ring
                    cv2.circle(annotated, (cx, cy), BUBBLE_R + 3, (40, 40, 65), 1)

                    if opt in selected:
                        clr = STATUS_COLORS.get(status, (180, 180, 180))
                        cv2.circle(annotated, (cx, cy), BUBBLE_R, clr, -1)
                        cv2.circle(annotated, (cx, cy), BUBBLE_R + 1, (255, 255, 255), 1)
                        cv2.putText(annotated, opt, (cx - 6, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    else:
                        fv = fills.get(opt, 0)
                        cv2.circle(annotated, (cx, cy), BUBBLE_R, (100, 100, 110), 1)
                        if fv > 0.06:
                            cv2.putText(annotated, f"{fv:.2f}", (cx - 10, cy + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 150, 70), 1)

                    # Highlight correct answer in green if missed
                    if key == opt and opt not in selected and status not in ('correct', 'unattempted'):
                        cv2.circle(annotated, (cx, cy), BUBBLE_R + 5, (50, 210, 50), 2)

                # Q label
                lx = BUBBLE_CX['A'][col_idx] - 55
                lc = STATUS_COLORS.get(status, (150, 150, 150))
                cv2.putText(annotated, f"Q{q:02d}", (lx, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, lc, 1)

        correct   = sum(1 for r in results if r.status == 'correct')
        wrong     = sum(1 for r in results if r.status == 'wrong')
        unat      = sum(1 for r in results if r.status == 'unattempted')
        multi     = sum(1 for r in results if r.status == 'multi')
        ps = correct * pos
        ns = (wrong + multi) * neg
        self.log(f"Score: {correct}✓ {wrong}✗ {unat}— {multi}M = {ps-ns:.1f}", 'ok')

        return OMRResult(
            bubbles=results, correct=correct, wrong=wrong, unattempted=unat, multi=multi,
            pos_score=ps, neg_score=ns, total_score=ps - ns,
            debug_log=list(self.debug_log),
            warp_quality=warp_quality, warp_error_px=warp_err
        ), warped, annotated


# ─── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ('answer_key', {i: random.choice(['A','B','C','D']) for i in range(1,61)}),
    ('result', None), ('original_img', None),
    ('warped_img', None), ('annotated_img', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

engine = OMREngine()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="omr-header">
  <div class="tricolor"><div></div><div></div><div></div></div>
  <h1 class="omr-title">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="omr-sub">Precision OMR Grader v5.0 &nbsp;·&nbsp; Tiranga Yuva Samiti &nbsp;·&nbsp; Auto-Crop + Exact Grid</p>
  <span class="tech-pill">■ Auto-Crop via Homography</span>
  &nbsp;
  <span class="tech-pill tech-pill-o">○ Hardcoded Verified Bubble Grid</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    pos_mark = st.number_input("✅ Correct (+)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    neg_mark = st.number_input("❌ Wrong (−)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    st.markdown("---")
    st.markdown("**🎯 Fill Detection**")
    fill_threshold = st.slider("Fill Threshold", 0.06, 0.55, 0.20, 0.01,
        help="Lower = picks up lighter pencil marks. 0.15–0.25 recommended.")
    st.caption(f"Current: **{fill_threshold:.2f}** — try 0.12 for very light marks")
    show_debug = st.checkbox("🔍 Show Debug Log", False)
    show_fills = st.checkbox("Show raw fill values", False)
    st.markdown("---")
    st.markdown("### 📋 Answer Key")
    bulk_key = st.text_area("Paste 60 answers (comma-separated)", placeholder="A,B,C,D,...", height=70)
    if st.button("Apply Bulk Key", use_container_width=True):
        parts = [p.strip().upper() for p in bulk_key.split(',')]
        for i, ans in enumerate(parts[:60]):
            if ans in ('A','B','C','D',''):
                st.session_state.answer_key[i+1] = ans
        st.success("✅ Applied!")
    opts_list = ['', 'A', 'B', 'C', 'D']
    st.caption("Or set individually:")
    ck = st.columns(2)
    for q in range(1, 61):
        col = ck[0] if q % 2 != 0 else ck[1]
        with col:
            st.session_state.answer_key[q] = st.selectbox(
                f"Q{q}", opts_list,
                index=opts_list.index(st.session_state.answer_key.get(q, '')),
                key=f"key_{q}"
            )

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh"><div class="sdot"></div><h3>Upload OMR Sheet</h3></div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend">
  <span class="chip cg-c">● Correct</span>
  <span class="chip cg-r">● Wrong</span>
  <span class="chip cg-g">○ Unattempted</span>
  <span class="chip cg-a">⚑ Skipped</span>
  <span class="chip cg-p">● Multi-Mark</span>
  <span class="chip cg-b">◎ Correct Answer (missed)</span>
</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 20px;margin-bottom:14px;">
  <p style="color:var(--muted);font-size:.85rem;line-height:1.7;margin:0;">
    <strong style="color:#F97316;">Auto-Crop:</strong> Detects the 4 corner ■ squares → computes perspective warp → normalizes to canonical frame<br>
    <strong style="color:#22C55E;">Exact Grid:</strong> All 60 × 4 = 240 bubble positions hardcoded from official blank sheet (verified at 300 DPI)<br>
    <strong style="color:#60A5FA;">Best results:</strong> PDF scan (any DPI) · PNG/JPG (≥150 DPI) · Keep sheet flat and well-lit
  </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Drop OMR sheet here — PDF or image",
                             type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'])

if uploaded:
    with st.spinner("Loading..."):
        fb = uploaded.read()
        if uploaded.type == 'application/pdf':
            img_cv = engine.load_pdf(fb, dpi=300)
        else:
            img_cv = engine.load_image(Image.open(io.BytesIO(fb)))
        st.session_state.original_img = img_cv.copy()
    st.success(f"✅ **{uploaded.name}** — {img_cv.shape[1]}×{img_cv.shape[0]}px")

    col_orig, col_act = st.columns([1, 1])
    with col_orig:
        st.markdown("**📄 Original Upload**")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_act:
        st.markdown("**🔬 Grade Settings**")
        st.info(f"""
**Marking:** +{pos_mark:.1f} correct · −{neg_mark:.1f} wrong  
**Fill Threshold:** {fill_threshold:.2f}  
**Auto-Crop:** Perspective warp via corner ■ anchors  
**Grid:** 60 questions × 4 options (hardcoded exact positions)
        """)
        if st.button("🚀  Grade OMR Sheet", use_container_width=True):
            bar = st.progress(0, text="Finding corner anchor squares...")
            time.sleep(0.1)
            bar.progress(20, text="Computing perspective warp...")
            time.sleep(0.1)
            bar.progress(40, text="Warping to canonical frame...")

            result, warped, annotated = engine.grade(
                img_cv, st.session_state.answer_key,
                pos=pos_mark, neg=neg_mark, fill_thresh=fill_threshold
            )

            bar.progress(75, text="Sampling 240 bubble positions...")
            st.session_state.result = result
            st.session_state.warped_img = warped
            st.session_state.annotated_img = annotated
            time.sleep(0.1)
            bar.progress(100, text="Done!")
            time.sleep(0.2)
            bar.empty()

            wq = result.warp_quality
            if wq == 'good':
                st.success(f"✅ Graded! Auto-crop: **GOOD** (reprojection error: {result.warp_error_px:.1f}px)")
            elif wq == 'approx':
                st.warning(f"⚠️ Graded with approximate alignment (error: {result.warp_error_px:.1f}px) — results may vary slightly")
            else:
                st.error("⚠️ Could not find corner anchors — used simple scale warp. Results may be inaccurate.")


# ─── RESULTS ──────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    result = st.session_state.result
    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Results Dashboard</h3></div>', unsafe_allow_html=True)

    # Warp status
    wq = result.warp_quality
    wq_cls = {'good': 'warp-ok', 'approx': 'warp-warn', 'none': 'warp-err'}.get(wq, '')
    wq_icon = {'good': '✅', 'approx': '⚠️', 'none': '❌'}.get(wq, '?')
    wq_desc = {
        'good': f'Perspective warp: GOOD — reprojection error {result.warp_error_px:.1f}px',
        'approx': f'Perspective warp: APPROXIMATE — error {result.warp_error_px:.1f}px (not all corners found)',
        'none': 'No warp applied — simple scale resize used (low confidence)',
    }.get(wq, '')
    st.markdown(f'<div class="warp-status {wq_cls}">{wq_icon} {wq_desc}</div>', unsafe_allow_html=True)

    max_score = 60 * pos_mark
    pct = (result.total_score / max_score * 100) if max_score > 0 else 0
    if pct >= 75:   bcls, bico, btxt = "rb-ex", "🏆", "Outstanding Performance!"
    elif pct >= 50: bcls, bico, btxt = "rb-gd", "👍", "Good Performance"
    elif pct >= 35: bcls, bico, btxt = "rb-av", "📚", "Average — Keep Practicing"
    else:            bcls, bico, btxt = "rb-pr", "⚠️", "Needs Improvement"
    bar_clr = "#22C55E" if pct>=75 else ("#F59E0B" if pct>=50 else ("#F97316" if pct>=35 else "#EF4444"))

    st.markdown(f"""
    <div class="rbanner {bcls}">
      <span style="font-size:2rem;">{bico}</span>
      <div>
        <div>{btxt}</div>
        <div style="font-size:.84rem;font-weight:400;opacity:.8;margin-top:3px;">
          Score: <strong>{result.total_score:.1f}</strong>/{max_score:.0f} &nbsp;·&nbsp; {pct:.1f}%
        </div>
      </div>
    </div>
    <div class="score-track"><div class="score-fill"
      style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{bar_clr},{bar_clr}88);"></div></div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card c"><div class="sn cg">{result.correct}</div><div class="sl">Correct</div><div class="glow"></div></div>
      <div class="stat-card w"><div class="sn cr">{result.wrong}</div><div class="sl">Wrong</div><div class="glow"></div></div>
      <div class="stat-card s"><div class="sn ca">{result.unattempted}</div><div class="sl">Skipped</div><div class="glow"></div></div>
      <div class="stat-card m"><div class="sn cp">{result.multi}</div><div class="sl">Multi-Mark</div><div class="glow"></div></div>
      <div class="stat-card sc"><div class="sn co">{result.total_score:.1f}</div><div class="sl">Net Score</div><div class="glow"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**📉 Breakdown**")
        attempted = result.correct + result.wrong + result.multi
        acc = (result.correct / attempted * 100) if attempted > 0 else 0
        m1, m2 = st.columns(2)
        m1.metric("Positive Score", f"+{result.pos_score:.1f}")
        m2.metric("Negative Score", f"−{result.neg_score:.1f}")
        m3, m4 = st.columns(2)
        m3.metric("Net Score", f"{result.total_score:.1f}")
        m4.metric("Accuracy", f"{acc:.1f}%")

        st.markdown(f"""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;
             padding:13px 16px;margin-top:8px;">
          <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Attempt Stats</div>
          <div style="display:flex;justify-content:space-between;font-size:.86rem;padding:3px 0;">
            <span>Attempted</span><span style="font-weight:700;">{attempted}/60 ({attempted/60*100:.0f}%)</span></div>
          <div style="display:flex;justify-content:space-between;font-size:.86rem;padding:3px 0;">
            <span>Unattempted</span><span style="font-weight:700;">{result.unattempted}/60</span></div>
          <div style="display:flex;justify-content:space-between;font-size:.86rem;padding:3px 0;">
            <span>Multi-marked</span><span style="font-weight:700;">{result.multi}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:.86rem;padding:3px 0;">
            <span>Warp quality</span>
            <span style="font-weight:700;color:{'#22C55E' if result.warp_quality=='good' else '#F59E0B' if result.warp_quality=='approx' else '#EF4444'}">
              {result.warp_quality.upper()}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if show_debug and result.debug_log:
            st.markdown("**🔍 Debug Log**")
            log_html = "".join(
                f'<div class="{"ok" if "[OK]" in l else "warn" if "[WARN]" in l else "err" if "[ERR]" in l else "info"}">{l}</div>'
                for l in result.debug_log
            )
            st.markdown(f'<div class="dlog">{log_html}</div>', unsafe_allow_html=True)

    with c2:
        tab1, tab2 = st.tabs(["🎯 Annotated", "🔲 Warped (Canonical)"])
        with tab1:
            if st.session_state.annotated_img is not None:
                ann_rgb = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
                st.image(ann_rgb, use_container_width=True)
                buf = io.BytesIO()
                Image.fromarray(ann_rgb).save(buf, format='PNG')
                st.download_button("⬇️ Download Annotated", buf.getvalue(),
                                   "annotated_omr.png", "image/png", use_container_width=True)
        with tab2:
            if st.session_state.warped_img is not None:
                w_rgb = cv2.cvtColor(st.session_state.warped_img, cv2.COLOR_BGR2RGB)
                st.image(w_rgb, use_container_width=True)
                buf2 = io.BytesIO()
                Image.fromarray(w_rgb).save(buf2, format='PNG')
                st.download_button("⬇️ Download Warped", buf2.getvalue(),
                                   "warped_omr.png", "image/png", use_container_width=True)

    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Question-wise Report</h3></div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        filter_status = st.multiselect("Filter", ['correct','wrong','unattempted','multi'],
                                        default=['correct','wrong','unattempted','multi'])
    with fc2:
        pass

    filtered = [b for b in result.bubbles if b.status in filter_status]

    rows_html = ""
    for b in filtered:
        det = ', '.join(b.detected) if b.detected else '—'
        key = b.answer_key if b.answer_key else '—'
        sc = f"+{b.score:.0f}" if b.score > 0 else (f"{b.score:.0f}" if b.score != 0 else "0")
        sc_cls = "cg" if b.score > 0 else ("cr" if b.score < 0 else "ca")
        bc = {'correct':'bs-c','wrong':'bs-w','unattempted':'bs-s','multi':'bs-m'}.get(b.status,'')
        bi = {'correct':'✓','wrong':'✗','unattempted':'—','multi':'×'}.get(b.status,'')
        fv_html = ""
        if show_fills and b.fill_values:
            fv_str = "  ".join(f"{k}:{v:.3f}" for k,v in sorted(b.fill_values.items()))
            fv_html = f"<td style='font-size:.7rem;color:var(--muted);font-family:JetBrains Mono,monospace;white-space:nowrap;'>{fv_str}</td>"
        rows_html += f"""<tr>
          <td>Q{b.q_num:02d}</td>
          <td><span style="color:#60A5FA;font-weight:700;font-family:'JetBrains Mono',monospace;">{det}</span></td>
          <td><span style="color:#34D399;font-weight:700;font-family:'JetBrains Mono',monospace;">{key}</span></td>
          <td><span class="bs {bc}">{bi} {b.status.upper()}</span></td>
          <td class="{sc_cls}" style="font-weight:800;font-family:'JetBrains Mono',monospace;">{sc}</td>
          {fv_html}
        </tr>"""

    extra_th = "<th>Fill A / B / C / D</th>" if show_fills else ""
    st.markdown(f"""
    <div style="max-height:520px;overflow-y:auto;border:1px solid var(--border);border-radius:12px;
         box-shadow:0 4px 20px rgba(0,0,0,.4);">
    <table class="btable">
      <thead><tr><th>Q#</th><th>Detected</th><th>Key</th><th>Status</th><th>Score</th>{extra_th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    st.write("")
    exp = [{'Q': b.q_num, 'Detected': ','.join(b.detected) if b.detected else '',
             'Key': b.answer_key, 'Status': b.status, 'Score': b.score,
             **{f'Fill_{k}': round(v,4) for k,v in b.fill_values.items()}}
            for b in result.bubbles]
    st.download_button("⬇️ Download Results CSV",
                        pd.DataFrame(exp).to_csv(index=False),
                        "omr_results.csv", "text/csv")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:var(--muted);font-size:.78rem;">
  <div class="tricolor" style="max-width:140px;margin:0 auto 10px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · OMR Grader v5.0<br>
  <span style="font-size:.68rem;opacity:.5;">Auto-Crop Homography · Verified Hardcoded Grid · 3-Method Fill Scoring</span>
</div>
""", unsafe_allow_html=True)
