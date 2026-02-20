"""
YUVA GYAN MAHOTSAV 2026 — PRECISION OMR GRADER v7.0
=====================================================
Pixel-perfect bubble detection verified on actual sheet (1240×1754 @ 150 DPI)

Structure per row: ■(L)  A○  B○  C○  D○  ■(R)   × 3 columns = 60 questions

VERIFIED EXACT PIXEL POSITIONS (150 DPI canonical):
  Col 1 (Q01-20): L_sq=135, A=189, B=250, C=310, D=372, R_sq=409
  Col 2 (Q21-40): L_sq=502, A=557, B=618, C=677, D=738, R_sq=776
  Col 3 (Q41-60): L_sq=870, A=924, B=983, C=1044, D=1105, R_sq=1142

  Row 1 Y=360, Row spacing=54.5px, Bubble radius=15px

AUTO-CROP via homography using 6 anchor squares per row (L+R of each col).
Fill detection: relative darkness method — filled bubbles ~1.0, empty ~0.33
Threshold: filled if fill_score > 0.5 AND at least 60% of max in row
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import random

try:
    from pdf2image import convert_from_bytes
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

st.set_page_config(
    page_title="YGM 2026 OMR Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --s:#F97316;--sd:#EA580C;--g:#22C55E;--gd:#16A34A;--gold:#F59E0B;
  --bg:#080D1A;--sf:#0F1923;--sf2:#162032;--sf3:#1D2D45;
  --br:rgba(255,255,255,.07);--tx:#E2E8F0;--mu:#64748B;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background:var(--bg);color:var(--tx);}
.stApp{background:var(--bg)!important;}
.main .block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1440px;}

.hdr{background:linear-gradient(135deg,#0A1628,#0F1F3D 50%,#0A1628);
  border:1px solid rgba(249,115,22,.22);border-radius:18px;padding:28px 36px 22px;
  margin-bottom:22px;position:relative;overflow:hidden;}
.hdr::before{content:'';position:absolute;top:-80px;right:-80px;width:260px;height:260px;
  background:radial-gradient(circle,rgba(249,115,22,.13),transparent 65%);border-radius:50%;}
.tricolor{display:flex;height:4px;border-radius:2px;overflow:hidden;margin-bottom:14px;max-width:340px;}
.tricolor div:nth-child(1){flex:1;background:linear-gradient(90deg,#F97316,#EA580C);}
.tricolor div:nth-child(2){flex:1;background:#E2E8F0;}
.tricolor div:nth-child(3){flex:1;background:linear-gradient(90deg,#22C55E,#16A34A);}
.htitle{font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;
  background:linear-gradient(130deg,#F1F5F9 20%,#F97316 60%,#22C55E 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.1;position:relative;z-index:1;}
.hsub{font-size:.86rem;color:var(--mu);margin-top:6px;z-index:1;position:relative;}
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;
  font-size:.72rem;font-weight:700;font-family:'JetBrains Mono',monospace;margin-top:8px;}
.pill-g{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);color:#22C55E;}
.pill-o{background:rgba(249,115,22,.1);border:1px solid rgba(249,115,22,.25);color:#F97316;}
.pill-b{background:rgba(96,165,250,.1);border:1px solid rgba(96,165,250,.25);color:#60A5FA;}

.sgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:16px 0;}
.sc{background:var(--sf);border:1px solid var(--br);border-radius:14px;padding:18px 14px;
  text-align:center;position:relative;overflow:hidden;transition:transform .2s;}
.sc:hover{transform:translateY(-2px);}
.sc.c{border-color:rgba(34,197,94,.3);}.sc.w{border-color:rgba(239,68,68,.3);}
.sc.s{border-color:rgba(245,158,11,.3);}.sc.m{border-color:rgba(168,85,247,.3);}
.sc.total{border-color:rgba(249,115,22,.4);background:linear-gradient(135deg,rgba(249,115,22,.06),var(--sf));}
.gl{position:absolute;bottom:0;left:0;right:0;height:2px;}
.sc.c .gl{background:#22C55E;box-shadow:0 0 8px #22C55E;}.sc.w .gl{background:#EF4444;box-shadow:0 0 8px #EF4444;}
.sc.s .gl{background:#F59E0B;box-shadow:0 0 8px #F59E0B;}.sc.m .gl{background:#A855F7;box-shadow:0 0 8px #A855F7;}
.sc.total .gl{background:#F97316;box-shadow:0 0 12px #F97316;}
.snum{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;line-height:1;letter-spacing:-1px;}
.slbl{font-size:.67rem;color:var(--mu);text-transform:uppercase;letter-spacing:2px;margin-top:5px;font-weight:600;}
.cg{color:#22C55E;}.cr{color:#EF4444;}.ca{color:#F59E0B;}.co{color:#F97316;}.cp{color:#A855F7;}

.rbanner{border-radius:14px;padding:18px 22px;margin:12px 0;display:flex;align-items:center;
  gap:14px;font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;}
.rb-ex{background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.3);color:#22C55E;}
.rb-gd{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.3);color:#F59E0B;}
.rb-av{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.3);color:#F97316;}
.rb-pr{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.3);color:#EF4444;}
.strack{height:7px;background:var(--sf3);border-radius:4px;overflow:hidden;margin-top:5px;}
.sfill{height:100%;border-radius:4px;}

.wstat{background:var(--sf2);border:1px solid var(--br);border-radius:10px;
  padding:11px 16px;font-family:'JetBrains Mono',monospace;font-size:.77rem;color:var(--mu);margin-bottom:10px;}
.wok{border-color:rgba(34,197,94,.35);}.wwarn{border-color:rgba(245,158,11,.35);}.werr{border-color:rgba(239,68,68,.35);}

.bt{width:100%;border-collapse:collapse;font-size:.84rem;background:var(--sf);}
.bt th{background:var(--sf2);color:var(--mu);text-transform:uppercase;font-size:.68rem;
  font-weight:700;letter-spacing:1.5px;padding:12px;border-bottom:1px solid var(--br);}
.bt td{padding:10px 12px;border-bottom:1px solid var(--br);}
.bt tr:hover td{background:rgba(249,115,22,.04);}
.bt td:first-child{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--mu);}
.bs{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;
  font-size:.7rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;}
.bs-c{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:#22C55E;}
.bs-w{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#EF4444;}
.bs-s{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:#F59E0B;}
.bs-m{background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.3);color:#A855F7;}

.legend{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px;}
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:600;border:1px solid;}
.ch-g{background:rgba(34,197,94,.08);border-color:rgba(34,197,94,.3);color:#22C55E;}
.ch-r{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.3);color:#EF4444;}
.ch-a{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.3);color:#F59E0B;}
.ch-p{background:rgba(168,85,247,.08);border-color:rgba(168,85,247,.3);color:#A855F7;}
.ch-n{background:rgba(100,116,139,.08);border-color:rgba(100,116,139,.3);color:#64748B;}
.ch-b{background:rgba(96,165,250,.08);border-color:rgba(96,165,250,.3);color:#60A5FA;}

.sh{display:flex;align-items:center;gap:9px;margin:20px 0 10px;}
.sh h3{font-family:'Syne',sans-serif;font-size:1.12rem;font-weight:700;}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--s);box-shadow:0 0 8px var(--s);}

.dlog{background:var(--sf2);border:1px solid var(--br);border-radius:9px;padding:13px;
  font-family:'JetBrains Mono',monospace;font-size:.73rem;color:var(--mu);
  max-height:200px;overflow-y:auto;line-height:1.75;}
.dlog .ok{color:#22C55E;}.dlog .warn{color:#F59E0B;}.dlog .info{color:#60A5FA;}.dlog .err{color:#EF4444;}

.struct-diagram{background:var(--sf2);border:1px solid rgba(249,115,22,.2);border-radius:10px;
  padding:12px 18px;font-family:'JetBrains Mono',monospace;font-size:.82rem;margin-bottom:14px;
  color:var(--tx);line-height:2;}
.sq{color:#F97316;font-weight:700;}.cir{color:#60A5FA;}.msq{color:#94A3B8;}

section[data-testid="stSidebar"]{background:var(--sf)!important;border-right:1px solid var(--br)!important;}
section[data-testid="stSidebar"] *{color:var(--tx)!important;}
section[data-testid="stSidebar"] label{font-weight:600!important;font-size:.82rem!important;}
.stSelectbox>div>div,.stTextInput>div>input,.stNumberInput>div>input{
  background:var(--sf2)!important;border-color:var(--br)!important;color:var(--tx)!important;border-radius:8px!important;}
.stButton>button{background:linear-gradient(135deg,#F97316,#EA580C)!important;color:#fff!important;
  border:none!important;border-radius:10px!important;padding:11px 26px!important;
  font-weight:700!important;font-family:'Syne',sans-serif!important;font-size:1rem!important;
  box-shadow:0 4px 16px rgba(249,115,22,.25)!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(249,115,22,.4)!important;}
.stProgress>div>div{background:var(--s)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--tx)!important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CANONICAL GRID — VERIFIED ON ACTUAL YUVA GYAN MAHOTSAV 2026 SHEET
#  Image reference: 1240×1754 pixels (150 DPI scan)
#
#  Structure per row: ■(L)  A○  B○  C○  D○  ■(R)   ← no "mid" square
#
#  Col 1 (Q01-20): L_sq=135  A=189  B=250  C=310  D=372  R_sq=409
#  Col 2 (Q21-40): L_sq=502  A=557  B=618  C=677  D=738  R_sq=776
#  Col 3 (Q41-60): L_sq=870  A=924  B=983  C=1044  D=1105  R_sq=1142
#
#  Row 1 Y=360,  Row spacing=54.5px,  Bubble sampling radius=12px
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical (150 DPI) dimensions
CANONICAL_W = 1240
CANONICAL_H = 1754

# Bubble center X per option, per column index
BUBBLE_CX = {
    0: {'A': 189, 'B': 250, 'C': 310, 'D': 372},   # Col1  Q01-20
    1: {'A': 557, 'B': 618, 'C': 677, 'D': 738},   # Col2  Q21-40
    2: {'A': 924, 'B': 983, 'C': 1044, 'D': 1105}, # Col3  Q41-60
}

# Anchor square CX (Left / Right of each column)
SQ_L = [135, 502, 870]   # Left anchors
SQ_R = [409, 776, 1142]  # Right anchors

# Row geometry
ROW_Y_START  = 360
ROW_SPACING  = 54.5    # pixels between rows
BUBBLE_R     = 15      # drawing radius
SAMPLE_R     = 12      # sampling radius (inner, avoids circle edge)
ROW_LAST_Y   = int(ROW_Y_START + 19 * ROW_SPACING)  # =1396

# Homography warp corners: TL=top-left L_sq of col1, TR=R_sq of col3, etc.
WARP_DST = np.float32([
    [SQ_L[0], ROW_Y_START],   # TL  (135, 360)
    [SQ_R[2], ROW_Y_START],   # TR  (1142, 360)
    [SQ_L[0], ROW_LAST_Y],    # BL  (135, 1396)
    [SQ_R[2], ROW_LAST_Y],    # BR  (1142, 1396)
])


# ─── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class BubbleResult:
    q_num: int
    detected: list
    answer_key: str
    status: str
    score: float
    fill_values: dict = field(default_factory=dict)
    confidence: float = 1.0

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
    warp_quality: str = "none"
    warp_error_px: float = 999.0
    scale_x: float = 1.0
    scale_y: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class OMREngine:

    def __init__(self):
        self.log_lines: List[str] = []

    def _log(self, msg: str, lvl: str = 'info'):
        tag = {'info': '[INFO]', 'ok': '[OK]', 'warn': '[WARN]', 'err': '[ERR]'}.get(lvl, '[INFO]')
        self.log_lines.append(f"{tag} {msg}")

    # ── Image Loaders ─────────────────────────────────────────────────────────
    def load_pdf(self, pdf_bytes: bytes, dpi: int = 150) -> np.ndarray:
        if not HAS_PDF:
            raise RuntimeError("pdf2image not installed")
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        self._log(f"PDF → {img.shape[1]}×{img.shape[0]}px at {dpi} DPI", 'ok')
        return img

    def load_img(self, pil: Image.Image) -> np.ndarray:
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._log(f"Image → {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    # ── Step 1: Flatten / Deskew via Anchor Squares ───────────────────────────
    def _find_anchor_squares(self, gray: np.ndarray, scale_x: float, scale_y: float) -> List[dict]:
        """Find solid black filled ■ squares on the sheet."""
        _, bw = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape
        squares = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            min_area = 60 * scale_x * scale_y
            max_area = 5000 * scale_x * scale_y
            if not (min_area < area < max_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.35 < w / max(h, 1) < 2.8):
                continue
            if y < H * 0.10 or (y + h) > H * 0.97:
                continue
            roi = bw[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            fill = np.count_nonzero(roi) / roi.size
            if fill < 0.55:
                continue
            squares.append({
                'cx': x + w // 2, 'cy': y + h // 2,
                'x': x, 'y': y, 'w': w, 'h': h
            })
        return squares

    def _cluster_into_rows(self, squares: List[dict], gap: int = 18) -> List[List[dict]]:
        if not squares:
            return []
        squares = sorted(squares, key=lambda s: s['cy'])
        rows, cur = [], [squares[0]]
        for s in squares[1:]:
            if abs(s['cy'] - cur[-1]['cy']) < gap:
                cur.append(s)
            else:
                rows.append(cur)
                cur = [s]
        rows.append(cur)
        return rows

    def auto_warp(self, img: np.ndarray) -> Tuple[np.ndarray, str, float, float, float]:
        """
        Find corner anchor squares → compute homography → warp to canonical frame.
        Returns (warped_img, quality, reprojection_error, scale_x, scale_y)
        """
        H, W = img.shape[:2]
        scale_x = W / CANONICAL_W
        scale_y = H / CANONICAL_H
        self._log(f"Input: {W}×{H}  scale: {scale_x:.3f}×{scale_y:.3f}", 'info')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squares = self._find_anchor_squares(gray, scale_x, scale_y)
        rows = self._cluster_into_rows(squares, gap=int(18 * scale_y))

        # Answer rows have 6+ squares (L+R per 3 columns, sometimes more from filled bubbles)
        ans_rows = [r for r in rows if len(r) >= 5]
        self._log(f"Anchor rows detected: {len(ans_rows)}", 'info')

        if len(ans_rows) < 4:
            self._log("Insufficient anchor rows → simple resize", 'warn')
            warped = cv2.resize(img, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_LINEAR)
            return warped, 'none', 999.0, scale_x, scale_y

        ans_rows.sort(key=lambda r: np.mean([s['cy'] for s in r]))
        first = sorted(ans_rows[0], key=lambda s: s['cx'])
        last  = sorted(ans_rows[-1], key=lambda s: s['cx'])

        tl, tr = first[0], first[-1]
        bl, br = last[0], last[-1]

        src = np.float32([
            [tl['cx'], tl['cy']], [tr['cx'], tr['cy']],
            [bl['cx'], bl['cy']], [br['cx'], br['cy']]
        ])

        M, _ = cv2.findHomography(src, WARP_DST, cv2.RANSAC, 5.0)
        if M is None:
            self._log("Homography failed → simple resize", 'warn')
            warped = cv2.resize(img, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_LINEAR)
            return warped, 'approx', 99.0, scale_x, scale_y

        pts   = src.reshape(-1, 1, 2)
        proj  = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        err   = float(np.mean(np.linalg.norm(proj - WARP_DST, axis=1)))
        qual  = 'good' if err < 12 else 'approx'
        self._log(f"Homography reprojection error: {err:.1f}px → {qual}", 'ok')

        warped = cv2.warpPerspective(img, M, (CANONICAL_W, CANONICAL_H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return warped, qual, err, 1.0, 1.0

    # ── Step 2: Fill Measurement ───────────────────────────────────────────────
    def _measure_fill(self, gray: np.ndarray, cx: int, cy: int, r: int = SAMPLE_R) -> float:
        """
        Relative darkness method:
          1. Sample pixels inside circle of radius r
          2. Compare mean to sheet background (90th percentile brightness)
          3. Returns [0..1] where ~0.33 = empty (paper), ~1.0 = fully filled
        
        This method is robust against:
          - Varying scan brightness / contrast
          - Pencil vs pen fills
          - Slightly off-center bubbles
        """
        H, W = gray.shape
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(W, cx + r), min(H, cy + r)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Circular mask
        h_r, w_r = y2 - y1, x2 - x1
        mask = np.zeros((h_r, w_r), dtype=np.uint8)
        cv2.circle(mask, (cx - x1, cy - y1), r - 1, 255, -1)
        roi_vals = gray[y1:y2, x1:x2][mask > 0]
        if len(roi_vals) == 0:
            return 0.0

        # Background = 90th percentile of full image brightness
        bg = float(np.percentile(gray, 90))
        mean_dark = float(np.mean(roi_vals))
        # Normalize: how much darker than background?
        darkness = max(0.0, (bg - mean_dark) / max(bg, 1.0))
        # Scale so that a well-inked bubble → ~1.0
        return min(1.0, darkness * 2.8)

    # ── Step 3: Classify Selected Bubbles ─────────────────────────────────────
    def _classify(self, fills: Dict[str, float]) -> Tuple[List[str], float]:
        """
        Determine which bubble(s) are filled using relative threshold.
        Returns (selected_options, confidence).
        
        Logic:
          - If max fill < ABSOLUTE_MIN → unattempted
          - One bubble dominates (ratio ≥ 2.5×) → single answer
          - Multiple bubbles above relative threshold → multi-mark
        """
        ABSOLUTE_MIN = 0.50   # must exceed this to be considered filled at all
        RELATIVE_THR = 0.60   # must be ≥ 60% of max to co-count

        if not fills:
            return [], 0.0

        max_fill = max(fills.values())
        if max_fill < ABSOLUTE_MIN:
            return [], max_fill  # unattempted

        sorted_fills = sorted(fills.items(), key=lambda x: x[1], reverse=True)
        top_opt, top_val = sorted_fills[0]
        second_val = sorted_fills[1][1] if len(sorted_fills) > 1 else 0.0

        # Clear single answer: top is 2.5× second
        if top_val / (second_val + 1e-6) >= 2.5:
            confidence = top_val / (second_val + 1e-6) / 10.0
            return [top_opt], min(1.0, confidence)

        # Multiple above threshold (multi-mark)
        above = [o for o, f in fills.items() if f >= ABSOLUTE_MIN and f >= top_val * RELATIVE_THR]
        if len(above) > 1:
            return above, 0.5  # multi-mark, lower confidence

        # Fallback: just the top one
        return [top_opt], top_val

    # ── Main Grade ─────────────────────────────────────────────────────────────
    def grade(
        self,
        img: np.ndarray,
        answer_key: dict,
        pos: float = 3.0,
        neg: float = 1.0,
    ) -> Tuple[OMRResult, np.ndarray, np.ndarray]:
        """
        Returns (OMRResult, warped_img, annotated_img)
        """
        self.log_lines = []

        # 1. Flatten & perspective correct
        warped, warp_quality, warp_err, sx, sy = self.auto_warp(img)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # 2. Annotated image base
        annotated = warped.copy()

        # Draw faint column guides
        for col_idx in range(3):
            for sx_val in [SQ_L[col_idx], SQ_R[col_idx]]:
                cv2.line(annotated, (sx_val, ROW_Y_START - 20), (sx_val, ROW_LAST_Y + 20),
                         (50, 45, 75), 1)

        STATUS_BGR = {
            'correct':     (50,  210, 50),
            'wrong':       (50,  50,  220),
            'multi':       (200, 50,  220),
            'unattempted': (100, 100, 110),
        }

        results: List[BubbleResult] = []

        for col_idx in range(3):
            for row_idx in range(20):
                q_num = col_idx * 20 + row_idx + 1
                cy = int(round(ROW_Y_START + row_idx * ROW_SPACING))
                key = answer_key.get(q_num, '')

                # Measure all 4 bubbles
                fills = {opt: self._measure_fill(gray, BUBBLE_CX[col_idx][opt], cy)
                         for opt in 'ABCD'}

                selected, confidence = self._classify(fills)

                # Grade
                if not selected:
                    status, score = 'unattempted', 0.0
                elif len(selected) > 1:
                    status = 'multi'
                    score = -neg if key else 0.0
                elif key and selected[0] == key:
                    status, score = 'correct', float(pos)
                elif key:
                    status, score = 'wrong', float(-neg)
                else:
                    status, score = 'unattempted', 0.0

                results.append(BubbleResult(
                    q_num=q_num, detected=selected, answer_key=key,
                    status=status, score=score, fill_values=fills,
                    confidence=confidence
                ))

                # ── Annotate ────────────────────────────────────────────────
                clr = STATUS_BGR.get(status, (120, 120, 120))

                for opt in 'ABCD':
                    cx = BUBBLE_CX[col_idx][opt]
                    fv = fills[opt]

                    # Detection ring (always)
                    cv2.circle(annotated, (cx, cy), BUBBLE_R + 4, (40, 38, 65), 1)

                    if opt in selected:
                        # Filled: solid colored circle
                        cv2.circle(annotated, (cx, cy), BUBBLE_R, clr, -1)
                        cv2.circle(annotated, (cx, cy), BUBBLE_R + 1, (255, 255, 255), 1)
                        cv2.putText(annotated, opt,
                                    (cx - 6, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        # Unfilled: light ring
                        cv2.circle(annotated, (cx, cy), BUBBLE_R, (85, 83, 100), 1)
                        if fv > 0.08:
                            # Show fill value for near-threshold bubbles
                            cv2.putText(annotated, f"{fv:.2f}",
                                        (cx - 12, cy + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                                        (160, 130, 50), 1, cv2.LINE_AA)

                    # Correct answer indicator when missed
                    if key == opt and opt not in selected and status not in ('correct', 'unattempted'):
                        cv2.circle(annotated, (cx, cy), BUBBLE_R + 6, (50, 210, 50), 2)

                # Draw anchor squares for this row
                for sq_x in [SQ_L[col_idx], SQ_R[col_idx]]:
                    half = int(10)
                    cv2.rectangle(annotated,
                                  (sq_x - half, cy - half),
                                  (sq_x + half, cy + half),
                                  (170, 90, 30), 1)

                # Q-number
                lx = BUBBLE_CX[col_idx]['A'] - 60
                cv2.putText(annotated, f"Q{q_num:02d}",
                            (lx, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            clr, 1, cv2.LINE_AA)

        # 3. Summarise
        correct  = sum(1 for r in results if r.status == 'correct')
        wrong    = sum(1 for r in results if r.status == 'wrong')
        unat     = sum(1 for r in results if r.status == 'unattempted')
        multi    = sum(1 for r in results if r.status == 'multi')
        ps = correct * pos
        ns = (wrong + multi) * neg
        self._log(
            f"Result: {correct}✓ {wrong}✗ {unat}— {multi}M  "
            f"→  +{ps:.0f} −{ns:.0f} = {ps-ns:.1f}",
            'ok'
        )

        return (
            OMRResult(
                bubbles=results, correct=correct, wrong=wrong,
                unattempted=unat, multi=multi,
                pos_score=ps, neg_score=ns, total_score=ps - ns,
                debug_log=list(self.log_lines),
                warp_quality=warp_quality, warp_error_px=warp_err,
                scale_x=sx, scale_y=sy,
            ),
            warped,
            annotated,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_defaults = {
    'answer_key':    {i: '' for i in range(1, 61)},
    'result':        None,
    'original_img':  None,
    'warped_img':    None,
    'annotated_img': None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

engine = OMREngine()


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <div class="tricolor"><div></div><div></div><div></div></div>
  <h1 class="htitle">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="hsub">Precision OMR Grader v7.0 &nbsp;·&nbsp; Tiranga Yuva Samiti</p>
  <span class="pill pill-g">■ Perspective Auto-Flatten</span>&nbsp;
  <span class="pill pill-o">○ Pixel-Verified Bubble Grid</span>&nbsp;
  <span class="pill pill-b">Relative Darkness Detection</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Marking Scheme")
    pos_mark = st.number_input("✅ Correct (+)", 0.5, 10.0, 3.0, 0.5)
    neg_mark = st.number_input("❌ Wrong (−)",   0.0,  5.0, 1.0, 0.5)
    st.divider()
    show_debug = st.checkbox("Show debug log", False)
    show_fills = st.checkbox("Show raw fill values", False)
    st.divider()

    st.markdown("### 📋 Answer Key")
    bulk = st.text_area(
        "Paste 60 answers (comma-separated)",
        placeholder="B,B,B,C,C,A,A,B,...",
        height=70,
    )
    if st.button("✅ Apply Bulk Key", use_container_width=True):
        parts = [p.strip().upper() for p in bulk.split(',')]
        for i, a in enumerate(parts[:60]):
            if a in list('ABCD') + ['']:
                st.session_state.answer_key[i + 1] = a
        st.success("Key applied!")

    opts = ['', 'A', 'B', 'C', 'D']
    st.caption("Or set individually (Q1–Q60):")
    kc = st.columns(2)
    for q in range(1, 61):
        with kc[0] if q % 2 else kc[1]:
            st.session_state.answer_key[q] = st.selectbox(
                f"Q{q}", opts,
                index=opts.index(st.session_state.answer_key.get(q, '')),
                key=f"k{q}",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  UPLOAD & GRADE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sh"><div class="sdot"></div><h3>Upload OMR Sheet</h3></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="struct-diagram">
  Verified sheet structure (Yuva Gyan Mahotsav 2026):<br>
  &nbsp;&nbsp;<span class="sq">■</span>(L anchor)&nbsp;
  <span class="cir">A○</span>&nbsp;
  <span class="cir">B○</span>&nbsp;
  <span class="cir">C○</span>&nbsp;
  <span class="cir">D○</span>&nbsp;
  <span class="sq">■</span>(R anchor)
  &nbsp;×&nbsp;3 columns&nbsp;×&nbsp;20 rows&nbsp;=&nbsp;<strong>60 questions</strong>
  <br><small style="color:#64748B;">
    Col1: L=135 A=189 B=250 C=310 D=372 R=409 &nbsp;|&nbsp;
    Col2: L=502 A=557 B=618 C=677 D=738 R=776 &nbsp;|&nbsp;
    Col3: L=870 A=924 B=983 C=1044 D=1105 R=1142 &nbsp;|&nbsp;
    Row1 Y=360 · Spacing=54.5px
  </small>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legend">
  <span class="chip ch-g">● Correct</span>
  <span class="chip ch-r">● Wrong</span>
  <span class="chip ch-n">○ Unattempted</span>
  <span class="chip ch-p">● Multi-Mark</span>
  <span class="chip ch-b">◎ Missed Correct</span>
</div>
""", unsafe_allow_html=True)

accepted = ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'] if HAS_PDF else ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
uploaded = st.file_uploader(
    "Drop OMR sheet — " + ("PDF or image" if HAS_PDF else "Image") + " (JPG/PNG/TIFF)",
    type=accepted,
)

if uploaded:
    with st.spinner("Loading image…"):
        fb = uploaded.read()
        if uploaded.type == 'application/pdf':
            img_cv = engine.load_pdf(fb, dpi=150)
        else:
            img_cv = engine.load_img(Image.open(io.BytesIO(fb)))
        st.session_state.original_img = img_cv.copy()
    st.success(f"✅ **{uploaded.name}** — {img_cv.shape[1]}×{img_cv.shape[0]}px")

    c_orig, c_act = st.columns(2)
    with c_orig:
        st.markdown("**📄 Original Upload**")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

    with c_act:
        st.markdown("**⚙️ Settings**")
        st.info(f"""
**Marking:** +{pos_mark:.1f} correct · −{neg_mark:.1f} wrong/multi  
**Auto-flatten:** Perspective warp via anchor ■ squares  
**Detection:** Relative darkness — 0.5 threshold · 60% dominance  
**Grid:** Pixel-verified on actual YGM 2026 sheet  
        """)

        if st.button("🚀  Grade Now", use_container_width=True):
            bar = st.progress(0, text="Finding anchor squares…")
            time.sleep(0.05)
            bar.progress(20, text="Computing perspective homography…")
            time.sleep(0.05)
            bar.progress(40, text="Warping to canonical frame…")

            result, warped, annotated = engine.grade(
                img_cv,
                st.session_state.answer_key,
                pos=pos_mark,
                neg=neg_mark,
            )

            bar.progress(85, text="Sampling 240 bubble positions…")
            st.session_state.result        = result
            st.session_state.warped_img    = warped
            st.session_state.annotated_img = annotated
            bar.progress(100, text="Done!")
            time.sleep(0.15)
            bar.empty()

            wq = result.warp_quality
            if wq == 'good':
                st.success(f"✅ Graded! Perspective warp: **GOOD** (error {result.warp_error_px:.1f}px)")
            elif wq == 'approx':
                st.warning(f"⚠️ Graded — perspective approx (error {result.warp_error_px:.1f}px)")
            else:
                st.error("⚠️ Could not find anchor squares — used simple resize. Verify results.")


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.result is not None:
    result = st.session_state.result
    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Results</h3></div>',
                unsafe_allow_html=True)

    # Warp quality banner
    wq   = result.warp_quality
    wcls = {'good': 'wok', 'approx': 'wwarn', 'none': 'werr'}.get(wq, '')
    wico = {'good': '✅', 'approx': '⚠️', 'none': '❌'}.get(wq, '?')
    wdsc = {
        'good':   f"Perspective warp: GOOD — reprojection error {result.warp_error_px:.1f}px",
        'approx': f"Perspective warp: APPROXIMATE — error {result.warp_error_px:.1f}px",
        'none':   "Perspective warp: NOT APPLIED — results may be less accurate",
    }.get(wq, '')
    st.markdown(f'<div class="wstat {wcls}">{wico}  {wdsc}</div>', unsafe_allow_html=True)

    # Score banner
    max_s = 60 * pos_mark
    pct   = result.total_score / max_s * 100 if max_s else 0
    if pct >= 75:   bcls, bico, btxt = "rb-ex", "🏆", "Outstanding!"
    elif pct >= 50: bcls, bico, btxt = "rb-gd", "👍", "Good Performance"
    elif pct >= 35: bcls, bico, btxt = "rb-av", "📚", "Average — Keep Practicing"
    else:           bcls, bico, btxt = "rb-pr", "⚠️", "Needs Improvement"
    bclr = "#22C55E" if pct >= 75 else ("#F59E0B" if pct >= 50 else ("#F97316" if pct >= 35 else "#EF4444"))

    st.markdown(f"""
    <div class="rbanner {bcls}">
      <span style="font-size:2rem;">{bico}</span>
      <div>
        <div>{btxt}</div>
        <div style="font-size:.83rem;font-weight:400;opacity:.8;margin-top:3px;">
          Score: <strong>{result.total_score:.1f}</strong> / {max_s:.0f} &nbsp;·&nbsp; {pct:.1f}%
        </div>
      </div>
    </div>
    <div class="strack">
      <div class="sfill" style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{bclr},{bclr}88);"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sgrid">
      <div class="sc c"><div class="snum cg">{result.correct}</div><div class="slbl">Correct</div><div class="gl"></div></div>
      <div class="sc w"><div class="snum cr">{result.wrong}</div><div class="slbl">Wrong</div><div class="gl"></div></div>
      <div class="sc s"><div class="snum ca">{result.unattempted}</div><div class="slbl">Skipped</div><div class="gl"></div></div>
      <div class="sc m"><div class="snum cp">{result.multi}</div><div class="slbl">Multi-Mark</div><div class="gl"></div></div>
      <div class="sc total"><div class="snum co">{result.total_score:.1f}</div><div class="slbl">Net Score</div><div class="gl"></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("**📉 Breakdown**")
        att = result.correct + result.wrong + result.multi
        acc = result.correct / att * 100 if att else 0
        m1, m2 = st.columns(2)
        m1.metric("Positive Score", f"+{result.pos_score:.1f}")
        m2.metric("Negative Score", f"−{result.neg_score:.1f}")
        m3, m4 = st.columns(2)
        m3.metric("Net Score", f"{result.total_score:.1f}")
        m4.metric("Accuracy", f"{acc:.1f}%")

        wclr = '#22C55E' if wq == 'good' else '#F59E0B' if wq == 'approx' else '#EF4444'
        rows_info = ''.join(
            f'<div style="display:flex;justify-content:space-between;font-size:.85rem;padding:3px 0;">'
            f'<span>{label}</span><span style="font-weight:700;{col}">{val}</span></div>'
            for label, val, col in [
                ("Attempted",    f"{att}/60 ({att/60*100:.0f}%)", ""),
                ("Unattempted",  str(result.unattempted), ""),
                ("Multi-marked", str(result.multi), ""),
                ("Warp quality", wq.upper(), f"color:{wclr};"),
            ]
        )
        st.markdown(f"""
        <div style="background:var(--sf2);border:1px solid var(--br);border-radius:10px;
             padding:13px 16px;margin-top:10px;">
          <div style="font-size:.68rem;color:var(--mu);text-transform:uppercase;
               letter-spacing:1.5px;margin-bottom:8px;">Summary</div>
          {rows_info}
        </div>
        """, unsafe_allow_html=True)

        if show_debug and result.debug_log:
            st.markdown("**🔍 Debug Log**")
            html = "".join(
                f'<div class="{"ok" if "[OK]" in l else "warn" if "[WARN]" in l else "err" if "[ERR]" in l else "info"}">{l}</div>'
                for l in result.debug_log
            )
            st.markdown(f'<div class="dlog">{html}</div>', unsafe_allow_html=True)

    with cb:
        t1, t2 = st.tabs(["🎯 Annotated OMR", "🔲 Warped Canonical"])
        with t1:
            if st.session_state.annotated_img is not None:
                ann = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
                st.image(ann, use_container_width=True)
                buf = io.BytesIO()
                Image.fromarray(ann).save(buf, 'PNG')
                st.download_button("⬇️ Download Annotated", buf.getvalue(),
                                   "annotated_omr.png", "image/png", use_container_width=True)
        with t2:
            if st.session_state.warped_img is not None:
                w2 = cv2.cvtColor(st.session_state.warped_img, cv2.COLOR_BGR2RGB)
                st.image(w2, use_container_width=True)
                buf2 = io.BytesIO()
                Image.fromarray(w2).save(buf2, 'PNG')
                st.download_button("⬇️ Download Warped", buf2.getvalue(),
                                   "warped_canonical.png", "image/png", use_container_width=True)

    # ── Question-wise Table ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Question-wise Report</h3></div>',
                unsafe_allow_html=True)

    filt = st.multiselect(
        "Filter by status",
        ['correct', 'wrong', 'unattempted', 'multi'],
        default=['correct', 'wrong', 'unattempted', 'multi'],
    )

    filtered = [b for b in result.bubbles if b.status in filt]
    rows_html = ""
    for b in filtered:
        det  = ', '.join(b.detected) if b.detected else '—'
        key  = b.answer_key or '—'
        sc   = f"+{b.score:.0f}" if b.score > 0 else (f"{b.score:.0f}" if b.score < 0 else "0")
        sclr = "cg" if b.score > 0 else ("cr" if b.score < 0 else "ca")
        bcls = {'correct': 'bs-c', 'wrong': 'bs-w', 'unattempted': 'bs-s', 'multi': 'bs-m'}.get(b.status, '')
        bico = {'correct': '✓', 'wrong': '✗', 'unattempted': '—', 'multi': '×'}.get(b.status, '')
        conf_html = f'<span style="color:#64748B;font-size:.72rem;">{b.confidence:.2f}</span>'
        fv_td = ""
        if show_fills and b.fill_values:
            fv = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(b.fill_values.items()))
            fv_td = f"<td style='font-size:.69rem;color:var(--mu);font-family:JetBrains Mono,monospace;white-space:nowrap;'>{fv}</td>"
        rows_html += f"""<tr>
          <td>Q{b.q_num:02d}</td>
          <td><span style="color:#60A5FA;font-weight:700;font-family:'JetBrains Mono',monospace;">{det}</span></td>
          <td><span style="color:#34D399;font-weight:700;font-family:'JetBrains Mono',monospace;">{key}</span></td>
          <td><span class="bs {bcls}">{bico} {b.status.upper()}</span></td>
          <td class="{sclr}" style="font-weight:800;font-family:'JetBrains Mono',monospace;">{sc}</td>
          <td>{conf_html}</td>
          {fv_td}
        </tr>"""

    xth = "<th>Fills A/B/C/D</th>" if show_fills else ""
    st.markdown(f"""
    <div style="max-height:540px;overflow-y:auto;border:1px solid var(--br);
         border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
    <table class="bt">
      <thead><tr>
        <th>Q#</th><th>Detected</th><th>Key</th><th>Status</th>
        <th>Score</th><th>Conf</th>{xth}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    st.write("")
    export_rows = [
        {
            'Q':        b.q_num,
            'Detected': ','.join(b.detected) if b.detected else '',
            'Key':      b.answer_key,
            'Status':   b.status,
            'Score':    b.score,
            'Conf':     round(b.confidence, 3),
            **{f'Fill_{k}': round(v, 4) for k, v in b.fill_values.items()},
        }
        for b in result.bubbles
    ]
    st.download_button(
        "⬇️ Download Full CSV",
        pd.DataFrame(export_rows).to_csv(index=False),
        "omr_results.csv",
        "text/csv",
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:var(--mu);font-size:.77rem;">
  <div class="tricolor" style="max-width:130px;margin:0 auto 10px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · OMR Grader v7.0<br>
  <span style="font-size:.67rem;opacity:.5;">
    Structure: ■ A○ B○ C○ D○ ■ · Pixel-verified grid · Relative darkness detection
  </span>
</div>
""", unsafe_allow_html=True)
