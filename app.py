"""
YUVA GYAN MAHOTSAV 2026 — PRECISION OMR GRADER v6.0
=====================================================
Structure per row (VERIFIED from actual sheet pixel analysis):

  ■(L)  A○  ■(mid)  B○  C○  D○  ■(R)

Per column (col-spacing = 735px, canonical 300 DPI = 2479×3508):
  Col 1  (Q01-20): ■L=270  A=379  ■mid=454  B=502  C=620  D=744  ■R=817
  Col 2  (Q21-40): ■L=1005 A=1114 ■mid=1189 B=1237 C=1355 D=1479 ■R=1552
  Col 3  (Q41-60): ■L=1740 A=1849 ■mid=1924 B=1972 C=2090 D=2214 ■R=2287

Row 1 Y = 720,  Row spacing = 109 px,  Bubble radius = 23 px

AUTO-CROP:
  Finds the 9 per-row solid ■ squares → picks 4 corner L/R anchors
  → computes perspective homography → warps to canonical frame
  → samples all 240 bubble positions at exact hardcoded coordinates
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

/* Structure diagram */
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
#  CANONICAL GRID (300 DPI — verified pixel-by-pixel on official blank sheet)
#
#  Per-row structure: ■(L)  A○  ■(mid)  B○  C○  D○  ■(R)
#  Col spacing = 735 px
# ═══════════════════════════════════════════════════════════════════════════════
COL_SPACING   = 735
COL_OFFSETS   = [0, COL_SPACING, COL_SPACING * 2]   # 0, 735, 1470

# Bubble CX per option (col0 values; add COL_OFFSETS[col_idx] for col 1/2)
_BASE_CX = {'A': 379, 'B': 502, 'C': 620, 'D': 744}
BUBBLE_CX = {opt: [_BASE_CX[opt] + off for off in COL_OFFSETS] for opt in 'ABCD'}

# Square anchor CX (L, mid, R) per col — used for homography
_BASE_SQ = {'L': 270, 'mid': 454, 'R': 817}
SQ_CX    = {k: [_BASE_SQ[k] + off for off in COL_OFFSETS] for k in _BASE_SQ}

ROW_Y_START  = 720
ROW_SPACING  = 109
BUBBLE_R     = 23
INNER_R      = 17       # Inner measurement radius (avoids circle-outline pixels)
CANONICAL_W  = 2479
CANONICAL_H  = 3508

# 4-corner warp reference: TL / TR / BL / BR using the ■L (col0) and ■R (col2)
# Row 1 Y = 720,  Row 20 Y = 720 + 19×109 = 2791
ROW_LAST_Y   = ROW_Y_START + 19 * ROW_SPACING   # 2791
WARP_DST = np.float32([
    [SQ_CX['L'][0], ROW_Y_START],   # TL
    [SQ_CX['R'][2], ROW_Y_START],   # TR
    [SQ_CX['L'][0], ROW_LAST_Y ],   # BL
    [SQ_CX['R'][2], ROW_LAST_Y ],   # BR
])  # = [(270,720), (2287,720), (270,2791), (2287,2791)]


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
    warp_quality: str = "none"
    warp_error_px: float = 999.0


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
    def load_pdf(self, pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        self._log(f"PDF → {img.shape[1]}×{img.shape[0]}px at {dpi} DPI", 'ok')
        return img

    def load_img(self, pil: Image.Image) -> np.ndarray:
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._log(f"Image → {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    # ── AUTO-CROP: Find ■ Squares & Compute Homography ───────────────────────
    def _find_solid_squares(self, gray: np.ndarray) -> List[dict]:
        """Detect all solid black filled ■ squares on the sheet."""
        H, W = gray.shape
        scale = W / CANONICAL_W
        _, bw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mn = int(150 * scale * scale)
        mx = int(6000 * scale * scale)
        result = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (mn < area < mx):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < int(6*scale) or h < int(6*scale):
                continue
            if not (0.4 < w/float(h) < 2.5):
                continue
            # Must be in answer-grid vertical band (skip header/footer)
            if y < H * 0.10 or (y + h) > H * 0.96:
                continue
            roi = bw[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            fill = np.count_nonzero(roi) / roi.size
            if fill < 0.62:
                continue
            result.append({'cx': x+w//2, 'cy': y+h//2,
                            'x': x, 'y': y, 'w': w, 'h': h, 'area': area})
        self._log(f"Solid squares detected: {len(result)}", 'info')
        return result

    def _cluster_rows(self, squares: List[dict], scale: float) -> List[List[dict]]:
        """Cluster squares into horizontal rows."""
        if not squares:
            return []
        squares = sorted(squares, key=lambda s: s['cy'])
        gap = max(15, int(14 * scale))
        rows, cur = [], [squares[0]]
        for s in squares[1:]:
            if abs(s['cy'] - cur[-1]['cy']) < gap:
                cur.append(s)
            else:
                rows.append(cur)
                cur = [s]
        rows.append(cur)
        return rows

    def auto_crop(self, img: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Detect the 4 corner ■ anchors → compute perspective homography
        → warp image to canonical 2479×3508.
        Returns (warped, quality, reprojection_error_px).
        """
        H, W = img.shape[:2]
        scale = W / CANONICAL_W
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        squares = self._find_solid_squares(gray)
        rows    = self._cluster_rows(squares, scale)

        # Keep rows that look like answer rows: 6–12 squares wide
        ans_rows = [r for r in rows if 6 <= len(r) <= 13]
        self._log(f"Answer rows: {len(ans_rows)}", 'info')

        if len(ans_rows) < 4:
            self._log("Too few answer rows — simple scale warp", 'warn')
            return self._scale_warp(img), 'none', 999.0

        # Sort answer rows by Y, take first and last
        ans_rows.sort(key=lambda r: np.mean([s['cy'] for s in r]))
        first_row = sorted(ans_rows[0],  key=lambda s: s['cx'])
        last_row  = sorted(ans_rows[-1], key=lambda s: s['cx'])

        # TL=leftmost of first row, TR=rightmost of first row
        # BL=leftmost of last row,  BR=rightmost of last row
        tl = first_row[0];  tr = first_row[-1]
        bl = last_row[0];   br = last_row[-1]

        src = np.float32([[tl['cx'], tl['cy']], [tr['cx'], tr['cy']],
                          [bl['cx'], bl['cy']], [br['cx'], br['cy']]])
        dst = WARP_DST.copy()

        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is None:
            self._log("Homography failed — scale warp", 'warn')
            return self._scale_warp(img), 'approx', 99.0

        # Reprojection error
        pts = src.reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
        quality = 'good' if err < 8 else 'approx'
        self._log(f"Homography OK: reprojection error = {err:.1f}px → {quality}", 'ok')

        warped = cv2.warpPerspective(img, M, (CANONICAL_W, CANONICAL_H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return warped, quality, err

    def _scale_warp(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (CANONICAL_W, CANONICAL_H), interpolation=cv2.INTER_LINEAR)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def _preprocess(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        adap = cv2.adaptiveThreshold(blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)
        _, otsu = cv2.threshold(blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return adap, otsu

    # ── Fill Measurement ──────────────────────────────────────────────────────
    def _fill(self, gray: np.ndarray, adap: np.ndarray, otsu: np.ndarray,
              cx: int, cy: int) -> float:
        """
        Measure how filled a bubble at (cx, cy) is, using three methods:
          1. Adaptive-threshold pixel ratio inside INNER_R circle
          2. OTSU-threshold pixel ratio
          3. Mean darkness vs sheet background
        Returns a combined score in [0, 1].
        """
        r = INNER_R
        H, W = gray.shape
        x1, y1 = max(0, cx-r), max(0, cy-r)
        x2, y2 = min(W, cx+r), min(H, cy+r)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        h_r, w_r = y2-y1, x2-x1
        mask = np.zeros((h_r, w_r), dtype=np.uint8)
        cv2.circle(mask, (cx-x1, cy-y1), r-1, 255, -1)
        total = int(np.count_nonzero(mask))
        if total == 0:
            return 0.0

        def ratio(thresh_img):
            roi = thresh_img[y1:y2, x1:x2]
            return int(np.count_nonzero(roi & (mask > 0))) / total

        ra = ratio(adap)
        ro = ratio(otsu)

        vals = gray[y1:y2, x1:x2][mask > 0]
        bg   = float(np.percentile(gray, 90))
        dark = max(0.0, (bg - float(np.mean(vals))) / max(bg, 1.0)) if len(vals) else 0.0

        return min(1.0, ra * 0.40 + ro * 0.30 + dark * 0.30)

    # ── Selection Classifier ──────────────────────────────────────────────────
    def _classify(self, fills: Dict[str, float], thresh: float) -> List[str]:
        if not fills:
            return []
        maxf = max(fills.values())
        if maxf < thresh * 0.35:
            return []

        srt = sorted(fills.items(), key=lambda x: x[1], reverse=True)

        # Dominance check: one bubble clearly above others
        if len(srt) >= 2 and srt[0][1] >= thresh * 0.55:
            if srt[0][1] / (srt[1][1] + 1e-6) >= 1.75:
                return [srt[0][0]]

        above = [opt for opt, f in fills.items() if f >= thresh]
        if above:
            return above

        # Soft fallback
        if srt[0][1] >= thresh * 0.50:
            return [srt[0][0]]
        return []

    # ── Main Grade ─────────────────────────────────────────────────────────────
    def grade(self, img: np.ndarray, answer_key: dict,
              pos: float = 3.0, neg: float = 1.0,
              fill_thresh: float = 0.20) -> Tuple[OMRResult, np.ndarray, np.ndarray]:
        """
        Returns (OMRResult, warped_img, annotated_img)
        """
        self.log_lines = []
        self._log(f"Input: {img.shape[1]}×{img.shape[0]}px", 'info')

        # 1. Auto-crop → canonical frame
        warped, wq, werr = self.auto_crop(img)
        gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        adap, otsu = self._preprocess(gray)

        # 2. Sample all 60 × 4 bubbles
        annotated = warped.copy()
        results   = []

        # Draw grid reference lines (subtle)
        for col_idx in range(3):
            for sq_type in ['L', 'R']:
                sx = SQ_CX[sq_type][col_idx]
                cv2.line(annotated, (sx, ROW_Y_START - 15), (sx, ROW_LAST_Y + 15),
                         (60, 55, 80), 1)

        STATUS_BGR = {
            'correct':     (50,  210, 50),
            'wrong':       (50,  50,  230),
            'multi':       (200, 50,  220),
            'unattempted': (90,  90,  100),
        }

        for col_idx in range(3):
            for row_idx in range(20):
                q_num = col_idx * 20 + row_idx + 1
                cy    = ROW_Y_START + row_idx * ROW_SPACING
                key   = answer_key.get(q_num, '')

                # Measure fills
                fills = {opt: self._fill(gray, adap, otsu,
                                         BUBBLE_CX[opt][col_idx], cy)
                         for opt in 'ABCD'}

                selected = self._classify(fills, fill_thresh)

                # Grade
                if not selected:
                    status, score = 'unattempted', 0.0
                elif len(selected) > 1:
                    status = 'multi'
                    score  = -neg if key else 0.0
                elif key and selected[0] == key:
                    status, score = 'correct', float(pos)
                elif key:
                    status, score = 'wrong', float(-neg)
                else:
                    status, score = 'unattempted', 0.0

                results.append(BubbleResult(
                    q_num=q_num, detected=selected, answer_key=key,
                    status=status, score=score, fill_values=fills))

                # ── Annotate ─────────────────────────────────────────────────
                clr = STATUS_BGR.get(status, (120, 120, 120))

                for opt in 'ABCD':
                    cx = BUBBLE_CX[opt][col_idx]
                    fv = fills[opt]

                    # Outer detection ring (always drawn)
                    cv2.circle(annotated, (cx, cy), BUBBLE_R + 4, (45, 42, 68), 1)

                    if opt in selected:
                        # Filled bubble
                        cv2.circle(annotated, (cx, cy), BUBBLE_R, clr, -1)
                        cv2.circle(annotated, (cx, cy), BUBBLE_R + 1, (255, 255, 255), 1)
                        cv2.putText(annotated, opt,
                                    (cx - 6, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        # Unfilled — light outline
                        cv2.circle(annotated, (cx, cy), BUBBLE_R, (90, 88, 105), 1)
                        if fv > 0.05:
                            cv2.putText(annotated, f"{fv:.2f}",
                                        (cx - 11, cy + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                                        (170, 140, 60), 1, cv2.LINE_AA)

                    # Correct-answer indicator (green circle) when answer was missed
                    if key == opt and opt not in selected and status not in ('correct', 'unattempted'):
                        cv2.circle(annotated, (cx, cy), BUBBLE_R + 6, (50, 210, 50), 2)

                # Draw the ■ L and ■ R squares for this row
                for sq_type, sq_clr in [('L', (180, 100, 40)), ('R', (180, 100, 40))]:
                    sx = SQ_CX[sq_type][col_idx]
                    half = 11
                    cv2.rectangle(annotated,
                                  (sx - half, cy - half),
                                  (sx + half, cy + half),
                                  sq_clr, 1)

                # Q-number label
                lx = BUBBLE_CX['A'][col_idx] - 60
                cv2.putText(annotated, f"Q{q_num:02d}",
                            (lx, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                            clr, 1, cv2.LINE_AA)

        # 3. Summarise
        correct = sum(1 for r in results if r.status == 'correct')
        wrong   = sum(1 for r in results if r.status == 'wrong')
        unat    = sum(1 for r in results if r.status == 'unattempted')
        multi   = sum(1 for r in results if r.status == 'multi')
        ps = correct * pos
        ns = (wrong + multi) * neg
        self._log(f"Final: {correct}✓ {wrong}✗ {unat}— {multi}M  →  score={ps-ns:.1f}", 'ok')

        return (
            OMRResult(bubbles=results, correct=correct, wrong=wrong,
                      unattempted=unat, multi=multi,
                      pos_score=ps, neg_score=ns, total_score=ps-ns,
                      debug_log=list(self.log_lines),
                      warp_quality=wq, warp_error_px=werr),
            warped,
            annotated
        )


# ─── Session State ─────────────────────────────────────────────────────────────
_defaults = {
    'answer_key': {i: random.choice(list('ABCD')) for i in range(1, 61)},
    'result':       None,
    'original_img': None,
    'warped_img':   None,
    'annotated_img':None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

engine = OMREngine()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="tricolor"><div></div><div></div><div></div></div>
  <h1 class="htitle">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="hsub">Precision OMR Grader v6.0 &nbsp;·&nbsp; Tiranga Yuva Samiti</p>
  <span class="pill pill-g">■ Auto-Crop via Homography</span>&nbsp;
  <span class="pill pill-o">○ Verified Exact Grid</span>&nbsp;
  <span class="pill pill-b">3-Method Fill Scoring</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    pos_mark = st.number_input("✅ Correct (+)", 0.5, 10.0, 3.0, 0.5)
    neg_mark = st.number_input("❌ Wrong (−)",   0.0,  5.0, 1.0, 0.5)
    st.divider()
    st.markdown("**🎯 Detection**")
    fill_thr = st.slider("Fill Threshold", 0.06, 0.55, 0.20, 0.01,
        help="How dark a bubble must be to count as filled. 0.15–0.25 works for most pens/pencils.")
    st.caption(f"Current: **{fill_thr:.2f}**  —  try lower (0.12) for light pencil")
    show_debug = st.checkbox("Show debug log", False)
    show_fills = st.checkbox("Show raw fill values", False)
    st.divider()
    st.markdown("### 📋 Answer Key")
    bulk = st.text_area("Paste 60 answers (comma-sep)", placeholder="A,B,C,D,...", height=68)
    if st.button("Apply Bulk Key", use_container_width=True):
        parts = [p.strip().upper() for p in bulk.split(',')]
        for i, a in enumerate(parts[:60]):
            if a in list('ABCD') + ['']:
                st.session_state.answer_key[i+1] = a
        st.success("✅ Applied!")
    opts = ['', 'A', 'B', 'C', 'D']
    st.caption("Or set individually:")
    kc = st.columns(2)
    for q in range(1, 61):
        with kc[0] if q % 2 else kc[1]:
            st.session_state.answer_key[q] = st.selectbox(
                f"Q{q}", opts,
                index=opts.index(st.session_state.answer_key.get(q, '')),
                key=f"k{q}")

# ─── UPLOAD ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sh"><div class="sdot"></div><h3>Upload OMR Sheet</h3></div>',
            unsafe_allow_html=True)

# Sheet structure diagram
st.markdown("""
<div class="struct-diagram">
  Per-row structure confirmed from official sheet:<br>
  <span class="sq">■</span>(L anchor)&nbsp;&nbsp;
  <span class="cir">A○</span>&nbsp;&nbsp;
  <span class="msq">■</span>(mid)&nbsp;&nbsp;
  <span class="cir">B○</span>&nbsp;&nbsp;
  <span class="cir">C○</span>&nbsp;&nbsp;
  <span class="cir">D○</span>&nbsp;&nbsp;
  <span class="sq">■</span>(R anchor)
  &nbsp;&nbsp;&nbsp;→&nbsp; 3 columns × 20 rows = 60 questions
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legend">
  <span class="chip ch-g">● Correct</span>
  <span class="chip ch-r">● Wrong</span>
  <span class="chip ch-n">○ Unattempted</span>
  <span class="chip ch-a">⚑ Skipped</span>
  <span class="chip ch-p">● Multi-Mark</span>
  <span class="chip ch-b">◎ Missed correct answer</span>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop OMR sheet — PDF or image (JPG/PNG/TIFF)",
    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'])

if uploaded:
    with st.spinner("Loading…"):
        fb = uploaded.read()
        if uploaded.type == 'application/pdf':
            img_cv = engine.load_pdf(fb, dpi=300)
        else:
            img_cv = engine.load_img(Image.open(io.BytesIO(fb)))
        st.session_state.original_img = img_cv.copy()
    st.success(f"✅ **{uploaded.name}** — {img_cv.shape[1]}×{img_cv.shape[0]}px")

    c_orig, c_act = st.columns(2)
    with c_orig:
        st.markdown("**📄 Original Upload**")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c_act:
        st.markdown("**⚙️ Grade Settings**")
        st.info(f"""
**Marking:** +{pos_mark:.1f} correct · −{neg_mark:.1f} wrong  
**Fill Threshold:** {fill_thr:.2f}  
**Auto-Crop:** Corner ■ anchor homography  
**Grid:** Hardcoded exact bubble positions  
**Structure:** ■ A○ ■mid B○ C○ D○ ■ per row
        """)
        if st.button("🚀  Grade Now", use_container_width=True):
            bar = st.progress(0, text="Detecting corner ■ anchors…")
            time.sleep(0.1)
            bar.progress(25, text="Computing perspective warp…")
            time.sleep(0.1)
            bar.progress(45, text="Warping to canonical frame…")

            result, warped, annotated = engine.grade(
                img_cv, st.session_state.answer_key,
                pos=pos_mark, neg=neg_mark, fill_thresh=fill_thr)

            bar.progress(80, text="Sampling 240 bubble positions…")
            st.session_state.result      = result
            st.session_state.warped_img  = warped
            st.session_state.annotated_img = annotated
            bar.progress(100, text="Done!")
            time.sleep(0.2); bar.empty()

            wq = result.warp_quality
            if wq == 'good':
                st.success(f"✅ Graded! Warp: **GOOD** (error: {result.warp_error_px:.1f}px)")
            elif wq == 'approx':
                st.warning(f"⚠️ Graded with approximate alignment (error: {result.warp_error_px:.1f}px)")
            else:
                st.error("⚠️ Corner anchors not found — used simple scale. Check image quality.")

# ─── RESULTS ──────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    result = st.session_state.result
    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Results</h3></div>',
                unsafe_allow_html=True)

    # Warp status bar
    wq   = result.warp_quality
    wcls = {'good': 'wok', 'approx': 'wwarn', 'none': 'werr'}.get(wq, '')
    wico = {'good': '✅', 'approx': '⚠️', 'none': '❌'}.get(wq, '?')
    wdsc = {
        'good':   f"Perspective warp: GOOD  (reprojection error {result.warp_error_px:.1f}px)",
        'approx': f"Perspective warp: APPROXIMATE  (error {result.warp_error_px:.1f}px — not all corners found)",
        'none':   "Perspective warp: NOT APPLIED — used simple scale (low confidence, verify results)",
    }.get(wq, '')
    st.markdown(f'<div class="wstat {wcls}">{wico}  {wdsc}</div>', unsafe_allow_html=True)

    max_s = 60 * pos_mark
    pct   = result.total_score / max_s * 100 if max_s else 0
    if pct >= 75:   bcls, bico, btxt = "rb-ex", "🏆", "Outstanding!"
    elif pct >= 50: bcls, bico, btxt = "rb-gd", "👍", "Good Performance"
    elif pct >= 35: bcls, bico, btxt = "rb-av", "📚", "Average — Keep Practicing"
    else:            bcls, bico, btxt = "rb-pr", "⚠️", "Needs Improvement"
    bclr = "#22C55E" if pct>=75 else ("#F59E0B" if pct>=50 else ("#F97316" if pct>=35 else "#EF4444"))

    st.markdown(f"""
    <div class="rbanner {bcls}">
      <span style="font-size:2rem;">{bico}</span>
      <div>
        <div>{btxt}</div>
        <div style="font-size:.83rem;font-weight:400;opacity:.8;margin-top:3px;">
          Score: <strong>{result.total_score:.1f}</strong>/{max_s:.0f} &nbsp;·&nbsp; {pct:.1f}%
        </div>
      </div>
    </div>
    <div class="strack"><div class="sfill"
      style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{bclr},{bclr}88);"></div></div>
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
        wclr = '#22C55E' if wq=='good' else '#F59E0B' if wq=='approx' else '#EF4444'
        st.markdown(f"""
        <div style="background:var(--sf2);border:1px solid var(--br);border-radius:10px;padding:13px 16px;margin-top:10px;">
          <div style="font-size:.68rem;color:var(--mu);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Summary</div>
          {''.join(f'<div style="display:flex;justify-content:space-between;font-size:.85rem;padding:3px 0;"><span>{label}</span><span style="font-weight:700;{col}">{val}</span></div>'
            for label,val,col in [
              ("Attempted", f"{att}/60 ({att/60*100:.0f}%)", ""),
              ("Unattempted", str(result.unattempted), ""),
              ("Multi-marked", str(result.multi), ""),
              ("Warp quality", wq.upper(), f"color:{wclr};"),
            ])}
        </div>
        """, unsafe_allow_html=True)

        if show_debug and result.debug_log:
            st.markdown("**🔍 Debug Log**")
            html = "".join(
                f'<div class="{"ok" if "[OK]" in l else "warn" if "[WARN]" in l else "err" if "[ERR]" in l else "info"}">{l}</div>'
                for l in result.debug_log)
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

    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Question-wise Report</h3></div>',
                unsafe_allow_html=True)

    f1, f2 = st.columns([2, 1])
    with f1:
        filt = st.multiselect("Filter by status",
            ['correct', 'wrong', 'unattempted', 'multi'],
            default=['correct', 'wrong', 'unattempted', 'multi'])

    filtered = [b for b in result.bubbles if b.status in filt]
    rows_html = ""
    for b in filtered:
        det  = ', '.join(b.detected) if b.detected else '—'
        key  = b.answer_key or '—'
        sc   = f"+{b.score:.0f}" if b.score > 0 else (f"{b.score:.0f}" if b.score else "0")
        sclr = "cg" if b.score > 0 else ("cr" if b.score < 0 else "ca")
        bcls = {'correct':'bs-c','wrong':'bs-w','unattempted':'bs-s','multi':'bs-m'}.get(b.status,'')
        bico = {'correct':'✓','wrong':'✗','unattempted':'—','multi':'×'}.get(b.status,'')
        fv_td = ""
        if show_fills and b.fill_values:
            fv = "  ".join(f"{k}:{v:.3f}" for k, v in sorted(b.fill_values.items()))
            fv_td = f"<td style='font-size:.69rem;color:var(--mu);font-family:JetBrains Mono,monospace;white-space:nowrap;'>{fv}</td>"
        rows_html += f"""<tr>
          <td>Q{b.q_num:02d}</td>
          <td><span style="color:#60A5FA;font-weight:700;font-family:'JetBrains Mono',monospace;">{det}</span></td>
          <td><span style="color:#34D399;font-weight:700;font-family:'JetBrains Mono',monospace;">{key}</span></td>
          <td><span class="bs {bcls}">{bico} {b.status.upper()}</span></td>
          <td class="{sclr}" style="font-weight:800;font-family:'JetBrains Mono',monospace;">{sc}</td>
          {fv_td}
        </tr>"""

    xth = "<th>Fills A/B/C/D</th>" if show_fills else ""
    st.markdown(f"""
    <div style="max-height:520px;overflow-y:auto;border:1px solid var(--br);
         border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
    <table class="bt">
      <thead><tr><th>Q#</th><th>Detected</th><th>Key</th><th>Status</th><th>Score</th>{xth}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    st.write("")
    exp = [{'Q': b.q_num, 'Detected': ','.join(b.detected) if b.detected else '',
             'Key': b.answer_key, 'Status': b.status, 'Score': b.score,
             **{f'Fill_{k}': round(v, 4) for k, v in b.fill_values.items()}}
            for b in result.bubbles]
    st.download_button("⬇️ Download Full CSV",
                        pd.DataFrame(exp).to_csv(index=False),
                        "omr_results.csv", "text/csv")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:var(--mu);font-size:.77rem;">
  <div class="tricolor" style="max-width:130px;margin:0 auto 10px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · OMR Grader v6.0<br>
  <span style="font-size:.67rem;opacity:.5;">Structure: ■ A○ ■mid B○ C○ D○ ■ · Auto-Crop Homography · Verified Pixel Grid</span>
</div>
""", unsafe_allow_html=True)
