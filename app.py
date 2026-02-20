"""
YUVA GYAN MAHOTSAV 2026 — Ultra Precise OMR Grader v4.0
========================================================
Built from EXACT analysis of the official OMR sheet:

SHEET STRUCTURE per row:
  ■  A○  B○  C○  D○  ■
  └── Left anchor square (solid filled ■)
       └── 4 circles: A, B, C, D
                               └── Right anchor square (solid filled ■)

LAYOUT:
  • 3 columns × 20 questions = 60 total
  • Each row has: left-■ | A○ | B○ | C○ | D○ | right-■
  • The solid black squares are MACHINE-PRINTED timing marks — perfect anchors
  • Bubbles are open circles (○) that candidates fill

DETECTION STRATEGY:
  1. Find ALL solid black squares (high fill, roughly square shape)
  2. Group squares into LEFT anchors and RIGHT anchors per column
  3. For each left anchor row → compute exact A/B/C/D circle positions
     using the distance from left-■ to right-■ to interpolate A/B/C/D
  4. Measure fill inside each circle using adaptive + intensity methods
  5. Classify with relative dominance logic
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import random

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yuva Gyan Mahotsav 2026 – OMR Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --saffron: #F97316; --saffron-d: #EA580C;
    --navy: #0F172A; --navy-m: #1E293B; --navy-l: #334155;
    --green: #22C55E; --green-d: #16A34A;
    --gold: #F59E0B; --white: #F8FAFC;
    --bg: #080D1A; --surface: #0F1923; --surface2: #162032; --surface3: #1D2D45;
    --border: rgba(255,255,255,0.07); --border-s: rgba(249,115,22,0.35);
    --text: #E2E8F0; --muted: #64748B;
    --correct: #22C55E; --wrong: #EF4444; --skip: #F59E0B; --multi: #A855F7;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background: var(--bg); color: var(--text); }
.stApp { background: var(--bg) !important; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1440px; }

/* HEADER */
.omr-header {
    background: linear-gradient(135deg, #0A1628 0%, #0F1F3D 50%, #0A1628 100%);
    border: 1px solid rgba(249,115,22,0.2);
    border-radius: 18px; padding: 28px 36px 24px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.omr-header::before {
    content:''; position:absolute; top:-80px; right:-80px; width:260px; height:260px;
    background: radial-gradient(circle, rgba(249,115,22,0.12) 0%, transparent 65%); border-radius:50%;
}
.omr-header::after {
    content:''; position:absolute; bottom:-50px; left:8%; width:180px; height:180px;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 65%); border-radius:50%;
}
.tricolor { display:flex; height:4px; border-radius:2px; overflow:hidden; margin-bottom:16px; max-width:360px; }
.tricolor div:nth-child(1){flex:1;background:linear-gradient(90deg,#F97316,#EA580C);}
.tricolor div:nth-child(2){flex:1;background:#E2E8F0;}
.tricolor div:nth-child(3){flex:1;background:linear-gradient(90deg,#22C55E,#16A34A);}
.omr-title {
    font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800;
    background: linear-gradient(130deg, #F1F5F9 20%, #F97316 60%, #22C55E 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    line-height:1.1; position:relative; z-index:1;
}
.omr-sub { font-size:0.87rem; color:var(--muted); margin-top:7px; z-index:1; position:relative; }
.tech-pill {
    display:inline-flex; align-items:center; gap:5px;
    padding:3px 10px; border-radius:20px; font-size:0.73rem; font-weight:700;
    background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.25); color:#22C55E;
    font-family:'JetBrains Mono',monospace; margin-top:8px;
}

/* STAT GRID */
.stat-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:18px 0; }
.stat-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:14px; padding:18px 14px; text-align:center;
    position:relative; overflow:hidden; transition:transform .2s;
}
.stat-card:hover { transform:translateY(-2px); }
.stat-card.c { border-color:rgba(34,197,94,.3); }
.stat-card.w { border-color:rgba(239,68,68,.3); }
.stat-card.s { border-color:rgba(245,158,11,.3); }
.stat-card.m { border-color:rgba(168,85,247,.3); }
.stat-card.sc { border-color:rgba(249,115,22,.4); background:linear-gradient(135deg,rgba(249,115,22,.06),var(--surface)); }
.glow { position:absolute; bottom:0; left:0; right:0; height:2px; }
.stat-card.c .glow{background:#22C55E;box-shadow:0 0 8px #22C55E;}
.stat-card.w .glow{background:#EF4444;box-shadow:0 0 8px #EF4444;}
.stat-card.s .glow{background:#F59E0B;box-shadow:0 0 8px #F59E0B;}
.stat-card.m .glow{background:#A855F7;box-shadow:0 0 8px #A855F7;}
.stat-card.sc .glow{background:#F97316;box-shadow:0 0 12px #F97316;}
.sn { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; line-height:1; letter-spacing:-1px; }
.sl { font-size:0.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:2px; margin-top:5px; font-weight:600; }
.cg{color:#22C55E;} .cr{color:#EF4444;} .ca{color:#F59E0B;} .co{color:#F97316;} .cp{color:#A855F7;}

/* RESULT BANNER */
.rbanner {
    border-radius:14px; padding:18px 22px; margin:14px 0;
    display:flex; align-items:center; gap:14px;
    font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
}
.rb-ex{background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.3);color:#22C55E;}
.rb-gd{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.3);color:#F59E0B;}
.rb-av{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.3);color:#F97316;}
.rb-pr{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.3);color:#EF4444;}

/* SCORE PROGRESS */
.score-track{height:7px;background:var(--surface3);border-radius:4px;overflow:hidden;margin-top:6px;}
.score-fill{height:100%;border-radius:4px;}

/* BUBBLE TABLE */
.btable{width:100%;border-collapse:collapse;font-size:0.84rem;background:var(--surface);}
.btable th{background:var(--surface2);color:var(--muted);text-transform:uppercase;font-size:0.69rem;
           font-weight:700;letter-spacing:1.5px;padding:13px 12px;border-bottom:1px solid var(--border);}
.btable td{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--text);}
.btable tr:hover td{background:rgba(249,115,22,.04);}
.btable td:first-child{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--muted);}
.bs{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;
    font-size:0.71rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;}
.bs-c{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:#22C55E;}
.bs-w{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#EF4444;}
.bs-s{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:#F59E0B;}
.bs-m{background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.3);color:#A855F7;}

/* LEGEND CHIPS */
.legend{display:flex;gap:9px;flex-wrap:wrap;margin:10px 0 16px;}
.chip{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:20px;font-size:0.73rem;font-weight:600;border:1px solid;}
.cg-c{background:rgba(34,197,94,.08);border-color:rgba(34,197,94,.3);color:#22C55E;}
.cg-r{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.3);color:#EF4444;}
.cg-a{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.3);color:#F59E0B;}
.cg-p{background:rgba(168,85,247,.08);border-color:rgba(168,85,247,.3);color:#A855F7;}
.cg-g{background:rgba(100,116,139,.08);border-color:rgba(100,116,139,.3);color:#64748B;}

/* DEBUG */
.dlog{background:var(--surface2);border:1px solid var(--border);border-radius:9px;
      padding:14px;font-family:'JetBrains Mono',monospace;font-size:0.74rem;
      color:var(--muted);margin-top:10px;max-height:220px;overflow-y:auto;}
.dlog .ok{color:#22C55E;} .dlog .warn{color:#F59E0B;} .dlog .info{color:#60A5FA;}

/* SECTION HEADER */
.sh{display:flex;align-items:center;gap:9px;margin:22px 0 12px;}
.sh h3{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--saffron);box-shadow:0 0 8px var(--saffron);}

/* SIDEBAR */
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
section[data-testid="stSidebar"] label{font-weight:600!important;font-size:0.83rem!important;}
.stSelectbox>div>div,.stTextInput>div>input,.stNumberInput>div>input{
    background:var(--surface2)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
.stSlider>div>div>div{background:var(--saffron)!important;}

/* BUTTON */
.stButton>button{
    background:linear-gradient(135deg,#F97316,#EA580C)!important;color:#fff!important;
    border:none!important;border-radius:10px!important;padding:11px 26px!important;
    font-weight:700!important;font-family:'Syne',sans-serif!important;font-size:1rem!important;
    box-shadow:0 4px 16px rgba(249,115,22,.25)!important;transition:all .2s!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(249,115,22,.4)!important;}
.stProgress>div>div{background:var(--saffron)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--text)!important;}
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
    correct: int = 0; wrong: int = 0; unattempted: int = 0; multi: int = 0
    pos_score: float = 0; neg_score: float = 0; total_score: float = 0
    debug_log: List[str] = field(default_factory=list)
    questions_found: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  PRECISE OMR ENGINE v4.0
#  Anchor-Square Based Detection
#
#  The official sheet has:  ■  A○  B○  C○  D○  ■  per row
#  Strategy: find solid ■ squares → use them as perfect position anchors
#  → interpolate A/B/C/D bubble positions between anchors
# ═══════════════════════════════════════════════════════════════════════════════
class PreciseOMREngine:

    def __init__(self):
        self.debug_log = []

    def log(self, msg: str, level: str = 'info'):
        tag = {'info': '[INFO]', 'ok': '[OK]', 'warn': '[WARN]', 'err': '[ERR]'}.get(level, '[INFO]')
        self.debug_log.append(f"{tag} {msg}")

    # ── Image Loading ──────────────────────────────────────────────────────────
    def pdf_to_image(self, pdf_bytes: bytes, dpi: int = 300) -> np.ndarray:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        self.log(f"PDF loaded at {dpi} DPI → {pix.width}×{pix.height}px", 'ok')
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def pil_to_cv(self, pil_img: Image.Image) -> np.ndarray:
        img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        self.log(f"Image loaded: {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def make_thresh(self, img: np.ndarray):
        """Returns (gray, binary_inv_thresh, adaptive_thresh)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Sharp binary threshold for finding solid black squares
        _, binary_inv = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        # Adaptive threshold for measuring bubble fills
        adaptive = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 6
        )
        return gray, binary_inv, adaptive

    # ── STEP 1: Find Solid Black Anchor Squares (■) ───────────────────────────
    def find_anchor_squares(self, binary_inv: np.ndarray, img_shape: tuple) -> List[dict]:
        """
        The OMR sheet has solid black filled squares ■ on BOTH sides of each row.
        These are machine-printed and highly reliable anchors.
        Filter by: solid fill > 75%, roughly square aspect, appropriate size.
        """
        H, W = img_shape[:2]
        contours, _ = cv2.findContours(binary_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        squares = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Size filter: squares shouldn't be too tiny or too large
            # At 300 DPI, the timing squares are roughly 10-20px each side
            min_area = max(60, (W * H) * 0.00005)
            max_area = max(1200, (W * H) * 0.0008)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 6 or h < 6:
                continue

            aspect = w / float(h)
            if not (0.45 < aspect < 2.2):  # Must be roughly square
                continue

            # FILL CHECK: Solid black square must be mostly filled
            roi = binary_inv[y:y+h, x:x+w]
            fill_ratio = np.count_nonzero(roi) / roi.size if roi.size > 0 else 0
            if fill_ratio < 0.65:
                continue

            # Exclude the header/footer regions
            if y < H * 0.12 or y > H * 0.92:
                continue

            cx, cy = x + w // 2, y + h // 2
            squares.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': cx, 'cy': cy,
                'area': area, 'fill': fill_ratio,
                'bbox': (x, y, w, h)
            })

        self.log(f"Raw anchor squares found: {len(squares)}", 'info')
        return squares

    # ── STEP 2: Classify Squares as LEFT or RIGHT Anchors ────────────────────
    def classify_anchors(self, squares: List[dict], img_shape: tuple) -> Tuple[List[dict], List[dict]]:
        """
        The sheet has 3 column groups. In each group:
          LEFT anchor ■ is at x ≈ col_left_edge
          RIGHT anchor ■ is at x ≈ col_right_edge
        Strategy: cluster by x-position to find left vs right anchors per column
        """
        H, W = img_shape[:2]
        if not squares:
            return [], []

        # Sort by x
        xs = sorted(set(s['cx'] for s in squares))
        
        # Median square width — used to group x-positions
        med_w = int(np.median([s['w'] for s in squares]))
        
        # Cluster x-positions using gap detection
        x_clusters = []
        curr = [xs[0]]
        for xv in xs[1:]:
            if xv - curr[-1] < med_w * 3:
                curr.append(xv)
            else:
                x_clusters.append(curr)
                curr = [xv]
        x_clusters.append(curr)
        
        # Find median x for each cluster
        cluster_centers = [int(np.median(c)) for c in x_clusters]
        self.log(f"X-clusters at: {cluster_centers}", 'info')

        # The sheet has 3 columns, each with a LEFT and RIGHT anchor
        # So we expect ~6 distinct x-positions
        # Left anchors: left side of each column group
        # Right anchors: right side of each column group
        # Column 1: Q1-Q20  |  Column 2: Q21-Q40  |  Column 3: Q41-Q60
        # Each column section: L_anchor ... [A][B][C][D] ... R_anchor
        
        # Assign each square to nearest cluster center
        for sq in squares:
            dists = [abs(sq['cx'] - cc) for cc in cluster_centers]
            sq['x_cluster'] = np.argmin(dists)
        
        # Split image into 3 vertical sections
        col_w = W / 3.0
        for sq in squares:
            if sq['cx'] < col_w:
                sq['col_section'] = 0
            elif sq['cx'] < 2 * col_w:
                sq['col_section'] = 1
            else:
                sq['col_section'] = 2

        # Within each column section, leftmost x-cluster = LEFT anchor, rightmost = RIGHT anchor
        left_anchors = []
        right_anchors = []

        for col_sec in range(3):
            col_sqs = [s for s in squares if s['col_section'] == col_sec]
            if not col_sqs:
                continue
            col_x_clusters = sorted(set(s['x_cluster'] for s in col_sqs))
            if len(col_x_clusters) >= 2:
                left_cl = min(col_x_clusters)
                right_cl = max(col_x_clusters)
                for sq in col_sqs:
                    if sq['x_cluster'] == left_cl:
                        sq['anchor_type'] = 'left'
                        left_anchors.append(sq)
                    elif sq['x_cluster'] == right_cl:
                        sq['anchor_type'] = 'right'
                        right_anchors.append(sq)
            elif len(col_x_clusters) == 1:
                # Only one x-cluster in this column section — treat as left
                for sq in col_sqs:
                    sq['anchor_type'] = 'left'
                    left_anchors.append(sq)

        self.log(f"Left anchors: {len(left_anchors)}, Right anchors: {len(right_anchors)}", 'ok')
        return left_anchors, right_anchors

    # ── STEP 3: Build Precise Question Grid ───────────────────────────────────
    def build_precise_grid(self, left_anchors: List[dict], right_anchors: List[dict],
                            img_shape: tuple) -> Dict[int, dict]:
        """
        For each left anchor (one per question row):
        - Find the corresponding right anchor (same y, same column section)
        - The 4 bubbles A/B/C/D are evenly spaced between left_anchor.right_edge and right_anchor.left_edge
        - Relative positions from the official sheet: A ≈ 25%, B ≈ 42%, C ≈ 58%, D ≈ 75% of span

        Returns: { q_num: { 'A': bubble_dict, 'B': ..., 'C': ..., 'D': ..., 'left': ..., 'right': ... } }
        """
        H, W = img_shape[:2]
        if not left_anchors:
            self.log("No left anchors found — cannot build grid!", 'err')
            return {}

        # Sort left anchors by (col_section, y)
        left_anchors.sort(key=lambda s: (s['col_section'], s['cy']))

        # Build a lookup: right anchors by (col_section, approximate_y)
        right_by_col = {0: [], 1: [], 2: []}
        for ra in right_anchors:
            right_by_col[ra['col_section']].append(ra)

        # Bubble relative positions within the span from left_anchor right-edge to right_anchor left-edge
        # From the official sheet analysis: 4 bubbles roughly at 20%, 40%, 60%, 80% of span
        # More precisely, the bubbles span most of the inter-anchor gap
        # We use a 5-division model: [0.18, 0.38, 0.58, 0.78] relative positions
        OPTION_RATIOS = [0.18, 0.38, 0.58, 0.78]

        question_map = {}
        q_counter = {0: 1, 1: 21, 2: 41}  # Starting q_num per col_section

        # Group left anchors by column section
        by_col = {0: [], 1: [], 2: []}
        for la in left_anchors:
            by_col[la['col_section']].append(la)

        for col_sec in range(3):
            col_left = by_col[col_sec]
            col_right = right_by_col[col_sec]
            
            if not col_left:
                continue

            col_left.sort(key=lambda s: s['cy'])
            col_right_sorted = sorted(col_right, key=lambda s: s['cy']) if col_right else []

            # Learn bubble radius from anchor square sizes
            med_sq_h = int(np.median([s['h'] for s in col_left]))
            bubble_r = max(8, int(med_sq_h * 0.7))

            # For each left anchor, find the best matching right anchor (closest y)
            for row_idx, la in enumerate(col_left[:20]):
                q_num = q_counter[col_sec] + row_idx

                # Find best right anchor for this row
                best_ra = None
                if col_right_sorted:
                    best_ra = min(col_right_sorted, key=lambda r: abs(r['cy'] - la['cy']))
                    # Only use if it's within reasonable vertical distance
                    if abs(best_ra['cy'] - la['cy']) > med_sq_h * 3:
                        best_ra = None

                # Compute span
                if best_ra is not None:
                    span_x_start = la['x'] + la['w']   # Right edge of left anchor
                    span_x_end = best_ra['x']           # Left edge of right anchor
                    row_y = (la['cy'] + best_ra['cy']) // 2
                else:
                    # No right anchor: estimate span from column width
                    col_width = W / 3.0
                    span_x_start = la['x'] + la['w']
                    span_x_end = int(la['cx'] + col_width * 0.75)
                    row_y = la['cy']

                span = span_x_end - span_x_start
                if span < 20:
                    # Fallback span
                    span = int(W / 6)
                    span_x_end = span_x_start + span

                opts = {}
                for i, opt in enumerate(['A', 'B', 'C', 'D']):
                    cx = int(span_x_start + OPTION_RATIOS[i] * span)
                    cy = int(row_y)
                    opts[opt] = {
                        'x': cx, 'y': cy, 'r': bubble_r,
                        'bbox': (cx - bubble_r, cy - bubble_r, bubble_r * 2, bubble_r * 2)
                    }
                opts['_left_anchor'] = la
                opts['_right_anchor'] = best_ra
                question_map[q_num] = opts

        self.log(f"Grid built: {len(question_map)} questions mapped", 'ok')
        return question_map

    # ── STEP 4: Calibrate Bubble Positions ────────────────────────────────────
    def calibrate_bubble_positions(self, question_map: Dict, gray: np.ndarray, adaptive: np.ndarray) -> Dict:
        """
        Fine-tune bubble positions: for each question row, find actual circles
        near the expected positions using local circle detection.
        This corrects any slight scanner skew or layout variation.
        """
        H, W = gray.shape[:2]
        calibrated_map = {}

        for q_num, opts in question_map.items():
            new_opts = {}
            for opt in ['A', 'B', 'C', 'D']:
                if opt not in opts:
                    continue
                b = opts[opt]
                cx, cy, r = b['x'], b['y'], b['r']
                
                # Search window around expected position
                search_pad = max(r, 12)
                x1 = max(0, cx - search_pad * 2)
                y1 = max(0, cy - search_pad)
                x2 = min(W, cx + search_pad * 2)
                y2 = min(H, cy + search_pad)

                roi_gray = gray[y1:y2, x1:x2]
                if roi_gray.size == 0:
                    new_opts[opt] = b
                    continue

                # Try Hough circles in the local search window
                roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 1)
                circles = cv2.HoughCircles(
                    roi_blur, cv2.HOUGH_GRADIENT, dp=1.0,
                    minDist=r, param1=50, param2=20,
                    minRadius=max(4, r - 6), maxRadius=r + 8
                )

                if circles is not None:
                    # Find closest circle to expected position
                    best_circle = None
                    best_dist = search_pad * 1.5
                    for x_, y_, r_ in circles[0]:
                        global_cx = int(x1 + x_)
                        global_cy = int(y1 + y_)
                        dist = np.hypot(global_cx - cx, global_cy - cy)
                        if dist < best_dist:
                            best_dist = dist
                            best_circle = (global_cx, global_cy, int(r_))

                    if best_circle is not None:
                        ncx, ncy, nr = best_circle
                        new_opts[opt] = {
                            'x': ncx, 'y': ncy, 'r': nr,
                            'bbox': (ncx - nr, ncy - nr, nr * 2, nr * 2),
                            'calibrated': True
                        }
                        continue

                # No local circle found — keep original
                new_opts[opt] = {**b, 'calibrated': False}

            new_opts['_left_anchor'] = opts.get('_left_anchor')
            new_opts['_right_anchor'] = opts.get('_right_anchor')
            calibrated_map[q_num] = new_opts

        calibrated = sum(1 for opts in calibrated_map.values()
                        for opt in ['A','B','C','D']
                        if opt in opts and opts[opt].get('calibrated'))
        self.log(f"Calibrated {calibrated}/{len(calibrated_map)*4} bubble positions", 'ok')
        return calibrated_map

    # ── STEP 5: Measure Bubble Fill ───────────────────────────────────────────
    def measure_fill(self, gray: np.ndarray, adaptive: np.ndarray, bubble: dict) -> float:
        """
        Multi-method fill measurement inside an elliptical mask:
        1. Adaptive threshold fill
        2. Intensity darkness vs background
        Returns combined fill score [0.0, 1.0]
        """
        cx, cy, r = bubble['x'], bubble['y'], bubble.get('r', 10)
        H, W = gray.shape[:2]

        # Slightly tighter mask (inner 80% of radius) to avoid circle outline interference
        r_inner = max(3, int(r * 0.80))
        x1 = max(0, cx - r_inner)
        y1 = max(0, cy - r_inner)
        x2 = min(W, cx + r_inner)
        y2 = min(H, cy + r_inner)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Create circular mask
        h_roi, w_roi = y2 - y1, x2 - x1
        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        local_cx, local_cy = cx - x1, cy - y1
        cv2.circle(mask, (local_cx, local_cy), r_inner, 255, -1)
        
        total_px = np.count_nonzero(mask)
        if total_px == 0:
            return 0.0

        # Method 1: Adaptive threshold fill
        roi_adaptive = adaptive[y1:y2, x1:x2]
        filled_adaptive = np.count_nonzero(roi_adaptive & (mask > 0))
        ratio_adaptive = filled_adaptive / total_px

        # Method 2: Intensity-based darkness
        roi_gray = gray[y1:y2, x1:x2]
        pixel_vals = roi_gray[mask > 0]
        if len(pixel_vals) == 0:
            return ratio_adaptive

        mean_intensity = np.mean(pixel_vals)
        # Background brightness estimate (use sheet edges)
        sheet_brightness = np.percentile(gray, 90)  # 90th percentile ≈ blank paper
        darkness = max(0.0, (sheet_brightness - mean_intensity) / max(sheet_brightness, 1))

        # Method 3: Count very dark pixels (< 128) inside bubble
        dark_px = np.count_nonzero(pixel_vals < 128)
        dark_ratio = dark_px / total_px

        # Weighted combination
        fill = ratio_adaptive * 0.45 + darkness * 0.30 + dark_ratio * 0.25
        return min(1.0, max(0.0, fill))

    # ── STEP 6: Classify Selections ───────────────────────────────────────────
    def classify_row(self, fills: dict, fill_thresh: float) -> List[str]:
        """
        Smart classification:
        - If max_fill < fill_thresh * 0.5: definitely unattempted
        - If one bubble dominates (≥1.8× second highest): single selection
        - If multiple above threshold and close in value: multi-mark
        """
        if not fills:
            return []

        max_fill = max(fills.values())

        # Clearly unattempted
        if max_fill < fill_thresh * 0.4:
            return []

        # Sort fills
        sorted_fills = sorted(fills.items(), key=lambda x: x[1], reverse=True)

        # Single clear selection
        if len(sorted_fills) >= 2:
            top, second = sorted_fills[0][1], sorted_fills[1][1]
            if top >= fill_thresh:
                dominance = top / (second + 1e-6)
                if dominance >= 1.8:
                    return [sorted_fills[0][0]]

        # Multiple selections above threshold
        above = [opt for opt, f in fills.items() if f >= fill_thresh]
        if above:
            return above

        # Fallback: only top selection if above half-threshold
        if sorted_fills[0][1] >= fill_thresh * 0.5:
            return [sorted_fills[0][0]]

        return []

    # ── MAIN GRADE PIPELINE ───────────────────────────────────────────────────
    def grade(self, img: np.ndarray, answer_key: dict,
              pos: float = 3.0, neg: float = 1.0,
              fill_thresh: float = 0.22) -> Tuple[OMRResult, np.ndarray]:

        self.debug_log = []
        self.log(f"=== GRADING START === shape={img.shape}", 'info')

        # Preprocess
        gray, binary_inv, adaptive = self.make_thresh(img)

        # Step 1: Find solid anchor squares
        squares = self.find_anchor_squares(binary_inv, img.shape)

        # Step 2: Classify as left/right anchors
        left_anchors, right_anchors = self.classify_anchors(squares, img.shape)

        # Step 3: Build grid using anchor-based interpolation
        question_map = self.build_precise_grid(left_anchors, right_anchors, img.shape)

        # Step 4: Calibrate using local Hough (fine-tune positions)
        question_map = self.calibrate_bubble_positions(question_map, gray, adaptive)

        # Step 5 & 6: Measure fills and classify
        annotated = img.copy()
        results = []

        for q in range(1, 61):
            key = answer_key.get(q, '')
            opts_map = question_map.get(q, {})

            # Measure fills for this question's bubbles
            fills = {}
            for opt in ['A', 'B', 'C', 'D']:
                if opt in opts_map:
                    fills[opt] = self.measure_fill(gray, adaptive, opts_map[opt])

            selected = self.classify_row(fills, fill_thresh)

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

            br = BubbleResult(q_num=q, detected=selected, answer_key=key,
                              status=status, score=score, fill_values=fills)
            results.append(br)

            # ── Annotate ──────────────────────────────────────────────────────
            # Draw anchor squares
            la = opts_map.get('_left_anchor')
            ra = opts_map.get('_right_anchor')
            if la:
                ax, ay, aw, ah = la['bbox']
                cv2.rectangle(annotated, (ax, ay), (ax+aw, ay+ah), (80, 80, 80), 1)
            if ra:
                ax, ay, aw, ah = ra['bbox']
                cv2.rectangle(annotated, (ax, ay), (ax+aw, ay+ah), (80, 80, 80), 1)

            # Status colors (BGR)
            COLOR = {
                'correct_sel':   (50, 210, 50),
                'wrong_sel':     (50, 50, 230),
                'multi_sel':     (200, 60, 230),
                'unatt_sel':     (50, 180, 220),
                'empty':         (110, 110, 110),
                'correct_miss':  (50, 210, 50),
            }

            for opt in ['A', 'B', 'C', 'D']:
                if opt not in opts_map:
                    continue
                b = opts_map[opt]
                cx, cy, r = b['x'], b['y'], b.get('r', 10)

                # Draw detection zone ring
                cv2.circle(annotated, (cx, cy), r + 3, (40, 40, 60), 1)

                if opt in selected:
                    if status == 'correct':    clr = COLOR['correct_sel']
                    elif status == 'wrong':    clr = COLOR['wrong_sel']
                    elif status == 'multi':    clr = COLOR['multi_sel']
                    else:                      clr = COLOR['unatt_sel']
                    cv2.circle(annotated, (cx, cy), r, clr, -1)
                    cv2.circle(annotated, (cx, cy), r, (255, 255, 255), 1)
                    cv2.putText(annotated, opt, (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
                else:
                    # Empty bubble — show outline + fill value
                    cv2.circle(annotated, (cx, cy), r, (100, 100, 100), 1)
                    fv = fills.get(opt, 0)
                    if fv > 0.05:
                        cv2.putText(annotated, f"{fv:.2f}", (cx - 9, cy + 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, (160, 140, 80), 1)

                # Circle the correct answer in green if it was missed
                if key == opt and opt not in selected and status != 'correct':
                    cv2.circle(annotated, (cx, cy), r + 5, (50, 210, 50), 2)

            # Q label
            if la:
                lx = max(0, la['cx'] - la['w'] * 2 - 5)
                ly = la['cy'] + 4
                lc_map = {'correct': (50,210,50), 'wrong': (50,50,230),
                          'unattempted': (120,120,120), 'multi': (200,60,230)}
                cv2.putText(annotated, f"Q{q}", (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            lc_map.get(status, (180,180,180)), 1)

        # Summarize
        correct = sum(1 for r in results if r.status == 'correct')
        wrong   = sum(1 for r in results if r.status == 'wrong')
        unat    = sum(1 for r in results if r.status == 'unattempted')
        multi   = sum(1 for r in results if r.status == 'multi')
        ps = correct * pos
        ns = (wrong + multi) * neg

        self.log(f"=== RESULTS: {correct}✓ {wrong}✗ {unat}— {multi}M | Score={ps-ns:.1f} ===", 'ok')

        return OMRResult(
            bubbles=results, correct=correct, wrong=wrong, unattempted=unat, multi=multi,
            pos_score=ps, neg_score=ns, total_score=ps - ns,
            debug_log=list(self.debug_log),
            questions_found=len(question_map)
        ), annotated


# ─── Session State ─────────────────────────────────────────────────────────────
if 'answer_key' not in st.session_state:
    st.session_state.answer_key = {i: random.choice(['A', 'B', 'C', 'D']) for i in range(1, 61)}
if 'result' not in st.session_state:
    st.session_state.result = None
if 'original_img' not in st.session_state:
    st.session_state.original_img = None
if 'annotated_img' not in st.session_state:
    st.session_state.annotated_img = None

engine = PreciseOMREngine()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="omr-header">
  <div class="tricolor"><div></div><div></div><div></div></div>
  <h1 class="omr-title">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="omr-sub">Precision OMR Grader v4.0 &nbsp;·&nbsp; Tiranga Yuva Samiti &nbsp;·&nbsp;
    Anchor-Square Based Detection</p>
  <span class="tech-pill">■ Anchor Squares → Interpolated Grid → Adaptive Fill Measurement</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    pos_mark = st.number_input("✅ Correct (+)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    neg_mark = st.number_input("❌ Wrong (−)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    st.markdown("**Detection**")
    fill_threshold = st.slider("Fill Threshold", 0.08, 0.55, 0.22, 0.01,
        help="0.15–0.25 for typical pencil marks. Lower for lighter marks.")
    st.caption(f"Current: {fill_threshold:.2f}")
    show_debug = st.checkbox("🔍 Show Debug Log", False)
    st.divider()
    st.markdown("### 📋 Answer Key")
    bulk_key = st.text_area("Paste 60 answers (comma-separated)", placeholder="A,B,C,D,...", height=70)
    if st.button("Apply Bulk Key", use_container_width=True):
        parts = [p.strip().upper() for p in bulk_key.split(',')]
        for i, ans in enumerate(parts[:60]):
            if ans in ('A','B','C','D',''):
                st.session_state.answer_key[i+1] = ans
        st.success("✅ Applied!")
    st.caption("Or set individually:")
    options_list = ['', 'A', 'B', 'C', 'D']
    ck = st.columns(2)
    for q in range(1, 61):
        col = ck[0] if q % 2 != 0 else ck[1]
        with col:
            st.session_state.answer_key[q] = st.selectbox(
                f"Q{q}", options_list,
                index=options_list.index(st.session_state.answer_key.get(q, '')),
                key=f"key_{q}"
            )

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh"><div class="sdot"></div><h3>Upload & Grade OMR Sheet</h3></div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend">
  <span class="chip cg-c">● Correct</span>
  <span class="chip cg-r">● Wrong</span>
  <span class="chip cg-g">○ Unattempted</span>
  <span class="chip cg-a">○ Skipped</span>
  <span class="chip cg-p">● Multi-Filled</span>
</div>
<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-bottom:14px;">
  <p style="color:var(--muted);font-size:0.86rem;line-height:1.65;">
    <strong style="color:#F97316;">Sheet Format:</strong> 3 columns × 20 rows = 60 questions &nbsp;|&nbsp;
    Each row: <code style="background:var(--surface2);padding:1px 5px;border-radius:4px;">■ A○ B○ C○ D○ ■</code><br>
    <strong style="color:#22C55E;">Engine:</strong> Detects solid black ■ squares as anchors → computes exact bubble positions → adaptive fill measurement<br>
    <strong style="color:#60A5FA;">Best results:</strong> 200–300 DPI scan, PDF format preferred
  </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Drop OMR sheet here", type=['pdf','png','jpg','jpeg','tiff','bmp'])

if uploaded:
    with st.spinner("Loading..."):
        fbytes = uploaded.read()
        if uploaded.type == 'application/pdf':
            img_cv = engine.pdf_to_image(fbytes, dpi=300)
        else:
            img_cv = engine.pil_to_cv(Image.open(io.BytesIO(fbytes)))
        st.session_state.original_img = img_cv.copy()
    st.success(f"✅ **{uploaded.name}** — {img_cv.shape[1]}×{img_cv.shape[0]}px")

    col_orig, col_act = st.columns([1, 1])
    with col_orig:
        st.markdown("**📄 Original Sheet**")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_act:
        st.markdown("**🔬 Settings**")
        st.info(f"""
**Marking:** +{pos_mark:.1f} correct · −{neg_mark:.1f} wrong  
**Fill Threshold:** {fill_threshold:.2f}  
**Detection:** Anchor-square grid method  
**Questions:** 60 (3 cols × 20)
        """)
        if st.button("🚀 Grade OMR Sheet", use_container_width=True):
            bar = st.progress(0, text="Finding anchor squares ■...")
            time.sleep(0.15)
            bar.progress(25, text="Classifying left/right anchors...")
            time.sleep(0.1)
            bar.progress(45, text="Building precise question grid...")
            time.sleep(0.1)
            bar.progress(60, text="Calibrating bubble positions...")
            result, annotated = engine.grade(
                img_cv, st.session_state.answer_key,
                pos=pos_mark, neg=neg_mark, fill_thresh=fill_threshold
            )
            bar.progress(85, text="Scoring & annotating...")
            st.session_state.result = result
            st.session_state.annotated_img = annotated
            time.sleep(0.15)
            bar.progress(100, text="Done!")
            time.sleep(0.2)
            bar.empty()
            found = result.questions_found
            if found >= 55:
                st.success(f"✅ Graded! {found}/60 questions detected.")
            elif found >= 40:
                st.warning(f"⚠️ {found}/60 questions detected. Try adjusting fill threshold.")
            else:
                st.error(f"⚠️ Only {found}/60 questions detected. Check image quality/DPI.")


# ─── RESULTS ──────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    result = st.session_state.result
    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Results Dashboard</h3></div>', unsafe_allow_html=True)

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
        <div style="font-size:0.84rem;font-weight:400;opacity:.8;margin-top:3px;">
          Score: <strong>{result.total_score:.1f}</strong>/{max_score:.0f} &nbsp;·&nbsp; {pct:.1f}%
          &nbsp;·&nbsp; Questions found: {result.questions_found}/60
        </div>
      </div>
    </div>
    <div class="score-track"><div class="score-fill" style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{bar_clr},{bar_clr}88);"></div></div>
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
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:13px 16px;margin-top:8px;">
          <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Attempt Stats</div>
          <div style="display:flex;justify-content:space-between;font-size:0.86rem;padding:3px 0;">
            <span>Attempted</span><span style="font-weight:700;">{attempted}/60 ({attempted/60*100:.0f}%)</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.86rem;padding:3px 0;">
            <span>Unattempted</span><span style="font-weight:700;">{result.unattempted}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.86rem;padding:3px 0;">
            <span>Multi-marked</span><span style="font-weight:700;">{result.multi}</span></div>
          <div style="display:flex;justify-content:space-between;font-size:0.86rem;padding:3px 0;">
            <span>Questions found</span><span style="font-weight:700;">{result.questions_found}/60</span></div>
        </div>
        """, unsafe_allow_html=True)

        if show_debug and result.debug_log:
            st.markdown("**🔍 Debug Log**")
            log_html = "".join(
                f'<div class="{"ok" if "[OK]" in l else "warn" if "[WARN]" in l else "info"}">{l}</div>'
                for l in result.debug_log
            )
            st.markdown(f'<div class="dlog">{log_html}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("**🎯 Graded OMR**")
        if st.session_state.annotated_img is not None:
            ann_rgb = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(ann_rgb).save(buf, format='PNG')
            st.download_button("⬇️ Download Graded Image", buf.getvalue(),
                               "graded_omr.png", "image/png", use_container_width=True)

    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Question-wise Report</h3></div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        filter_status = st.multiselect("Filter by status",
            ['correct','wrong','unattempted','multi'],
            default=['correct','wrong','unattempted','multi'])
    with fc2:
        show_fills = st.checkbox("Show fill values", False, help="Raw fill scores for debugging")

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
            fv_str = " ".join(f"{k}:{v:.3f}" for k, v in sorted(b.fill_values.items()))
            fv_html = f"<td style='font-size:.71rem;color:var(--muted);font-family:JetBrains Mono,monospace;'>{fv_str}</td>"
        rows_html += f"""
        <tr>
          <td>Q{b.q_num:02d}</td>
          <td><span style="color:#60A5FA;font-weight:700;font-family:'JetBrains Mono',monospace;">{det}</span></td>
          <td><span style="color:#34D399;font-weight:700;font-family:'JetBrains Mono',monospace;">{key}</span></td>
          <td><span class="bs {bc}">{bi} {b.status.upper()}</span></td>
          <td class="{sc_cls}" style="font-weight:800;font-family:'JetBrains Mono',monospace;">{sc}</td>
          {fv_html}
        </tr>"""

    extra_th = "<th>Fill Values</th>" if show_fills else ""
    st.markdown(f"""
    <div style="max-height:520px;overflow-y:auto;border:1px solid var(--border);border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">
    <table class="btable">
      <thead><tr><th>Q#</th><th>Detected</th><th>Answer Key</th><th>Status</th><th>Score</th>{extra_th}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    st.write("")
    exp_data = [{'Question': b.q_num, 'Detected': ','.join(b.detected) if b.detected else '',
                 'Answer Key': b.answer_key, 'Status': b.status, 'Score': b.score,
                 **{f'Fill_{k}': round(v, 4) for k, v in b.fill_values.items()}}
                for b in result.bubbles]
    st.download_button("⬇️ Download Full Results CSV", pd.DataFrame(exp_data).to_csv(index=False),
                       "omr_results.csv", "text/csv")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:var(--muted);font-size:0.78rem;">
  <div class="tricolor" style="max-width:140px;margin:0 auto 10px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · Precision OMR v4.0<br>
  <span style="font-size:0.68rem;opacity:.5;">Anchor-Square Detection · Hough Calibration · Adaptive Fill Measurement</span>
</div>
""", unsafe_allow_html=True)
