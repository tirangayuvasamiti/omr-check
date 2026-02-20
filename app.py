import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import tempfile
import os
import time
import random

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yuva Gyan Mahotsav 2026 – OMR Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Noto+Sans:wght@300;400;600&display=swap');

:root {
    --saffron: #FF6B00;
    --saffron-light: #FF8C3A;
    --navy: #003580;
    --navy-light: #0A4FA6;
    --green: #00A550;
    --gold: #FFB800;
    --white: #FFFFFF;
    --bg: #F8FAFC;         /* Light Background */
    --surface: #FFFFFF;    /* White cards */
    --surface2: #F1F5F9;   /* Light gray table headers */
    --border: #E2E8F0;     /* Light border */
    --text: #1E293B;       /* Dark text for visibility */
    --muted: #64748B;      /* Muted dark gray */
    --correct: #16A34A;
    --wrong: #DC2626;
    --skip: #D97706;
    --multi: #9333EA;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* HEADER */
.omr-header {
    background: linear-gradient(135deg, #003580 0%, #0A4FA6 40%, #FF6B00 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.omr-header::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.08'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.omr-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    line-height: 1.1;
    position: relative;
}
.omr-subtitle {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.9);
    margin-top: 6px;
    position: relative;
}
.tricolor-bar {
    display: flex;
    height: 5px;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 16px;
}
.tricolor-bar div:nth-child(1) { flex:1; background:#FF6B00; }
.tricolor-bar div:nth-child(2) { flex:1; background:#FFFFFF; }
.tricolor-bar div:nth-child(3) { flex:1; background:#00A550; }

/* CARDS */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.stat-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 4px;
}
.stat-card.correct::after { background: var(--correct); }
.stat-card.wrong::after   { background: var(--wrong); }
.stat-card.skip::after    { background: var(--skip); }
.stat-card.total::after   { background: var(--multi); }
.stat-card.score::after   { background: var(--gold); }
.stat-num {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
}
.stat-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
    font-weight: 600;
}
.c-green { color: var(--correct); }
.c-red   { color: var(--wrong); }
.c-amber { color: var(--skip); }
.c-orange { color: var(--saffron); }
.c-gold  { color: #D97706; }
.c-purple { color: var(--multi); }

/* LEGEND CHIPS */
.legend { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    border: 1px solid;
}
.chip-green  { background: rgba(22,163,74,0.1);  border-color: #16A34A; color: #16A34A; }
.chip-red    { background: rgba(220,38,38,0.1);  border-color: #DC2626; color: #DC2626; }
.chip-amber  { background: rgba(217,119,6,0.1); border-color: #D97706; color: #D97706; }
.chip-purple { background: rgba(147,51,234,0.1); border-color: #9333EA; color: #9333EA; }
.chip-gray   { background: rgba(100,116,139,0.1); border-color: #64748B; color: #64748B; }

/* BUBBLE TABLE */
.bubble-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; background: var(--surface); }
.bubble-table th {
    background: var(--surface2);
    color: var(--text);
    text-transform: uppercase;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 12px 10px;
    border-bottom: 2px solid var(--border);
}
.bubble-table td { padding: 10px; border-bottom: 1px solid var(--border); color: var(--text); }
.bubble-table tr:hover td { background: rgba(255,107,0,0.03); }
.badge {
    display: inline-block; padding: 4px 10px;
    border-radius: 12px; font-size: 0.75rem; font-weight: 600;
}
.badge-correct  { background: rgba(22,163,74,0.15);  color: #16A34A; }
.badge-wrong    { background: rgba(220,38,38,0.15);  color: #DC2626; }
.badge-skip     { background: rgba(217,119,6,0.15); color: #D97706; }
.badge-multi    { background: rgba(147,51,234,0.15); color: #9333EA; }

/* UPLOAD AREA */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    background: var(--surface);
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--saffron); }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] label { color: var(--text) !important; font-weight: 600;}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--saffron), var(--saffron-light));
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    box-shadow: 0 4px 10px rgba(255,107,0,0.2);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e05e00, var(--saffron));
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(255,107,0,0.4);
}
.stProgress > div > div { background: var(--saffron) !important; }

.stTextInput > div > input,
.stNumberInput > div > input,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

h1,h2,h3,h4 { font-family: 'Rajdhani', sans-serif; color: var(--text); font-weight: 700; }
.stAlert { border-radius: 10px; }

div[data-testid="column"] { gap: 8px; }

.result-banner {
    border-radius: 12px;
    padding: 20px 28px;
    margin: 16px 0;
    border-left: 6px solid;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    background: var(--surface);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.banner-excellent { border-color: #16A34A; color: #16A34A; }
.banner-good      { border-color: #D97706; color: #D97706; }
.banner-average   { border-color: #EA580C; color: #EA580C; }
.banner-poor      { border-color: #DC2626; color: #DC2626; }
</style>
""", unsafe_allow_html=True)

# ─── Data Classes ────────────────────────────────────────────────────────────
@dataclass
class BubbleResult:
    q_num: int
    detected: list   # list of detected options (e.g. ['A'], ['A','B'], [])
    answer_key: str  # '' if not set
    status: str      # 'correct', 'wrong', 'unattempted', 'multi'
    score: float

@dataclass
class OMRResult:
    bubbles: list[BubbleResult]
    correct: int = 0
    wrong: int = 0
    unattempted: int = 0
    multi: int = 0
    pos_score: float = 0
    neg_score: float = 0
    total_score: float = 0
    student_info: dict = field(default_factory=dict)

# ─── OMR Engine ──────────────────────────────────────────────────────────────
class OMREngine:
    """
    Detects bubbles from the Yuva Gyan Mahotsav OMR sheet using machine reading squares.
    """

    def pdf_to_image(self, pdf_bytes: bytes, dpi: int = 200) -> np.ndarray:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def pil_to_cv(self, pil_img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh

    def detect_bubbles(self, img: np.ndarray) -> list:
        # Now identifies ALL valid shapes (both machine squares and circles)
        thresh = self.preprocess(img)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 6000:
                continue
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if not (0.4 < aspect < 2.5):
                continue
            if w < 5 or h < 5:
                continue
            cx, cy = x + w // 2, y + h // 2
            features.append({'x': cx, 'y': cy, 'w': w, 'h': h, 'area': area,
                             'cnt': cnt, 'bbox': (x, y, w, h)})
        return features

    def cluster_bubbles_grid(self, features: list, img_shape: tuple) -> dict:
        """
        Groups utilizing the Machine Reading Squares printed to the left of each question.
        """
        if not features: return {}

        H, W = img_shape[:2]
        margin_top = int(H * 0.05)
        margin_bot = int(H * 0.95)
        features = [f for f in features if margin_top < f['y'] < margin_bot]
        if not features: return {}

        # 1. Split into 3 main columns based on X geometry
        col_width = W / 3.0
        cols = [[], [], []]
        for f in features:
            if f['x'] < col_width: cols[0].append(f)
            elif f['x'] < 2 * col_width: cols[1].append(f)
            else: cols[2].append(f)

        question_map = {}
        q = 1

        sizes = [f['h'] for f in features]
        y_thresh = max(np.median(sizes) * 1.5, 10)

        # 2. Process each column individually
        for col_feats in cols:
            if not col_feats: continue

            # Group into rows by Y
            col_feats.sort(key=lambda f: f['y'])
            rows = []
            current_row = [col_feats[0]]
            for f in col_feats[1:]:
                row_y = np.mean([item['y'] for item in current_row])
                if abs(f['y'] - row_y) < y_thresh:
                    current_row.append(f)
                else:
                    rows.append(current_row)
                    current_row = [f]
            rows.append(current_row)

            # Filter out stray contours (logos/text) by enforcing vertical alignment of markers
            valid_rows = []
            if rows:
                # The leftmost item of each valid row is the machine square
                first_items_x = [r[0]['x'] for r in rows if r]
                if first_items_x:
                    median_marker_x = np.median(first_items_x)
                    for r in rows:
                        if r and abs(r[0]['x'] - median_marker_x) < W * 0.05:
                            valid_rows.append(r)

            # Learn standard spacing from a perfect row (Square + 4 bubbles = 5 items)
            ref_dx = [40, 80, 120, 160] # Fallback pixels
            for r in valid_rows:
                if len(r) >= 5:
                    r.sort(key=lambda f: f['x'])
                    ref_dx = [
                        r[1]['x'] - r[0]['x'],
                        r[2]['x'] - r[0]['x'],
                        r[3]['x'] - r[0]['x'],
                        r[4]['x'] - r[0]['x']
                    ]
                    break

            # Map the options anchoring to the machine reading square
            for row in valid_rows:
                if q > 60: break
                row.sort(key=lambda f: f['x'])
                
                # Element 0 is guaranteed to be the square marker due to sorting & alignment filter
                marker = row[0]
                opts = {'marker': marker}
                
                if len(row) >= 5:
                    opts['A'] = row[1]
                    opts['B'] = row[2]
                    opts['C'] = row[3]
                    opts['D'] = row[4]
                else:
                    # Interpolate perfectly if bubbles are faint/missing
                    w, h = marker['w'], marker['h']
                    cy = marker['y']
                    for i, opt in enumerate(['A', 'B', 'C', 'D']):
                        cx = int(marker['x'] + ref_dx[i])
                        opts[opt] = {
                            'x': cx, 'y': int(cy), 'w': int(w), 'h': int(h),
                            'bbox': (int(cx - w//2), int(cy - h//2), int(w), int(h))
                        }
                
                question_map[q] = opts
                q += 1

        return question_map

    def measure_fill(self, img: np.ndarray, bubble: dict, thresh: np.ndarray) -> float:
        """Returns fill ratio (0=empty, 1=fully filled) for a bubble."""
        x, y, w, h = bubble['bbox']
        pad = 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(thresh.shape[1], x + w + pad), min(thresh.shape[0], y + h + pad)
        roi = thresh[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return np.count_nonzero(roi) / roi.size

    def classify_bubbles(self, img: np.ndarray, question_map: dict,
                         fill_thresh: float = 0.28,
                         multi_thresh: float = 0.28) -> dict:
        """
        Determine which bubbles are filled based only on the A/B/C/D mapping.
        """
        thresh = self.preprocess(img)
        results = {}
        for q, opts in question_map.items():
            fills = {}
            for opt in ['A', 'B', 'C', 'D']:
                if opt in opts:
                    fills[opt] = self.measure_fill(img, opts[opt], thresh)
            selected = [opt for opt, f in fills.items() if f >= fill_thresh]
            results[q] = {'selected': selected, 'fills': fills}
        return results

    def grade(self, img: np.ndarray, answer_key: dict,
              pos: float = 3.0, neg: float = 1.0,
              fill_thresh: float = 0.28) -> tuple:
        """
        Full grading pipeline.
        Returns (OMRResult, annotated_img)
        """
        bubbles_raw = self.detect_bubbles(img)
        question_map = self.cluster_bubbles_grid(bubbles_raw, img.shape)
        bubble_results_raw = self.classify_bubbles(img, question_map, fill_thresh)

        annotated = img.copy()
        results = []

        for q in range(1, 61):
            key = answer_key.get(q, '')
            raw = bubble_results_raw.get(q, {'selected': [], 'fills': {}})
            selected = raw['selected']
            opts_map = question_map.get(q, {})

            # Determine status
            if len(selected) == 0:
                status = 'unattempted'
                score = 0
            elif len(selected) > 1:
                status = 'multi'
                score = -neg if key else 0
            elif key and selected[0] == key:
                status = 'correct'
                score = pos
            elif key:
                status = 'wrong'
                score = -neg
            else:
                status = 'unattempted'
                score = 0

            br = BubbleResult(
                q_num=q,
                detected=selected,
                answer_key=key,
                status=status,
                score=score
            )
            results.append(br)

            # Annotate machine reading square first
            marker = opts_map.get('marker')
            if marker:
                mx, my, mw, mh = marker['bbox']
                cv2.rectangle(annotated, (mx, my), (mx+mw, my+mh), (0, 107, 255), 2) # Saffron box

            # Annotate bubbles
            for opt in ['A', 'B', 'C', 'D']:
                if opt not in opts_map: continue
                bubble = opts_map[opt]
                
                cx, cy = bubble['x'], bubble['y']
                r = max(bubble['w'], bubble['h']) // 2 + 3

                if opt in selected:
                    # Filled bubble colors
                    if status == 'correct':
                        color = (74, 163, 22)   # Green for correct
                    elif status == 'wrong':
                        color = (38, 38, 220)   # Red for wrong
                    elif status == 'multi':
                        color = (234, 51, 147)  # Purple for multiple
                    else:
                        color = (6, 119, 217)   # Amber/Orange for un-keyed fill
                        
                    cv2.circle(annotated, (cx, cy), r, color, -1)
                    cv2.circle(annotated, (cx, cy), r+2, color, 2)
                    # Option label in white over filled color
                    cv2.putText(annotated, opt, (cx-5, cy+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                else:
                    # Unfilled - gray outline
                    cv2.circle(annotated, (cx, cy), r, (160, 160, 160), 2)

                # Mark correct answer with a thick green ring if missed
                if key == opt and opt not in selected and status != 'correct':
                    cv2.circle(annotated, (cx, cy), r+5, (74, 163, 22), 2)

            # Question number label drawn strictly using the marker
            if marker:
                cv2.putText(annotated, str(q),
                            (marker['x'] - 30, marker['y'] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

        # Tally
        correct = sum(1 for r in results if r.status == 'correct')
        wrong = sum(1 for r in results if r.status == 'wrong')
        unattempted = sum(1 for r in results if r.status == 'unattempted')
        multi = sum(1 for r in results if r.status == 'multi')
        pos_score = correct * pos
        neg_score = (wrong + multi) * neg
        total = pos_score - neg_score

        omr_result = OMRResult(
            bubbles=results,
            correct=correct,
            wrong=wrong,
            unattempted=unattempted,
            multi=multi,
            pos_score=pos_score,
            neg_score=neg_score,
            total_score=total
        )
        return omr_result, annotated

# ─── Session State ───────────────────────────────────────────────────────────
# Pre-fill answer key with random values automatically as requested
if 'answer_key' not in st.session_state:
    st.session_state.answer_key = {i: random.choice(['A', 'B', 'C', 'D']) for i in range(1, 61)}
if 'result' not in st.session_state:
    st.session_state.result = None
if 'original_img' not in st.session_state:
    st.session_state.original_img = None
if 'annotated_img' not in st.session_state:
    st.session_state.annotated_img = None

engine = OMREngine()

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="omr-header">
  <div class="tricolor-bar"><div></div><div></div><div></div></div>
  <h1 class="omr-title">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="omr-subtitle">Automated OMR Grading System &nbsp;|&nbsp; Tiranga Yuva Samiti &nbsp;|&nbsp; Marking: +3 Correct · −1 Wrong · 0 Unattempted</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("**Marking Scheme**")
    pos_mark = st.number_input("✅ Correct (+)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    neg_mark = st.number_input("❌ Wrong (−)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    
    st.markdown("**Detection Sensitivity**")
    fill_threshold = st.slider("Fill Threshold", 0.10, 0.60, 0.28, 0.02,
                               help="Lower = more sensitive (detects lighter marks)")
    
    st.divider()
    st.markdown("### 📋 Answer Key (Pre-filled Randomly)")
    st.caption("Answers are pre-populated randomly for testing. You can edit them below.")
    
    # Bulk input
    bulk_key = st.text_area(
        "Paste 60 answers (comma-separated, e.g. A,B,C,D,...)",
        placeholder="A,B,C,D,A,B,C,D,...",
        height=80
    )
    if st.button("Apply Bulk Key"):
        parts = [p.strip().upper() for p in bulk_key.split(',')]
        for i, ans in enumerate(parts[:60]):
            if ans in ('A','B','C','D',''):
                st.session_state.answer_key[i+1] = ans
        st.success("Answer key applied!")
    
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

# ─── MAIN CONTENT (Single Page Flow) ─────────────────────────────────────────

st.markdown("### 📤 Upload & Grade OMR Sheet")
st.markdown("""
<div class="legend">
  <span class="chip chip-green">🟢 Correct Answer</span>
  <span class="chip chip-red">🔴 Wrong Answer</span>
  <span class="chip chip-gray">⚪ Unfilled Option</span>
  <span class="chip chip-amber">🟡 Unattempted/Missed</span>
  <span class="chip chip-purple">🟣 Multiple Marked</span>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop the OMR PDF or Image here",
    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="Supports: PDF (scanned), PNG, JPG, TIFF"
)

if uploaded:
    with st.spinner("🔍 Processing OMR sheet..."):
        file_bytes = uploaded.read()
        if uploaded.type == 'application/pdf':
            img_cv = engine.pdf_to_image(file_bytes, dpi=200)
        else:
            pil_img = Image.open(io.BytesIO(file_bytes))
            img_cv = engine.pil_to_cv(pil_img)
        st.session_state.original_img = img_cv.copy()
    
    st.success(f"✅ File loaded: {uploaded.name} ({img_cv.shape[1]}×{img_cv.shape[0]}px)")
    
    col_orig, col_grade_btn = st.columns([1, 1])
    with col_orig:
        st.markdown("#### 📄 Original OMR")
        orig_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        st.image(orig_rgb, use_container_width=True)
    
    with col_grade_btn:
        st.write("") # Spacer
        if st.button("🚀 Grade OMR Sheet Now", use_container_width=True):
            progress = st.progress(0, text="Detecting anchors & bubbles...")
            time.sleep(0.3)
            progress.progress(30, text="Clustering grid via Machine Squares...")
            
            result, annotated = engine.grade(
                img_cv,
                st.session_state.answer_key,
                pos=pos_mark,
                neg=neg_mark,
                fill_thresh=fill_threshold
            )
            progress.progress(80, text="Scoring...")
            st.session_state.result = result
            st.session_state.annotated_img = annotated
            time.sleep(0.2)
            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()
            
            st.success("✅ Grading complete! Scroll down to see results.")

# ─── RESULTS DISPLAY (Shown sequentially on the same window) ────────────────
if st.session_state.result is not None:
    result = st.session_state.result
    
    st.divider()
    st.markdown("### 📊 Results Dashboard")
    
    # Score banner
    total_q = 60
    pct = (result.total_score / (total_q * pos_mark)) * 100 if total_q * pos_mark > 0 else 0
    if pct >= 75:
        banner_cls, grade_txt = "banner-excellent", "🏆 Outstanding Performance!"
    elif pct >= 50:
        banner_cls, grade_txt = "banner-good", "👍 Good Performance"
    elif pct >= 35:
        banner_cls, grade_txt = "banner-average", "📚 Average – Keep Practicing"
    else:
        banner_cls, grade_txt = "banner-poor", "⚠️ Needs Improvement"
    
    st.markdown(f'<div class="result-banner {banner_cls}">{grade_txt} &nbsp;|&nbsp; Score: {result.total_score:.1f} / {total_q * pos_mark:.0f} &nbsp;|&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
    
    # Stat cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="stat-card correct"><div class="stat-num c-green">{result.correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card wrong"><div class="stat-num c-red">{result.wrong}</div><div class="stat-label">Wrong</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card skip"><div class="stat-num c-amber">{result.unattempted}</div><div class="stat-label">Skipped</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card total"><div class="stat-num c-purple">{result.multi}</div><div class="stat-label">Multi-filled</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="stat-card score"><div class="stat-num c-gold">{result.total_score:.1f}</div><div class="stat-label">Final Score</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Score breakdown & Annotated image side by side
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### 📉 Score Breakdown")
        breakdown_data = {
            'Component': ['Positive Score', 'Negative Score', 'Net Score'],
            'Value': [f"+{result.pos_score:.1f}", f"−{result.neg_score:.1f}", f"{result.total_score:.1f}"]
        }
        df_break = pd.DataFrame(breakdown_data)
        st.dataframe(df_break, hide_index=True, use_container_width=True)
        
        st.markdown("#### 📈 Accuracy")
        attempted = result.correct + result.wrong + result.multi
        accuracy = (result.correct / attempted * 100) if attempted > 0 else 0
        st.metric("Attempt Rate", f"{attempted}/60 ({attempted/60*100:.0f}%)")
        st.metric("Accuracy (of attempted)", f"{accuracy:.1f}%")
    
    with col_b:
        st.markdown("#### 🎯 Graded OMR Preview")
        if st.session_state.annotated_img is not None:
            ann_rgb = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)
            
            # Download button for annotated image
            ann_pil = Image.fromarray(ann_rgb)
            buf = io.BytesIO()
            ann_pil.save(buf, format='PNG')
            st.download_button("⬇️ Download Graded OMR", buf.getvalue(),
                               file_name="graded_omr.png", mime="image/png", use_container_width=True)

    st.divider()
    
    # ─── DETAILED REPORT ────────────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Question-wise Report")
    
    # Filters
    filter_status = st.multiselect(
        "Filter by status",
        ['correct', 'wrong', 'unattempted', 'multi'],
        default=['correct', 'wrong', 'unattempted', 'multi']
    )
    
    filtered = [b for b in result.bubbles if b.status in filter_status]
    
    # Build HTML table for light theme
    rows_html = ""
    for b in filtered:
        detected_str = ', '.join(b.detected) if b.detected else '—'
        key_str = b.answer_key if b.answer_key else '—'
        score_str = f"+{b.score:.0f}" if b.score > 0 else (f"{b.score:.0f}" if b.score != 0 else "0")
        score_color = "c-green" if b.score > 0 else ("c-red" if b.score < 0 else "c-amber")
        
        badge_map = {
            'correct': 'badge-correct',
            'wrong': 'badge-wrong',
            'unattempted': 'badge-skip',
            'multi': 'badge-multi'
        }
        badge_cls = badge_map.get(b.status, '')
        status_label = b.status.upper()
        
        rows_html += f"""
        <tr>
          <td style="font-weight:700;">Q{b.q_num}</td>
          <td><span style="color:#2563EB; font-weight:700;">{detected_str}</span></td>
          <td><span style="color:#059669; font-weight:700;">{key_str}</span></td>
          <td><span class="badge {badge_cls}">{status_label}</span></td>
          <td class="{score_color}" style="font-weight:800;">{score_str}</td>
        </tr>"""
    
    table_html = f"""
    <div style="max-height:500px; overflow-y:auto; border:1px solid var(--border); border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <table class="bubble-table">
      <thead>
        <tr>
          <th>Q#</th><th>Detected</th><th>Answer Key</th><th>Status</th><th>Score</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)
    
    # CSV Export
    st.write("") # Spacer
    export_data = []
    for b in result.bubbles:
        export_data.append({
            'Question': b.q_num,
            'Detected': ', '.join(b.detected) if b.detected else '',
            'Answer Key': b.answer_key,
            'Status': b.status,
            'Score': b.score
        })
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False)
    st.download_button(
        "⬇️ Download Results CSV",
        csv,
        file_name="omr_results.csv",
        mime="text/csv"
    )

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:30px 0 10px; color:var(--muted); font-size:0.85rem; font-weight:600;">
  <div class="tricolor-bar" style="max-width:200px; margin:0 auto 12px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · OMR Auto-Grader
</div>
""", unsafe_allow_html=True)
