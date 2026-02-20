"""
YUVA GYAN MAHOTSAV 2026 — PRECISION OMR GRADER v8.0
=====================================================
Pixel-verified bubble grid. No broken homography.
Clean annotated output aligned perfectly on the sheet.

Structure: ■(L)  A○  B○  C○  D○  ■(R)  × 3 cols × 20 rows = 60 questions

VERIFIED PIXEL POSITIONS (1240×1754 canonical):
  Col 1 (Q01-20): L=135  A=189  B=250  C=310  D=372  R=409
  Col 2 (Q21-40): L=502  A=557  B=618  C=677  D=738  R=776
  Col 3 (Q41-60): L=870  A=924  B=983  C=1044  D=1105  R=1142
  Row 1 Y=360, spacing=54.5px
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import time

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
  --s:#F97316;--sd:#EA580C;--g:#22C55E;--gd:#16A34A;
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
.sc.c .gl{background:#22C55E;box-shadow:0 0 8px #22C55E;}
.sc.w .gl{background:#EF4444;box-shadow:0 0 8px #EF4444;}
.sc.s .gl{background:#F59E0B;box-shadow:0 0 8px #F59E0B;}
.sc.m .gl{background:#A855F7;box-shadow:0 0 8px #A855F7;}
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
.sq{color:#F97316;font-weight:700;}.cir{color:#60A5FA;}
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
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;color:var(--tx)!important;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CANONICAL GRID  —  verified pixel-by-pixel on actual YGM 2026 sheet
# ═══════════════════════════════════════════════════════════════════════════════
CANONICAL_W = 1240
CANONICAL_H = 1754

BUBBLE_CX = {
    0: {'A': 189, 'B': 250, 'C': 310, 'D': 372},
    1: {'A': 557, 'B': 618, 'C': 677, 'D': 738},
    2: {'A': 924, 'B': 983, 'C': 1044, 'D': 1105},
}
SQ_L        = [135, 502, 870]
SQ_R        = [409, 776, 1142]
ROW_Y_START = 360
ROW_SPACING = 54.5
SAMPLE_R    = 11
DRAW_R      = 13
FONT        = cv2.FONT_HERSHEY_SIMPLEX

C_CORRECT   = ( 34, 185,  34)
C_WRONG     = ( 35,  35, 210)
C_MULTI     = (170,  35, 210)
C_UNATTEMPT = (120, 120, 130)
C_EMPTY     = (175, 170, 190)
C_CORR_IND  = ( 34, 185,  34)
C_WHITE     = (255, 255, 255)


@dataclass
class BubbleResult:
    q_num:       int
    detected:    list
    answer_key:  str
    status:      str
    score:       float
    fill_values: dict = field(default_factory=dict)

@dataclass
class OMRResult:
    bubbles:     List[BubbleResult]
    correct:     int   = 0
    wrong:       int   = 0
    unattempted: int   = 0
    multi:       int   = 0
    pos_score:   float = 0
    neg_score:   float = 0
    total_score: float = 0
    debug_log:   List[str] = field(default_factory=list)


class OMREngine:

    def __init__(self):
        self._logs: List[str] = []

    def _log(self, msg: str, lvl: str = 'info'):
        t = {'ok':'[OK]','warn':'[WARN]','err':'[ERR]'}.get(lvl,'[INFO]')
        self._logs.append(f"{t} {msg}")

    def load_pdf(self, data: bytes, dpi: int = 150) -> np.ndarray:
        pages = convert_from_bytes(data, dpi=dpi)
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        self._log(f"PDF → {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    def load_img(self, pil: Image.Image) -> np.ndarray:
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._log(f"Image → {img.shape[1]}×{img.shape[0]}px", 'ok')
        return img

    def _to_canonical(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        if W == CANONICAL_W and H == CANONICAL_H:
            self._log("Already canonical — no resize needed", 'ok')
            return img.copy()
        interp = cv2.INTER_AREA if W >= CANONICAL_W else cv2.INTER_LINEAR
        out = cv2.resize(img, (CANONICAL_W, CANONICAL_H), interpolation=interp)
        self._log(f"Resized {W}×{H} → {CANONICAL_W}×{CANONICAL_H}", 'ok')
        return out

    def _try_deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Attempt perspective correction using the ■ anchor squares.
        Only applies the warp if reprojection error < 20px AND first/last
        answer rows are cleanly detected.  Falls back silently otherwise.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape

        sqs = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (80 < area < 3500):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.35 < w / max(h, 1) < 2.8):
                continue
            if y < H * 0.18 or (y + h) > H * 0.82:   # strict: only answer rows
                continue
            roi = bw[y:y+h, x:x+w]
            if roi.size and np.count_nonzero(roi) / roi.size >= 0.55:
                sqs.append({'cx': x + w // 2, 'cy': y + h // 2})

        if len(sqs) < 10:
            self._log(f"Deskew: only {len(sqs)} squares found, skip", 'warn')
            return img

        sqs.sort(key=lambda s: s['cy'])
        rows, cur = [], [sqs[0]]
        for s in sqs[1:]:
            if abs(s['cy'] - cur[-1]['cy']) < 18:
                cur.append(s)
            else:
                rows.append(cur); cur = [s]
        rows.append(cur)

        ans_rows = sorted([r for r in rows if 5 <= len(r) <= 12],
                          key=lambda r: np.mean([s['cy'] for s in r]))
        if len(ans_rows) < 2:
            self._log("Deskew: not enough answer rows, skip", 'warn')
            return img

        first = sorted(ans_rows[0],  key=lambda s: s['cx'])
        last  = sorted(ans_rows[-1], key=lambda s: s['cx'])

        LAST_Y = int(ROW_Y_START + 19 * ROW_SPACING)
        DST = np.float32([[SQ_L[0], ROW_Y_START], [SQ_R[2], ROW_Y_START],
                           [SQ_L[0], LAST_Y],      [SQ_R[2], LAST_Y]])
        SRC = np.float32([[first[0]['cx'], first[0]['cy']],
                          [first[-1]['cx'], first[-1]['cy']],
                          [last[0]['cx'],  last[0]['cy']],
                          [last[-1]['cx'], last[-1]['cy']]])

        M, _ = cv2.findHomography(SRC, DST, cv2.RANSAC, 4.0)
        if M is None:
            self._log("Deskew: homography failed", 'warn')
            return img

        proj = cv2.perspectiveTransform(SRC.reshape(-1, 1, 2), M).reshape(-1, 2)
        err  = float(np.mean(np.linalg.norm(proj - DST, axis=1)))
        if err > 20:
            self._log(f"Deskew: error {err:.1f}px too large, skip", 'warn')
            return img

        warped = cv2.warpPerspective(img, M, (CANONICAL_W, CANONICAL_H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        self._log(f"Deskew applied, error={err:.1f}px", 'ok')
        return warped

    @staticmethod
    def _fill(gray: np.ndarray, cx: int, cy: int) -> float:
        """
        Relative darkness inside bubble vs sheet background.
        Empty ≈ 0.33,  Filled ≈ 1.00
        """
        r = SAMPLE_R
        H, W = gray.shape
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        vals = gray[mask > 0]
        if not len(vals):
            return 0.0
        bg   = float(np.percentile(gray, 90))
        dark = max(0.0, (bg - float(np.mean(vals))) / max(bg, 1.0))
        return min(1.0, dark * 2.8)

    @staticmethod
    def _classify(fills: Dict[str, float]) -> List[str]:
        """
        Absolute threshold 0.50 + dominance ratio 2.5× → single or multi.
        Empty bubbles score ~0.33 so they never cross the threshold.
        """
        max_f = max(fills.values())
        if max_f < 0.50:
            return []
        srt = sorted(fills.items(), key=lambda x: x[1], reverse=True)
        top_v  = srt[0][1]
        sec_v  = srt[1][1] if len(srt) > 1 else 0.0
        if top_v / (sec_v + 1e-6) > 2.5:
            return [srt[0][0]]
        above = [o for o, f in fills.items() if f >= 0.50]
        return above if above else [srt[0][0]]

    def grade(
        self,
        img_raw: np.ndarray,
        answer_key: Dict[int, str],
        pos: float = 3.0,
        neg: float = 1.0,
    ) -> Tuple[OMRResult, np.ndarray]:
        self._logs = []

        # Step 1 — scale to canonical
        img = self._to_canonical(img_raw)

        # Step 2 — optional deskew (only if sheet is skewed)
        img = self._try_deskew(img)

        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canvas = img.copy()   # annotation drawn on the clean canonical image

        results: List[BubbleResult] = []

        for col_idx in range(3):
            for row_idx in range(20):
                q_num = col_idx * 20 + row_idx + 1
                cy    = int(round(ROW_Y_START + row_idx * ROW_SPACING))
                key   = answer_key.get(q_num, '')

                fills    = {o: self._fill(gray, BUBBLE_CX[col_idx][o], cy) for o in 'ABCD'}
                selected = self._classify(fills)

                if not selected:
                    status, score = 'unattempted', 0.0
                elif len(selected) > 1:
                    status, score = 'multi', (-neg if key else 0.0)
                elif key and selected[0] == key:
                    status, score = 'correct', float(pos)
                elif key:
                    status, score = 'wrong', float(-neg)
                else:
                    status, score = 'unattempted', 0.0

                results.append(BubbleResult(
                    q_num=q_num, detected=selected, answer_key=key,
                    status=status, score=score, fill_values=fills,
                ))

                # ── Draw ──────────────────────────────────────────────────
                clr = {'correct':C_CORRECT,'wrong':C_WRONG,
                       'multi':C_MULTI,'unattempted':C_UNATTEMPT}[status]

                for opt in 'ABCD':
                    cx = BUBBLE_CX[col_idx][opt]
                    if opt in selected:
                        cv2.circle(canvas, (cx, cy), DRAW_R, clr, -1)
                        cv2.circle(canvas, (cx, cy), DRAW_R, C_WHITE, 1)
                        cv2.putText(canvas, opt, (cx - 5, cy + 4),
                                    FONT, 0.35, C_WHITE, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, (cx, cy), DRAW_R, C_EMPTY, 1)

                    # Green ring: show where correct answer was, if student was wrong
                    if (key == opt and opt not in selected
                            and status not in ('correct', 'unattempted')):
                        cv2.circle(canvas, (cx, cy), DRAW_R + 4, C_CORR_IND, 2)

        correct = sum(1 for r in results if r.status == 'correct')
        wrong   = sum(1 for r in results if r.status == 'wrong')
        unat    = sum(1 for r in results if r.status == 'unattempted')
        multi   = sum(1 for r in results if r.status == 'multi')
        ps, ns  = correct * pos, (wrong + multi) * neg
        self._log(f"Done: {correct}✓ {wrong}✗ {unat}— {multi}M  → {ps-ns:.1f}", 'ok')

        return (
            OMREngine(bubbles=results, correct=correct, wrong=wrong,
                      unattempted=unat, multi=multi,
                      pos_score=ps, neg_score=ns, total_score=ps - ns,
                      debug_log=list(self._logs)) if isinstance(self, OMREngine) else 
            OMRResult(bubbles=results, correct=correct, wrong=wrong,
                      unattempted=unat, multi=multi,
                      pos_score=ps, neg_score=ns, total_score=ps - ns,
                      debug_log=list(self._logs)),
            canvas,
        )


# ── Session state ─────────────────────────────────────────────────────────────
default_keys_str = "B,C,A,D,A,B,D,C,B,A,C,D,D,A,B,C,C,D,A,B,B,A,D,C,A,C,B,D,D,B,A,C,C,A,D,B,B,D,C,A,A,B,C,D,D,C,B,A,C,A,B,D,B,C,A,D,C,B,A,D"
default_keys = default_keys_str.split(',')
initial_answer_key = {i + 1: default_keys[i] for i in range(60)}

for k, v in {'answer_key': initial_answer_key,
              'result': None, 'annotated_img': None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

engine = OMREngine()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="tricolor"><div></div><div></div><div></div></div>
  <h1 class="htitle">🎓 Yuva Gyan Mahotsav 2026</h1>
  <p class="hsub">Precision OMR Grader v8.0 &nbsp;·&nbsp; Tiranga Yuva Samiti</p>
  <span class="pill pill-g">■ Pixel-Verified Grid</span>&nbsp;
  <span class="pill pill-o">○ Relative Darkness Detection</span>&nbsp;
  <span class="pill pill-b">Clean Aligned Annotation</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Marking Scheme")
    pos_mark = st.number_input("✅ Correct (+)", 0.5, 10.0, 3.0, 0.5)
    neg_mark = st.number_input("❌ Wrong (−)",   0.0,  5.0, 1.0, 0.5)
    st.divider()
    show_debug = st.checkbox("Show debug log", False)
    show_fills = st.checkbox("Show raw fill values", False)
    st.divider()
    st.markdown("### 📋 Answer Key")
    bulk = st.text_area("Paste 60 answers (comma-separated)",
                        placeholder="B,B,B,C,C,A,...", height=70)
    if st.button("✅ Apply Bulk Key", use_container_width=True):
        parts = [p.strip().upper() for p in bulk.split(',')]
        for i, a in enumerate(parts[:60]):
            if a in list('ABCD') + ['']:
                st.session_state.answer_key[i + 1] = a
        st.success("Key applied!")
    opts = ['', 'A', 'B', 'C', 'D']
    st.caption("Or set individually:")
    kc = st.columns(2)
    for q in range(1, 61):
        with kc[0] if q % 2 else kc[1]:
            st.session_state.answer_key[q] = st.selectbox(
                f"Q{q}", opts,
                index=opts.index(st.session_state.answer_key.get(q, '')),
                key=f"k{q}")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sh"><div class="sdot"></div><h3>Upload OMR Sheet</h3></div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="legend">
  <span class="chip ch-g">● Correct</span>
  <span class="chip ch-r">● Wrong</span>
  <span class="chip ch-n">○ Unattempted</span>
  <span class="chip ch-p">● Multi-Mark</span>
  <span class="chip ch-g">◎ Missed correct</span>
</div>""", unsafe_allow_html=True)

accepted = (['pdf','png','jpg','jpeg','tiff','bmp'] if HAS_PDF
            else ['png','jpg','jpeg','tiff','bmp'])
uploaded = st.file_uploader(
    "Drop OMR sheet — JPG / PNG / TIFF" + (" / PDF" if HAS_PDF else ""),
    type=accepted)

if uploaded:
    with st.spinner("Loading…"):
        fb = uploaded.read()
        img_cv = (engine.load_pdf(fb, 150) if uploaded.type == 'application/pdf'
                  else engine.load_img(Image.open(io.BytesIO(fb))))
    st.success(f"✅ **{uploaded.name}** — {img_cv.shape[1]}×{img_cv.shape[0]}px")

    st.markdown("**⚙️ Settings**")
    st.info(f"**Marking:** +{pos_mark:.1f} correct · −{neg_mark:.1f} wrong/multi")
    
    if st.button("🚀  Grade Now", use_container_width=True):
        bar = st.progress(0, text="Scaling to canonical frame…")
        time.sleep(0.05); bar.progress(30, text="Checking skew…")
        time.sleep(0.05); bar.progress(55, text="Sampling bubbles…")

        result, annotated = engine.grade(
            img_cv, st.session_state.answer_key,
            pos=pos_mark, neg=neg_mark)

        bar.progress(95, text="Rendering annotation…")
        st.session_state.result = result
        st.session_state.annotated_img = annotated
        bar.progress(100); time.sleep(0.1); bar.empty()
        st.success("✅ Graded!")

    # ── Side-by-Side Display Logic ────────────────────────────────────────────
    if st.session_state.result is not None and st.session_state.annotated_img is not None:
        st.markdown("---")
        c_orig, c_ann = st.columns(2)
        with c_orig:
            st.markdown("**📄 Original Upload**")
            st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c_ann:
            st.markdown("**🎯 Annotated OMR Sheet**")
            ann = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
            st.image(ann, use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(ann).save(buf, 'PNG')
            st.download_button("⬇️ Download Annotated OMR", buf.getvalue(),
                               "annotated_omr.png", "image/png", use_container_width=True)
    else:
        st.markdown("**📄 Original Upload**")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    result: OMRResult = st.session_state.result
    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Results</h3></div>',
                unsafe_allow_html=True)

    max_s = 60 * pos_mark
    pct   = result.total_score / max_s * 100 if max_s else 0
    if pct >= 75:   bcls,bico,btxt = "rb-ex","🏆","Outstanding!"
    elif pct >= 50: bcls,bico,btxt = "rb-gd","👍","Good Performance"
    elif pct >= 35: bcls,bico,btxt = "rb-av","📚","Average — Keep Practicing"
    else:           bcls,bico,btxt = "rb-pr","⚠️","Needs Improvement"
    bclr = ("#22C55E" if pct>=75 else "#F59E0B" if pct>=50 else "#F97316" if pct>=35 else "#EF4444")

    st.markdown(f"""
    <div class="rbanner {bcls}">
      <span style="font-size:2rem;">{bico}</span>
      <div><div>{btxt}</div>
        <div style="font-size:.83rem;font-weight:400;opacity:.8;margin-top:3px;">
          Score: <strong>{result.total_score:.1f}</strong>/{max_s:.0f} &nbsp;·&nbsp; {pct:.1f}%
        </div>
      </div>
    </div>
    <div class="strack"><div class="sfill"
      style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{bclr},{bclr}88);">
    </div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sgrid">
      <div class="sc c"><div class="snum cg">{result.correct}</div><div class="slbl">Correct</div><div class="gl"></div></div>
      <div class="sc w"><div class="snum cr">{result.wrong}</div><div class="slbl">Wrong</div><div class="gl"></div></div>
      <div class="sc s"><div class="snum ca">{result.unattempted}</div><div class="slbl">Skipped</div><div class="gl"></div></div>
      <div class="sc m"><div class="snum cp">{result.multi}</div><div class="slbl">Multi-Mark</div><div class="gl"></div></div>
      <div class="sc total"><div class="snum co">{result.total_score:.1f}</div><div class="slbl">Net Score</div><div class="gl"></div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📉 Breakdown**")
    att = result.correct + result.wrong + result.multi
    acc = result.correct / att * 100 if att else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positive Score", f"+{result.pos_score:.1f}")
    m2.metric("Negative Score", f"−{result.neg_score:.1f}")
    m3.metric("Net Score", f"{result.total_score:.1f}")
    m4.metric("Accuracy", f"{acc:.1f}%")
    
    info_rows = ''.join(
        f'<div style="display:flex;justify-content:space-between;font-size:.85rem;padding:3px 0;">'
        f'<span>{l}</span><span style="font-weight:700;">{v}</span></div>'
        for l,v in [("Attempted",f"{att}/60 ({att/60*100:.0f}%)"),
                    ("Unattempted",str(result.unattempted)),
                    ("Multi-marked",str(result.multi))])
    st.markdown(f'<div style="background:var(--sf2);border:1px solid var(--br);border-radius:10px;'
                f'padding:13px 16px;margin-top:10px;"><div style="font-size:.68rem;color:var(--mu);'
                f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Summary</div>'
                f'{info_rows}</div>', unsafe_allow_html=True)
                
    if show_debug and result.debug_log:
        html = "".join(f'<div class="{"ok" if "[OK]" in l else "warn" if "[WARN]" in l else "err" if "[ERR]" in l else "info"}">{l}</div>'
                       for l in result.debug_log)
        st.markdown(f'<div class="dlog">{html}</div>', unsafe_allow_html=True)


    st.divider()
    st.markdown('<div class="sh"><div class="sdot"></div><h3>Question-wise Report</h3></div>',
                unsafe_allow_html=True)
    filt = st.multiselect("Filter by status",
                          ['correct','wrong','unattempted','multi'],
                          default=['correct','wrong','unattempted','multi'])
    filtered = [b for b in result.bubbles if b.status in filt]

    rows_html = ""
    for b in filtered:
        det  = ', '.join(b.detected) if b.detected else '—'
        key  = b.answer_key or '—'
        sc   = f"+{b.score:.0f}" if b.score>0 else (f"{b.score:.0f}" if b.score<0 else "0")
        sclr = "cg" if b.score>0 else ("cr" if b.score<0 else "ca")
        bcls = {'correct':'bs-c','wrong':'bs-w','unattempted':'bs-s','multi':'bs-m'}.get(b.status,'')
        bico = {'correct':'✓','wrong':'✗','unattempted':'—','multi':'×'}.get(b.status,'')
        fv_td = (f"<td style='font-size:.69rem;color:var(--mu);font-family:JetBrains Mono,monospace;"
                 f"white-space:nowrap;'>{'  '.join(f'{k}:{v:.2f}' for k,v in sorted(b.fill_values.items()))}</td>"
                 if show_fills else "")
        rows_html += (f"<tr><td>Q{b.q_num:02d}</td>"
                      f"<td><span style='color:#60A5FA;font-weight:700;font-family:JetBrains Mono,monospace;'>{det}</span></td>"
                      f"<td><span style='color:#34D399;font-weight:700;font-family:JetBrains Mono,monospace;'>{key}</span></td>"
                      f"<td><span class='bs {bcls}'>{bico} {b.status.upper()}</span></td>"
                      f"<td class='{sclr}' style='font-weight:800;font-family:JetBrains Mono,monospace;'>{sc}</td>"
                      f"{fv_td}</tr>")

    xth = "<th>Fills A/B/C/D</th>" if show_fills else ""
    st.markdown(
        f'<div style="max-height:540px;overflow-y:auto;border:1px solid var(--br);'
        f'border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.4);">'
        f'<table class="bt"><thead><tr><th>Q#</th><th>Detected</th><th>Key</th>'
        f'<th>Status</th><th>Score</th>{xth}</tr></thead><tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True)

    st.write("")
    export = [{'Q':b.q_num,'Detected':','.join(b.detected) if b.detected else '',
               'Key':b.answer_key,'Status':b.status,'Score':b.score,
               **{f'Fill_{k}':round(v,4) for k,v in b.fill_values.items()}}
              for b in result.bubbles]
    st.download_button("⬇️ Download Full CSV",
                       pd.DataFrame(export).to_csv(index=False),
                       "omr_results.csv", "text/csv")

st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:var(--mu);font-size:.77rem;">
  <div class="tricolor" style="max-width:130px;margin:0 auto 10px;"><div></div><div></div><div></div></div>
  Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti · OMR Grader
</div>""", unsafe_allow_html=True)
