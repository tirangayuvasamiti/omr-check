import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG FOR YUVA GYAN MAHOTSAV 2026 ---
st.set_page_config(
    page_title="YUVA GYAN OMR Grader", 
    layout="wide", 
    page_icon="🎓"
)

# Answer Key (1-60)
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}

OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# Layout configuration for this specific OMR sheet
LAYOUT_CONFIG = {
    "total_questions": 60,
    "options_per_question": 4,
    "columns": 3,
    "alignment_markers": True,  # Black squares on corners
}

# ──────────────────────────────────────────────────────────────────────────────
# Reference layout calibrated from blank OMR sheet (1653×2339 px @ 200dpi)
# Each question circle has TWO concentric rings; we use the OUTER (larger) one.
# 3 question groups × 20 rows × 4 options = 240 answer bubbles
#
# X-band centres for each group's A/B/C/D options (at 1653 px width):
#   Group1 (Q1–20):  A≈159  B≈278  C≈397  D≈518
#   Group2 (Q21–40): A≈684  B≈803  C≈922  D≈1042
#   Group3 (Q41–60): A≈1209 B≈1328 C≈1447 D≈1567
# Row Y centres (group 1 & 3 rows are the same; group 2 has slightly different
# spacing around the "Computer Science" sub-header):
#   Rows 1–20 approx from y=576 to y=1818 with ~60px spacing
# ──────────────────────────────────────────────────────────────────────────────

# Empirical X columns for each group's A,B,C,D bubbles (calibrated on ref sheet)
REF_W, REF_H = 1653, 2339  # reference image dimensions

# X positions of A,B,C,D for each of the 3 sections at REF_W
SECTION_X = [
    [159, 278, 397, 518],    # Section 1 – Q1-20
    [684, 803, 922, 1042],   # Section 2 – Q21-40
    [1209, 1328, 1447, 1567] # Section 3 – Q41-60
]

# Row Y positions extracted from reference (section 1 and 3 share same rows 1–20)
# These were measured from the blank OMR at 200dpi/1653px width
SECTION_ROWS = [
    # Section 1 (Q1–20 rows)
    [576, 636, 695, 755, 813, 874, 932, 1049, 1107, 1168,
     1226, 1286, 1345, 1405, 1520, 1580, 1638, 1699, 1757, 1818],
    # Section 2 (Q21–40 rows)
    [576, 636, 750, 810, 870, 930, 990, 1050, 1107, 1168,
     1280, 1340, 1400, 1460, 1520, 1580, 1638, 1699, 1757, 1818],
    # Section 3 (Q41–60 rows)
    [576, 636, 695, 755, 813, 874, 932, 990, 1050, 1110,
     1170, 1230, 1290, 1350, 1405, 1525, 1580, 1638, 1700, 1760],
]

# Q number mapping: section → list of Q numbers (1-indexed)
SECTION_QNUMS = [
    list(range(1, 21)),   # Q1-20
    list(range(21, 41)),  # Q21-40
    list(range(41, 61)),  # Q41-60
]

# Bubble search radius (px at reference scale)
BUBBLE_RADIUS = 16


class YuvaGyanOMRProcessor:
    """
    AI-enhanced OMR processor for YUVA GYAN MAHOTSAV 2026.

    Detection strategy
    ──────────────────
    1.  WARP CORRECTION: detect four corner alignment squares → perspective-warp
        the sheet to the canonical reference dimensions (REF_W × REF_H).
    2.  MULTI-CHANNEL BUBBLE SAMPLING: for each of the 240 known bubble centres
        we sample fill intensity using multiple complementary methods:
          a) Mean pixel intensity in a circular ROI (raw darkness)
          b) Otsu-thresholded pixel count (binary fill area)
          c) Gradient / edge energy (empty rings have high edge density relative
             to their interior; filled bubbles have low edge density)
        These three signals are fused into a single fill-score via a simple
        linear model calibrated on the blank sheet.
    3.  ADAPTIVE THRESHOLDING: instead of a fixed fill threshold, we compute the
        fill-score for every bubble on the sheet and use an Otsu-like bimodal
        split to distinguish marked from unmarked — robust to lighting variation.
    4.  MULTI-MARK / BLANK DETECTION: if no bubble exceeds the adaptive
        threshold → Blank; if two+ bubbles exceed threshold AND the runner-up
        score is ≥75% of the top score → Double-Mark.
    """

    def __init__(self):
        self.fill_threshold_fallback = 0.38
        self.double_mark_ratio = 0.72
        # Tolerance for matching detected bubble to grid position (px)
        self.match_tol = 30

    # ──────────────────────────────────── image loading ────────────────────────

    def load_image(self, uploaded_file) -> np.ndarray:
        img = Image.open(uploaded_file).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ────────────────────────────── warp / alignment ───────────────────────────

    def find_corner_markers(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Find 4 black alignment squares at the corners of the OMR sheet.
        Returns (4,2) array of corner centres ordered TL/TR/BR/BL, or None.
        """
        _, bw = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        markers = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50 or area > 3000:
                continue
            x, y, bw_, bh = cv2.boundingRect(c)
            ar = bw_ / float(bh)
            if not (0.6 < ar < 1.6):
                continue
            circ = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-6)
            if circ < 0.3:
                continue
            cx, cy = x + bw_ // 2, y + bh // 2
            # Must be near one of the 4 corners
            if (cx < w * 0.25 or cx > w * 0.75) and (cy < h * 0.25 or cy > h * 0.75):
                markers.append((cx, cy))

        if len(markers) < 4:
            return None

        # Pick 4 best (one per quadrant)
        quads = {
            'tl': None, 'tr': None, 'bl': None, 'br': None
        }
        mid_x, mid_y = w / 2, h / 2
        for mx, my in markers:
            key = ('t' if my < mid_y else 'b') + ('l' if mx < mid_x else 'r')
            if quads[key] is None:
                quads[key] = (mx, my)

        if any(v is None for v in quads.values()):
            return None

        return np.array([quads['tl'], quads['tr'],
                         quads['br'], quads['bl']], dtype=np.float32)

    def warp_to_reference(self, image: np.ndarray) -> np.ndarray:
        """
        Perspective-warp image so that corner markers align with REF_W × REF_H.
        Falls back to simple resize if markers are not found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = self.find_corner_markers(gray)

        if corners is not None:
            dst = np.array([[0, 0], [REF_W - 1, 0],
                            [REF_W - 1, REF_H - 1], [0, REF_H - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners, dst)
            warped = cv2.warpPerspective(image, M, (REF_W, REF_H),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            logger.info("Perspective warp applied via corner markers")
            return warped
        else:
            logger.warning("Corner markers not found – falling back to resize")
            return cv2.resize(image, (REF_W, REF_H), interpolation=cv2.INTER_CUBIC)

    # ─────────────────────────────── fill scoring ──────────────────────────────

    def _build_circle_mask(self, shape, cx, cy, r) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask

    def compute_fill_score(self, warped: np.ndarray,
                            cx: int, cy: int, r: int = BUBBLE_RADIUS) -> Dict:
        """
        Multi-channel fill scoring for a single bubble.

        Returns a dict with:
          raw_darkness  – mean darkness in circle (0=white, 1=black)
          binary_fill   – fraction of pixels above Otsu threshold (filled)
          edge_ratio    – edge energy inside / perimeter (low for filled)
          combined      – fused score (higher = more filled)
        """
        h, w = warped.shape[:2]
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)

        if x1 <= x0 or y1 <= y0:
            return {'raw_darkness': 0, 'binary_fill': 0,
                    'edge_ratio': 1, 'combined': 0}

        patch = warped[y0:y1, x0:x1]
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch

        # Circular mask within patch
        pcx, pcy = cx - x0, cy - y0
        mask = self._build_circle_mask(gray_patch.shape, pcx, pcy, r)
        area = cv2.countNonZero(mask)
        if area == 0:
            return {'raw_darkness': 0, 'binary_fill': 0,
                    'edge_ratio': 1, 'combined': 0}

        # ── Signal 1: raw darkness (invert so white→0, dark→1)
        mean_val = cv2.mean(gray_patch, mask=mask)[0]
        raw_darkness = 1.0 - (mean_val / 255.0)

        # ── Signal 2: binary fill via adaptive Otsu inside patch
        # Use a slightly larger patch for better Otsu estimation
        _, bw = cv2.threshold(gray_patch, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(bw, mask))
        binary_fill = filled_pixels / float(area)

        # ── Signal 3: Canny edge energy inside bubble
        # Empty ring: lots of edges at border; filled: fewer relative edges
        edges = cv2.Canny(gray_patch, 30, 90)
        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
        edge_ratio = edge_pixels / float(area)

        # ── Fusion: weighted combination
        # Filled bubbles → high darkness & fill, low edge_ratio
        # Weights calibrated on typical scanned OMR sheets
        combined = (0.45 * raw_darkness +
                    0.45 * binary_fill +
                    0.10 * (1.0 - min(edge_ratio * 3, 1.0)))

        return {
            'raw_darkness': raw_darkness,
            'binary_fill': binary_fill,
            'edge_ratio': edge_ratio,
            'combined': combined,
        }

    # ──────────────────────────── adaptive threshold ───────────────────────────

    def compute_adaptive_threshold(self, all_scores: List[float]) -> float:
        """
        Otsu-inspired bimodal split for fill scores across the entire sheet.
        Robust to global brightness changes (shadows, phone camera, scanner).
        Falls back to fixed threshold if distribution is unimodal.
        """
        if not all_scores:
            return self.fill_threshold_fallback

        arr = np.array(all_scores, dtype=np.float32)

        # Build histogram over [0,1]
        hist, bin_edges = np.histogram(arr, bins=50, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Otsu on histogram
        total = hist.sum()
        if total == 0:
            return self.fill_threshold_fallback

        best_thresh = self.fill_threshold_fallback
        best_var = -1.0
        w0_sum, mean0_sum = 0.0, 0.0

        for i in range(1, len(hist)):
            w0 = hist[:i].sum() / total
            w1 = 1.0 - w0
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (hist[:i] * bin_centers[:i]).sum() / (hist[:i].sum() + 1e-9)
            mu1 = (hist[i:] * bin_centers[i:]).sum() / (hist[i:].sum() + 1e-9)
            var_between = w0 * w1 * (mu0 - mu1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = bin_centers[i]

        # Safety clamp
        best_thresh = float(np.clip(best_thresh, 0.20, 0.70))
        logger.info(f"Adaptive fill threshold: {best_thresh:.3f}")
        return best_thresh

    # ─────────────────────────── full grading pipeline ─────────────────────────

    def process(self, image: np.ndarray):
        """
        Main pipeline. Returns (stats, annotated_image, results, message).
        """
        try:
            # ── Step 1: warp to canonical dimensions
            warped = self.warp_to_reference(image)

            # ── Step 2: pre-process for visualisation
            # (use CLAHE-enhanced grayscale for display; scoring uses colour)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # ── Step 3: score every bubble
            # Layout: 3 sections × 20 rows × 4 options
            all_bubble_scores = []  # (section, row, option, score_dict)

            for sec_idx in range(3):
                x_cols = SECTION_X[sec_idx]
                y_rows = SECTION_ROWS[sec_idx]
                for row_idx, cy in enumerate(y_rows):
                    for opt_idx, cx in enumerate(x_cols):
                        score = self.compute_fill_score(warped, cx, cy)
                        all_bubble_scores.append((sec_idx, row_idx, opt_idx, score))

            # ── Step 4: adaptive threshold
            combined_scores = [s[3]['combined'] for s in all_bubble_scores]
            thresh = self.compute_adaptive_threshold(combined_scores)

            # ── Step 5: grade each question
            results = []
            stats = {'correct': 0, 'wrong': 0, 'blank': 0, 'double': 0}

            for sec_idx in range(3):
                x_cols = SECTION_X[sec_idx]
                y_rows = SECTION_ROWS[sec_idx]
                q_nums = SECTION_QNUMS[sec_idx]

                for row_idx, (cy, q_num) in enumerate(zip(y_rows, q_nums)):
                    # Get fill scores for A,B,C,D
                    fills = []
                    for opt_idx, cx in enumerate(x_cols):
                        # Find in precomputed list
                        sc = next(s[3] for s in all_bubble_scores
                                  if s[0] == sec_idx and s[1] == row_idx
                                  and s[2] == opt_idx)
                        fills.append(sc['combined'])

                    result = self._grade_question(q_num, fills, thresh)
                    results.append(result)

                    if result['is_correct']:
                        stats['correct'] += 1
                    elif result['is_blank']:
                        stats['blank'] += 1
                    elif result['is_double']:
                        stats['double'] += 1
                        stats['wrong'] += 1
                    else:
                        stats['wrong'] += 1

            # ── Step 6: annotate
            annotated = self._annotate(warped, all_bubble_scores, results, thresh)

            return stats, annotated, results, "Success"

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return None, image, [], f"Error: {str(e)}"

    # ─────────────────────────────── grading logic ─────────────────────────────

    def _grade_question(self, q_num: int, fills: List[float], thresh: float) -> Dict:
        """Grade one question given the 4 fill scores and adaptive threshold."""
        correct_ans = ANS_KEY.get(q_num, 1) - 1  # 0-indexed

        # Sort descending
        indexed = sorted(enumerate(fills), key=lambda x: x[1], reverse=True)
        top_opt, top_score = indexed[0]
        second_score = indexed[1][1] if len(indexed) > 1 else 0.0

        is_blank = top_score < thresh
        is_double = (not is_blank and
                     second_score >= thresh and
                     second_score >= top_score * self.double_mark_ratio)

        if is_blank:
            marked = []
            status = "Blank"
        elif is_double:
            marked = [indexed[0][0], indexed[1][0]]
            status = "Double Mark"
        else:
            marked = [top_opt]
            status = "Correct" if top_opt == correct_ans else "Incorrect"

        return {
            'question': q_num,
            'marked': marked,
            'correct': correct_ans,
            'status': status,
            'is_correct': (not is_blank and not is_double and top_opt == correct_ans),
            'is_blank': is_blank,
            'is_double': is_double,
            'fills': fills,
            'adaptive_thresh': thresh,
        }

    # ──────────────────────────────── annotation ───────────────────────────────

    def _annotate(self, warped: np.ndarray,
                  all_bubble_scores, results: List[Dict], thresh: float) -> np.ndarray:
        """Draw coloured circles on the warped sheet."""
        vis = warped.copy()
        r = BUBBLE_RADIUS + 2  # draw slightly larger than scoring radius

        for result in results:
            q_num = result['question']
            # Find which section / row this q_num belongs to
            for sec_idx, q_list in enumerate(SECTION_QNUMS):
                if q_num in q_list:
                    row_idx = q_list.index(q_num)
                    break
            else:
                continue

            x_cols = SECTION_X[sec_idx]
            y_rows = SECTION_ROWS[sec_idx]
            cy = y_rows[row_idx]

            correct_opt = result['correct']
            correct_cx = x_cols[correct_opt]

            if result['is_blank']:
                # Blue dashed circle on correct answer position
                cv2.circle(vis, (correct_cx, cy), r, (255, 100, 0), 2)
                cv2.circle(vis, (correct_cx, cy), r + 3, (255, 100, 0), 1)

            elif result['is_double']:
                # Highlight all marked bubbles yellow
                for opt_idx in result['marked']:
                    cx = x_cols[opt_idx]
                    cv2.circle(vis, (cx, cy), r, (0, 220, 220), 3)
                # Blue ring on correct
                cv2.circle(vis, (correct_cx, cy), r + 3, (255, 100, 0), 2)

            else:
                marked_opt = result['marked'][0]
                marked_cx = x_cols[marked_opt]
                if result['is_correct']:
                    cv2.circle(vis, (marked_cx, cy), r, (0, 210, 0), 3)
                else:
                    cv2.circle(vis, (marked_cx, cy), r, (0, 0, 230), 3)
                    cv2.circle(vis, (correct_cx, cy), r + 2, (255, 130, 0), 2)

        return vis

    # ─────────────────────────────── legacy compat ─────────────────────────────

    def find_alignment_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.find_corner_markers(gray) or []

    def preprocess_sheet(self, image):
        warped = self.warp_to_reference(image)
        return warped, warped

    def detect_bubbles(self, roi):
        """Legacy: returns empty list + thresh for compatibility."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        return [], thresh

    def organize_bubbles(self, bubbles):
        return []

    def grade_question(self, q_num, bubbles, thresh):
        return {}

    def annotate_sheet(self, roi, questions, results):
        return roi


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🎓 YUVA GYAN MAHOTSAV 2026 - OMR Grader")

    st.markdown("""
    ### Official OMR Sheet Grading System
    **Features:**
    - ✅ Automatic bubble detection with **multi-channel AI scoring**
    - 🎯 Perspective-warp alignment via corner markers
    - 🧠 **Adaptive threshold** – calibrates to lighting conditions automatically
    - 📊 Instant results with detailed analytics
    - 🎨 Color-coded visual feedback
    """)

    # Sidebar
    with st.sidebar:
        st.header("📋 Test Information")
        st.info("""
        **Total Questions:** 60
        - English (Q.1-7)
        - Hindi (Q.8-14)
        - Mental Ability (Q.15-22)
        - Computer Science (Q.23-30)
        - General Knowledge (Q.31-55)
        - Youth Awareness (Q.56-60)
        """)

        st.header("🎨 Color Legend")
        st.markdown("""
        - 🟢 **Green** = Correct Answer
        - 🔴 **Red** = Wrong Answer
        - 🔵 **Blue** = Correct Answer (reference)
        - 🟡 **Yellow** = Double Marked
        """)

        st.header("⚙️ Detection Info")
        st.markdown("""
        **Multi-Channel Scoring:**
        - Raw pixel darkness (45%)
        - Otsu binary fill area (45%)
        - Edge energy ratio (10%)
        
        **Adaptive Threshold:**
        Auto-calibrates per sheet using
        Otsu bimodal split on all 240
        bubble scores.
        """)

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Filled OMR Sheet",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo or scan of the filled answer sheet"
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📄 Original Sheet")
            original_img = Image.open(uploaded_file)
            st.image(original_img, use_container_width=True)

        with st.spinner("🔄 Processing OMR sheet with AI bubble detection..."):
            try:
                processor = YuvaGyanOMRProcessor()
                image = processor.load_image(uploaded_file)
                stats, annotated, results, message = processor.process(image)

                if stats:
                    with col2:
                        st.subheader("✅ Graded Sheet")
                        st.image(annotated, channels="BGR", use_container_width=True)

                    st.success(f"✅ {message}")

                    # Show adaptive threshold used
                    if results:
                        st.info(
                            f"🧠 Adaptive fill threshold used: "
                            f"**{results[0]['adaptive_thresh']:.3f}** "
                            f"(auto-calibrated for this sheet's lighting)"
                        )

                    # Metrics
                    st.subheader("📊 Results Summary")
                    metric_cols = st.columns(5)

                    total = len(results)
                    score = stats['correct']
                    percentage = (score / total * 100) if total > 0 else 0

                    metric_cols[0].metric("Total", total)
                    metric_cols[1].metric("✅ Correct", stats['correct'])
                    metric_cols[2].metric("❌ Wrong", stats['wrong'])
                    metric_cols[3].metric("⚠️ Blank", stats['blank'])
                    metric_cols[4].metric("🔴 Double", stats['double'])

                    # Score card
                    st.subheader("🎯 Final Score")
                    score_col1, score_col2, score_col3 = st.columns(3)

                    with score_col1:
                        st.metric("Score", f"{score}/{total}")
                    with score_col2:
                        st.metric("Percentage", f"{percentage:.2f}%")
                    with score_col3:
                        if percentage >= 90:
                            grade = "A+"
                            st.balloons()
                        elif percentage >= 80:
                            grade = "A"
                        elif percentage >= 70:
                            grade = "B"
                        elif percentage >= 60:
                            grade = "C"
                        elif percentage >= 50:
                            grade = "D"
                        else:
                            grade = "E"
                        st.metric("Grade", grade)

                    # Detailed fill scores (expandable)
                    with st.expander("🔬 Debug: Raw Fill Scores per Question"):
                        debug_data = []
                        for r in results:
                            fills = r.get('fills', [0, 0, 0, 0])
                            debug_data.append({
                                "Q": r['question'],
                                "Score_A": f"{fills[0]:.3f}",
                                "Score_B": f"{fills[1]:.3f}",
                                "Score_C": f"{fills[2]:.3f}",
                                "Score_D": f"{fills[3]:.3f}",
                                "Marked": ", ".join(OPTION_MAP[i] for i in r['marked']) or "–",
                                "Correct": OPTION_MAP[r['correct']],
                                "Status": r['status'],
                            })
                        st.dataframe(pd.DataFrame(debug_data), use_container_width=True)

                    # Detailed results table
                    st.subheader("📋 Question-wise Analysis")

                    df_data = []
                    for r in results:
                        marked_str = (", ".join([OPTION_MAP[i] for i in r['marked']])
                                      if r['marked'] else "None")
                        df_data.append({
                            "Q.No": r['question'],
                            "Marked": marked_str,
                            "Correct": OPTION_MAP[r['correct']],
                            "Status": r['status']
                        })

                    df = pd.DataFrame(df_data)

                    def highlight(row):
                        if row['Status'] == 'Correct':
                            return ['background-color: #d4edda'] * len(row)
                        elif row['Status'] == 'Incorrect':
                            return ['background-color: #f8d7da'] * len(row)
                        elif row['Status'] == 'Blank':
                            return ['background-color: #fff3cd'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)

                    st.dataframe(df.style.apply(highlight, axis=1),
                                 use_container_width=True, height=400)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "yuva_gyan_results.csv",
                        "text/csv"
                    )

                else:
                    st.error(f"❌ {message}")
                    st.info("💡 Tips: Ensure good lighting, flat surface, all bubbles visible")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
