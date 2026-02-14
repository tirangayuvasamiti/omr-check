import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Precision Grader", layout="wide", page_icon="🎯")

# --- CONSTANTS ---
CORRECT_PTS = 3
WRONG_PTS = 1
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}
COLORS = {
    'green': (0, 200, 0),    # Correct
    'red': (0, 0, 255),      # Wrong
    'blue': (255, 0, 0),     # Missed
    'yellow': (0, 255, 255), # Double
    'grid': (200, 200, 200)  # Visual Helper
}

def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_document_corners(image):
    """
    Robust Corner Finder: Uses dilation to connect broken lines.
    Falls back to full image if corners are not found.
    """
    # Resize for processing speed
    h_process = 1000
    ratio = image.shape[0] / float(h_process)
    process_img = imutils.resize(image, height=h_process)
    
    gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # DILATION: Critical for faint/broken border lines
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break
            
    if doc_cnt is None:
        return np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype="float32")
    
    return doc_cnt.reshape(4, 2) * ratio

def process_roi_grader(image, show_missed=True):
    # 1. Flatten the Page
    corners = get_document_corners(image)
    warped = four_point_transform(image, corners)
    
    # 2. Standardize Full Page Size (1200x1600)
    # We do this to ensure our "Cut Coordinates" are always accurate
    warped = cv2.resize(warped, (1200, 1600))
    
    # 3. ROI EXTRACTION (The "Surgical" Crop)
    # We cut out the top header (Instructions) and bottom footer (Official Use)
    # Based on the Yuva Gyan template:
    # Top Cut: Y=360 (Just below "Correct: Wrong:")
    # Bottom Cut: Y=1540 (Just above "For Official Use Only")
    CROP_TOP = 360
    CROP_BOTTOM = 1540
    
    # Create the ROI image
    roi_img = warped[CROP_TOP:CROP_BOTTOM, 0:1200]
    roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # 4. MATH GRID (Relative to the ROI, not the full page)
    # Now (0,0) is the top-left of the cropped bubble zone.
    # Total Height of ROI is approx 1180px
    
    # Grid Calibration
    ROWS = 20
    COLS = 3
    COL_X_START = [180, 560, 940]  # X coordinates for Col 1, 2, 3
    ROW_Y_START = 20               # Start near the top of the crop
    ROW_GAP = 58.5                 # Vertical distance between rows
    OPT_GAP = 55                   # Horizontal distance between bubbles
    BUBBLE_SIZE = 42
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown = []
    
    def draw_roi(box, color, thick=3):
        cv2.rectangle(roi_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, thick)

    # 5. Grading Loop
    for q_abs in range(1, 61):
        # Determine Grid Position
        if q_abs <= 20:
            col_idx, row_idx = 0, q_abs - 1
        elif q_abs <= 40:
            col_idx, row_idx = 1, q_abs - 21
        else:
            col_idx, row_idx = 2, q_abs - 41
            
        base_x = COL_X_START[col_idx]
        base_y = int(ROW_Y_START + (row_idx * ROW_GAP))
        
        bubbles = []
        scores = []
        
        for i in range(4):
            bx = base_x + (i * OPT_GAP)
            by = base_y
            
            # --- AI PIXEL DENSITY CHECK ---
            # We focus on the center 40% of the bubble to avoid edge noise
            # Thresholding handles the "Blue/Black Pen" requirement
            cell = roi_gray[by:by+BUBBLE_SIZE, bx:bx+BUBBLE_SIZE]
            mask = np.zeros(cell.shape, dtype="uint8")
            cv2.circle(mask, (BUBBLE_SIZE//2, BUBBLE_SIZE//2), int(BUBBLE_SIZE*0.40), 255, -1)
            
            mean_val = cv2.mean(cell, mask=mask)[0]
            scores.append((mean_val, i))
            bubbles.append((bx, by, BUBBLE_SIZE, BUBBLE_SIZE))
            
            # Draw Grid (Debug)
            cv2.rectangle(roi_img, (bx, by), (bx+BUBBLE_SIZE, by+BUBBLE_SIZE), COLORS['grid'], 1)

        # Sort: Darkest bubble first (lowest pixel value)
        scores.sort(key=lambda x: x[0])
        best_val, best_idx = scores[0]
        lightest_val = scores[-1][0]
        
        # --- ROBUST DECISION LOGIC ---
        # A mark is valid ONLY if it is < 82% brightness of the empty paper
        marked = [idx for val, idx in scores if val < (lightest_val * 0.82)]
        
        correct_ans = ANS_KEY[q_abs] - 1
        status = ""
        
        if len(marked) == 0:
            status = "Blank"
            results['blank'] += 1
            if show_missed:
                draw_roi(bubbles[correct_ans], COLORS['blue'], 2)
                
        elif len(marked) > 1:
            status = "Double"
            results['double'] += 1
            results['wrong'] += 1
            for idx in marked:
                draw_roi(bubbles[idx], COLORS['yellow'], 3)
                
        else:
            student_ans = marked[0]
            if student_ans == correct_ans:
                status = "Correct"
                results['correct'] += 1
                draw_roi(bubbles[student_ans], COLORS['green'], 3)
            else:
                status = "Incorrect"
                results['wrong'] += 1
                draw_roi(bubbles[student_ans], COLORS['red'], 3)
                if show_missed:
                    draw_roi(bubbles[correct_ans], COLORS['blue'], 2)
                    
        breakdown.append({"Q": q_abs, "Status": status, "Marked": OPTS.get(marked[0], "-") if len(marked)==1 else "Multi/None"})

    return results, roi_img, breakdown

# --- STREAMLIT UI ---
st.title("🎯 Precision ROI Grader")
st.markdown("**Scanning Zone:** Below 'INSTRUCTIONS' ➡ Above 'OFFICIAL USE ONLY'")



col1, col2 = st.columns([1, 3])
with col1:
    st.write("### ⚙️ Settings")
    show_missed = st.checkbox("Show Correct Answers", value=True)
    st.info("This mode creates a 'Surgical Crop' of the bubble area to eliminate header/footer noise.")

with col2:
    uploaded_file = st.file_uploader("Upload OMR (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("Isolating Bubble Zone & Grading..."):
        try:
            img = load_image(uploaded_file)
            stats, processed_roi, logs = process_roi_grader(img, show_missed)
            
            # SCORING
            score = (stats['correct'] * CORRECT_PTS) - (stats['wrong'] * WRONG_PTS)
            
            st.write("### 📊 Grading Report")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("✅ Correct", stats['correct'])
            m2.metric("❌ Incorrect", stats['wrong'])
            m3.metric("⚠️ Blank/Double", stats['blank'] + stats['double'])
            m4.metric("🏆 Total Score", score)
            
            st.divider()
            
            t1, t2 = st.tabs(["🖼️ Scanned ROI View", "📝 Question Data"])
            
            with t1:
                st.write("**Visual Confirmation:** The image below is the *exact* cropped area the AI analyzed.")
                st.image(processed_roi, channels="BGR", use_container_width=True)
                
            with t2:
                df = pd.DataFrame(logs)
                
                def color_rows(row):
                    s = row['Status']
                    if s == 'Correct': return ['background-color: #d4edda'] * len(row)
                    if s == 'Incorrect': return ['background-color: #f8d7da'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(df.style.apply(color_rows, axis=1), use_container_width=True, height=500)
                
        except Exception as e:
            st.error(f"Processing Error: {e}")
