import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Enterprise Grader", layout="wide", page_icon="📝")

# --- GRADING CONSTANTS ---
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

# --- UTILITY FUNCTIONS ---

def order_points(pts):
    """
    Orders coordinates: top-left, top-right, bottom-right, bottom-left.
    Essential for correct perspective wrapping.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def four_point_transform(image, pts):
    """
    Mathematically flattens the image to a 'bird's eye view'.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def sort_contours(cnts, method="left-to-right"):
    """
    Sorts contours based on spatial coordinates.
    """
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

# --- CORE PIPELINE ---

def process_omr(image_file, show_missed=True):
    # 1. Load & Preprocess
    img_pil = Image.open(image_file).convert('RGB')
    image = np.array(img_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize for consistent processing
    h_orig, w_orig = image.shape[:2]
    # We maintain aspect ratio but ensure it's large enough for contour detection
    scaling_factor = 1600 / h_orig
    image = cv2.resize(image, (int(w_orig * scaling_factor), 1600))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Find Document Corners (The Paper)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    # Fallback to image corners if no paper edge found
    if docCnt is None:
        h, w = image.shape[:2]
        docCnt = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # 3. Perspective Transform (Flatten)
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped_gray = four_point_transform(gray, docCnt.reshape(4, 2))
    
    # 4. Bubble Detection
    # Otsu's thresholding automatically finds the best separation between ink and paper
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    questionCnts = []
    # Heuristics for the provided OMR Layout
    # Based on standard A4 OMR, bubbles are small circles.
    paper_h, paper_w = paper.shape[:2]
    min_area = (paper_w * paper_h) * 0.0001 # approx size of a bubble
    max_area = (paper_w * paper_h) * 0.002
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        # Filter: Aspect ratio ~1.0 (Square/Circle) and size check
        if w >= 18 and h >= 18 and 0.8 <= ar <= 1.2:
            if min_area < cv2.contourArea(c) < max_area:
                # IMPORTANT: Exclude the top header (Roll No/Test ID)
                # In standard OMRs, questions usually start after the top 20%
                if y > paper_h * 0.20: 
                    questionCnts.append(c)

    # 5. Grid Validation
    # We expect exactly 60 Questions * 4 Options = 240 Bubbles.
    # If we find slightly more, we take the 240 most "circular" ones.
    if len(questionCnts) != 240:
        # Emergency heuristic: sort by y position (top-down) and take the most relevant ones
        # or show error.
        if len(questionCnts) > 240:
             # Sort by Y position to roughly keep structure, but this is risky without exact match.
             # Better to sort by Area deviation from median area
             areas = [cv2.contourArea(c) for c in questionCnts]
             median_area = np.median(areas)
             # Keep contours closest to median area
             questionCnts = sorted(questionCnts, key=lambda c: abs(cv2.contourArea(c) - median_area))[:240]
        else:
            return None, paper, None, f"Error: Detected {len(questionCnts)} bubbles. Expected 240. Ensure lighting is even."

    # 6. Sorting Logic (The Math Part)
    # Sort top-to-bottom? No, OMRs usually have columns.
    # Strategy: Sort all by X coordinate first to separate the 3 columns.
    
    questionCnts = sort_contours(questionCnts, method="left-to-right")[0]
    
    # Split into 3 columns (80 bubbles each)
    col_chunks = [questionCnts[i:i + 80] for i in range(0, 240, 80)]
    
    grading_results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown = []
    global_q_idx = 1
    
    for col in col_chunks:
        # Within a column, sort Top-to-Bottom (Rows)
        col = sort_contours(col, method="top-to-bottom")[0]
        
        # Process rows (4 bubbles per row)
        for i in range(0, 80, 4):
            row = col[i:i+4]
            # Sort Row Left-to-Right (A, B, C, D)
            row = sort_contours(row, method="left-to-right")[0]
            
            # Identify the marked bubble
            bubbled = None
            detected_indices = []
            
            # Local thresholding for marking detection
            fill_values = []
            
            for (j, c) in enumerate(row):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                
                # Apply mask to thresholded image to count white pixels (ink)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                fill_values.append((total, j, c))
            
            # Dynamic threshold for "marked": Must be > 50% of the max filled bubble in this row
            # AND absolute pixel count must be significant.
            fill_values.sort(key=lambda x: x[0], reverse=True)
            max_fill = fill_values[0][0]
            
            # Threshold: bubble is marked if it has > 500 pixels (tuned for 1600px height)
            # and is at least 85% as dark as the darkest bubble
            for (fill, idx, cnt) in fill_values:
                if fill > 400 and fill > (max_fill * 0.9): 
                    detected_indices.append(idx)

            correct_idx = ANS_KEY.get(global_q_idx) - 1
            status = ""
            selected_txt = ""

            # Drawing Colors (BGR)
            COLOR_CORRECT = (0, 255, 0)
            COLOR_WRONG = (0, 0, 255) # Red
            COLOR_MISSED = (255, 0, 0) # Blue (as requested)
            
            (x, y, w, h) = cv2.boundingRect(row[correct_idx])

            if len(detected_indices) == 0:
                grading_results["blank"] += 1
                status = "Blank"
                selected_txt = "-"
                if show_missed:
                    cv2.rectangle(paper, (x, y), (x+w, y+h), COLOR_MISSED, 2)
                    
            elif len(detected_indices) > 1:
                grading_results["double"] += 1
                grading_results["wrong"] += 1
                status = "Double"
                selected_txt = "Multi"
                for idx in detected_indices:
                    (bx, by, bw, bh) = cv2.boundingRect(row[idx])
                    cv2.rectangle(paper, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2) # Yellow for double
                if show_missed:
                    cv2.rectangle(paper, (x, y), (x+w, y+h), COLOR_MISSED, 2)
                    
            else:
                k = detected_indices[0]
                selected_txt = OPTS[k]
                (bx, by, bw, bh) = cv2.boundingRect(row[k])
                
                if k == correct_idx:
                    grading_results["correct"] += 1
                    status = "Correct"
                    cv2.rectangle(paper, (bx, by), (bx+bw, by+bh), COLOR_CORRECT, 3)
                else:
                    grading_results["wrong"] += 1
                    status = "Incorrect"
                    cv2.rectangle(paper, (bx, by), (bx+bw, by+bh), COLOR_WRONG, 3)
                    if show_missed:
                         cv2.rectangle(paper, (x, y), (x+w, y+h), COLOR_MISSED, 2)

            breakdown.append({
                "Q": global_q_idx,
                "Correct": OPTS[correct_idx],
                "Marked": selected_txt,
                "Status": status
            })
            global_q_idx += 1

    return grading_results, paper, breakdown, "Success"

# --- UI LOGIC ---

st.sidebar.title("⚙️ Grader Settings")
st.sidebar.markdown("""
**Grading Rules:**
* Correct: +3
* Wrong: -1
* Blank: 0
""")
show_missed = st.sidebar.checkbox("Highlight Missed Answers (Blue)", value=True)

st.title("🏆 Yuva Gyan Enterprise Grader")
st.markdown("Automated 60-Question OMR Processor (Math-Based)")

uploaded_file = st.file_uploader("Upload OMR Sheet (Image)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    with st.spinner("Processing Geometry & Calculating Scores..."):
        results, img_out, data, msg = process_omr(uploaded_file, show_missed)
        
        if results is None:
            st.error(msg)
            st.image(img_out, caption="Debug View (Paper Detection)", use_container_width=True)
        else:
            # Calc Score
            score = (results['correct'] * CORRECT_PTS) - (results['wrong'] * WRONG_PTS)
            acc = (results['correct'] / 60) * 100
            
            # 1. Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Score", score)
            c2.metric("Accuracy", f"{acc:.1f}%")
            c3.metric("Correct", results['correct'])
            c4.metric("Incorrect / Double", results['wrong'])
            
            # 2. Main Visual
            st.image(img_out, caption="Graded OMR Sheet", use_container_width=True)
            
            # 3. Data
            with st.expander("Detailed Question Breakdown"):
                df = pd.DataFrame(data)
                
                def highlight_status(val):
                    color = 'black'
                    if val == 'Correct': color = 'green'
                    elif val == 'Incorrect': color = 'red'
                    elif val == 'Double': color = 'orange'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df.style.map(highlight_status, subset=['Status']), use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "omr_report.csv", "text/csv")
