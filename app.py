import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours as imutils_contours
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Grader Pro", layout="wide", page_icon="📝")

# --- SCORING RULES ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted
EXPECTED_BUBBLES = 240

# --- VERIFIED ANSWER KEY ---
# NORMAL FORMAT: A=1, B=2, C=3, D=4
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}

def get_warp_points(gray, mode):
    """Finds the 4 anchor points of the document using the chosen method."""
    
    # METHOD 1: Page Edges (Standard)
    def scan_page_edges():
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # Must be 4 points and reasonably large
                if len(approx) == 4 and cv2.contourArea(approx) > 50000:
                    return approx.reshape(4, 2)
        return None

    # METHOD 2: Corner Marks (Fiducials)
    def scan_corner_marks():
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        markers = []
        for c in cnts:
            area = cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if 100 < area < 15000 and 0.5 <= ar <= 1.5:
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) > 0:
                    solidity = area / float(cv2.contourArea(hull))
                    if solidity > 0.6:  # Mostly solid shape
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            markers.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
        
        if len(markers) >= 4:
            pts = np.array(markers)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            return np.array([tl, tr, br, bl], dtype="float32")
        return None

    # Logic based on user selection
    if mode == "Page Edges (Standard)":
        return scan_page_edges()
    elif mode == "Corner Marks (Fiducials)":
        return scan_corner_marks()
    else:  # Auto (Try Edges first, then Marks)
        pts = scan_page_edges()
        if pts is None:
            pts = scan_corner_marks()
        return pts

def process_omr(image_np, align_mode, debug=False, show_missed=False):
    orig = image_np.copy()
    
    # 1. Resize for standard math
    orig = imutils.resize(orig, height=1500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # 2. Get the anchor points using chosen method
    anchor_pts = get_warp_points(gray, align_mode)

    # 3. Perspective Transform
    if anchor_pts is not None:
        paper = four_point_transform(gray, anchor_pts)
        color_paper = four_point_transform(orig, anchor_pts)
    else:
        # Fallback if both methods completely fail
        paper = gray
        color_paper = orig

    # 4. Bubble Thresholding
    paper = imutils.resize(paper, height=1500)
    color_paper = imutils.resize(color_paper, height=1500)
    
    thresh = cv2.adaptiveThreshold(paper, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 11)

    # 5. Find Bubbles
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Widened constraints to catch bubbles even if warp is imperfect
        if 15 <= w <= 60 and 15 <= h <= 60 and 0.6 <= ar <= 1.4:
            bubbles.append(c)

    # Noise filter: Keep only the 240 most perfectly circular bubbles
    if len(bubbles) > EXPECTED_BUBBLES:
        bubbles = sorted(bubbles, key=lambda c: abs(1.0 - (cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]))))
        bubbles = bubbles[:EXPECTED_BUBBLES]

    # Diagnostic image
    debug_img = color_paper.copy()
    cv2.drawContours(debug_img, bubbles, -1, (255, 0, 255), 2)

    if len(bubbles) != EXPECTED_BUBBLES:
        return None, len(bubbles), color_paper, debug_img, f"Bubble count mismatch. Alignment Mode used: '{align_mode}'."

    # 6. Sorting & Grading
    try:
        bubbles = imutils_contours.sort_contours(bubbles, method="left-to-right")[0]
        cols = [bubbles[0:80], bubbles[80:160], bubbles[160:240]]
        
        results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
        q_idx = 1

        for col in cols:
            col_cnts = imutils_contours.sort_contours(col, method="top-to-bottom")[0]
            
            for i in np.arange(0, len(col_cnts), 4):
                row = imutils_contours.sort_contours(col_cnts[i:i+4], method="left-to-right")[0]
                
                pixel_counts = []
                for (j, c) in enumerate(row):
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(mask)
                    pixel_counts.append((total, j))
                
                pixel_counts.sort(key=lambda x: x[0], reverse=True)
                
                darkest_val, darkest_idx = pixel_counts[0]
                second_darkest_val = pixel_counts[1][0]
                
                correct_ans = ANS_KEY.get(q_idx)
                correct_ans_idx = correct_ans - 1 
                
                # --- GRADING LOGIC ---
                if darkest_val < 300: 
                    results["blank"] += 1
                    if show_missed:
                        cv2.drawContours(color_paper, [row[correct_ans_idx]], -1, (255, 0, 0), 2)
                        
                elif second_darkest_val > (darkest_val * 0.75): 
                    results["double"] += 1
                    results["wrong"] += 1
                    cv2.drawContours(color_paper, [row[darkest_idx], row[pixel_counts[1][1]]], -1, (0, 255, 255), 3)
                    if show_missed:
                        cv2.drawContours(color_paper, [row[correct_ans_idx]], -1, (255, 0, 0), 2)
                        
                elif darkest_idx == correct_ans_idx:
                    results["correct"] += 1
                    cv2.drawContours(color_paper, [row[darkest_idx]], -1, (0, 255, 0), 3)
                    
                else:
                    results["wrong"] += 1
                    cv2.drawContours(color_paper, [row[darkest_idx]], -1, (0, 0, 255), 3) 
                    if show_missed:
                        cv2.drawContours(color_paper, [row[correct_ans_idx]], -1, (255, 0, 0), 2)
                
                q_idx += 1

        return results, len(bubbles), color_paper, debug_img, "Success"

    except Exception as e:
        return None, len(bubbles), color_paper, debug_img, f"Sorting Error: Image could not be aligned properly. ({str(e)})"


# --- UI ---
st.title("📝 Yuva Gyan Mahotsav OMR Grader")

with st.sidebar:
    st.header("⚙️ Scanner Controls")
    
    # NEW FEATURE: Let the user choose the alignment method!
    align_mode = st.radio(
        "Alignment Method (If it fails, try a different one):",
        ["Auto (Try Both)", "Page Edges (Standard)", "Corner Marks (Fiducials)"]
    )
    
    st.divider()
    st.header("⚙️ Display Settings")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=False, help="Draws a blue circle to show the right answer on mistakes.")
    debug_mode = st.toggle("View AI Diagnostics", value=False, help="Shows exactly which bubbles the AI is finding if an error occurs.")

# --- FULL WIDTH TABBED INTERFACE ---
tab1, tab2 = st.tabs(["📱 Native Mobile Camera (Best for Focus)", "📸 Quick Browser Scanner"])

with tab1:
    st.write("### 📸 High-Quality Capture")
    st.success("💡 **Tip for Mobile:** Tap **Browse files** below to use your native Camera app (enables **tap-to-focus** and **flash**).")
    upload_img = st.file_uploader("Take Photo or Upload Image", type=['jpg','png','jpeg'])

with tab2:
    st.write("### 🎥 Live Browser Scan")
    st.warning("Web browsers block manual focus controls. If the image is blurry, please use the Native Mobile Camera tab instead.")
    camera_img = st.camera_input("Live Camera", label_visibility="collapsed")

input_file = upload_img if upload_img is not None else camera_img

if input_file:
    img = Image.open(input_file).convert('RGB')
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing OMR Sheet..."):
        output = process_omr(img_np, align_mode=align_mode, debug=debug_mode, show_missed=show_missed)
        
        if output[0] is None:
            data, count, processed_img, debug_img, status_msg = output
            st.error(f"⚠️ **Evaluation Failed:** Found {count}/{EXPECTED_BUBBLES} bubbles. {status_msg}")
            st.info("💡 **Fix it:** Look at the sidebar on the left and change the **Alignment Method** to something else, then try again.")
            
            st.markdown("### 🔍 Diagnostic View (What the AI Sees)")
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        else:
            data, count, processed_img, debug_img, status_msg = output
            pos = data['correct'] * CORRECT_PTS
            neg = data['wrong'] * WRONG_PTS
            total = pos - neg
            
            # --- SCORECARD ---
            st.markdown("---")
            st.markdown("### 📊 OFFICIAL SCORECARD")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Correct", data['correct'], f"+{pos} pts")
            m2.metric("Incorrect", data['wrong'], f"-{neg} pts")
            m3.metric("Blank", data['blank'])
            m4.metric("Double Marked", data['double'], help="Counted as incorrect")
            m5.metric("FINAL SCORE", total)
            st.markdown("---")
            
            # --- VISUAL VERIFICATION ---
            col_img, col_legend = st.columns([3, 1])
            with col_img:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_legend:
                st.write("### Legend")
                st.success("🟢 **Correct**")
                st.error("🔴 **Incorrect**")
                st.warning("🟡 **Double Bubble**")
                if show_missed:
                    st.info("🔵 **Missed Answer**")
