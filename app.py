import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours as imutils_contours
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Grader Pro", layout="wide")

# --- SCORING RULES ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted
EXPECTED_BUBBLES = 240

# --- VERIFIED ANSWER KEY ---
# A=0, B=1, C=2, D=3
ANS_KEY = {
    1: 1, 2: 3, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 3, 10: 1,
    11: 0, 12: 0, 13: 3, 14: 0, 15: 1, 16: 0, 17: 3, 18: 1, 19: 1, 20: 2,
    21: 3, 22: 1, 23: 0, 24: 1, 25: 3, 26: 0, 27: 2, 28: 3, 29: 3, 30: 2,
    31: 0, 32: 2, 33: 1, 34: 1, 35: 3, 36: 1, 37: 3, 38: 2, 39: 0, 40: 3,
    41: 1, 42: 0, 43: 3, 44: 2, 45: 1, 46: 2, 47: 0, 48: 0, 49: 0, 50: 0,
    51: 0, 52: 1, 53: 1, 54: 3, 55: 1, 56: 2, 57: 0, 58: 1, 59: 1, 60: 1
}

def process_omr(image_np, debug=False):
    orig = image_np.copy()
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Paper & Perspective Transform
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is not None:
        paper = four_point_transform(gray, docCnt.reshape(4, 2))
        color_paper = four_point_transform(orig, docCnt.reshape(4, 2))
    else:
        paper = gray
        color_paper = orig

    # 2. Thresholding (Resizing is crucial for consistent bubble size)
    paper = imutils.resize(paper, height=1500)
    color_paper = imutils.resize(color_paper, height=1500)
    
    thresh = cv2.adaptiveThreshold(paper, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 11)

    # 3. Find Bubbles
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Relaxed constraints to catch bubbles even if slightly stretched by camera angle
        if 18 <= w <= 55 and 18 <= h <= 55 and 0.7 <= ar <= 1.3:
            bubbles.append(c)

    # NOISE FILTER: If too many bubbles, keep only the most circular/correctly sized ones
    if len(bubbles) > EXPECTED_BUBBLES:
        # Sort by circularity (w/h closest to 1.0)
        bubbles = sorted(bubbles, key=lambda c: abs(1.0 - (cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]))))
        bubbles = bubbles[:EXPECTED_BUBBLES]

    # Debug Image Generation
    debug_img = color_paper.copy()
    cv2.drawContours(debug_img, bubbles, -1, (255, 0, 255), 2) # Draw found bubbles in purple

    if len(bubbles) != EXPECTED_BUBBLES:
        return None, len(bubbles), color_paper, debug_img, "Bubble count mismatch."

    # 4. Sorting & Grading
    try:
        bubbles = imutils_contours.sort_contours(bubbles, method="left-to-right")[0]
        # Split into 3 columns of 80 bubbles (20 questions * 4 options each)
        cols = [bubbles[0:80], bubbles[80:160], bubbles[160:240]]
        
        results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
        q_idx = 1

        for col in cols:
            # Sort each column top-to-bottom
            col_cnts = imutils_contours.sort_contours(col, method="top-to-bottom")[0]
            
            for i in np.arange(0, len(col_cnts), 4):
                # Sort the 4 bubbles in the row left-to-right (A, B, C, D)
                row = imutils_contours.sort_contours(col_cnts[i:i+4], method="left-to-right")[0]
                
                pixel_counts = []
                for (j, c) in enumerate(row):
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(mask)
                    pixel_counts.append((total, j))
                
                # Sort by darkest bubble
                pixel_counts.sort(key=lambda x: x[0], reverse=True)
                
                darkest_val, darkest_idx = pixel_counts[0]
                second_darkest_val = pixel_counts[1][0]
                
                correct_ans = ANS_KEY.get(q_idx)
                
                # Grading Logic
                if darkest_val < 300: 
                    # If even the darkest bubble has less than 300 filled pixels, it's blank
                    results["blank"] += 1
                elif second_darkest_val > (darkest_val * 0.75): 
                    # If the 2nd darkest is at least 75% as dark as the 1st, it's a double mark
                    results["double"] += 1
                    results["wrong"] += 1
                    cv2.drawContours(color_paper, [row[darkest_idx], row[pixel_counts[1][1]]], -1, (0, 255, 255), 3) # Yellow
                elif darkest_idx == correct_ans:
                    results["correct"] += 1
                    cv2.drawContours(color_paper, [row[darkest_idx]], -1, (0, 255, 0), 3) # Green
                else:
                    results["wrong"] += 1
                    cv2.drawContours(color_paper, [row[darkest_idx]], -1, (0, 0, 255), 3) # Red
                    cv2.drawContours(color_paper, [row[correct_ans]], -1, (255, 0, 0), 2) # Blue for correct answer
                
                q_idx += 1

        return results, len(bubbles), color_paper, debug_img, "Success"

    except Exception as e:
        return None, len(bubbles), color_paper, debug_img, f"Sorting Error: The camera angle is too crooked to align the grid properly. ({str(e)})"


# --- UI ---
st.title("📝 Yuva Gyan Mahotsav 2026 OMR Grader")

with st.sidebar:
    st.header("⚙️ Diagnostics")
    debug_mode = st.toggle("View AI Diagnostics", value=True, help="Shows exactly which bubbles the AI is finding.")
    st.info("💡 **Tips for perfect scanning:**\n\n1. Lay paper flat on a dark surface.\n2. Avoid harsh shadows or glare.\n3. Hold camera directly overhead.")

camera_img = st.camera_input("Take a photo of the OMR sheet")
upload_img = st.file_uploader("Or upload an image", type=['jpg','png','jpeg'])

input_file = camera_img if camera_img else upload_img

if input_file:
    img = Image.open(input_file).convert('RGB')
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Grading..."):
        data, count, processed_img, debug_img, status_msg = process_omr(img_np, debug=debug_mode)
        
        if data is None:
            st.error(f"⚠️ **Evaluation Failed:** Found {count}/{EXPECTED_BUBBLES} bubbles. {status_msg}")
            
            st.markdown("### 🔍 What the AI Sees (Diagnostic View)")
            st.write("If you see fewer/more than 240 purple circles below, lighting or camera angle is the issue.")
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        else:
            pos = data['correct'] * CORRECT_PTS
            neg = data['wrong'] * WRONG_PTS
            total = pos - neg
            
            # --- SCORECARD ---
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Correct", data['correct'], f"+{pos}")
            col2.metric("Incorrect", data['wrong'], f"-{neg}")
            col3.metric("Blank", data['blank'])
            col4.metric("Double Marked", data['double'], help="Counted as incorrect")
            col5.metric("FINAL SCORE", total)
            st.markdown("---")
            
            # --- VISUAL VERIFICATION ---
            col_img, col_legend = st.columns([3, 1])
            with col_img:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_legend:
                st.write("### Legend")
                st.success("🟢 **Correct**")
                st.error("🔴 **Incorrect**")
                st.info("🔵 **Missed Answer**")
                st.warning("🟡 **Double Bubble**")
