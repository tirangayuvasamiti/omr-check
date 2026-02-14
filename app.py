import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours as imutils_contours
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Grader Pro", layout="wide", page_icon="🎓")

# --- CONSTANTS & SCORING RULES ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted
TOTAL_QUESTIONS = 60
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
    """Processes the OMR sheet and returns scores, bubble count, and processed images."""
    orig = image_np.copy()
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Paper & Perspective Transform (Added dilation for better edge detection)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    edged = cv2.dilate(edged, None, iterations=1) 

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

    # Apply warp perspective
    if docCnt is not None:
        paper = four_point_transform(gray, docCnt.reshape(4, 2))
        color_paper = four_point_transform(orig, docCnt.reshape(4, 2))
    else:
        paper = gray
        color_paper = orig

    # 2. Thresholding
    paper = imutils.resize(paper, height=1500)
    color_paper = imutils.resize(color_paper, height=1500)
    
    # Using Otsu's thresholding for better dynamic range handling
    blurred_paper = cv2.GaussianBlur(paper, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred_paper, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 11)

    # 3. Find Bubbles
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Slightly relaxed constraints to handle slight skews
        if 15 <= w <= 55 and 15 <= h <= 55 and 0.7 <= ar <= 1.3:
            bubbles.append(c)

    # If we found too many bubbles (noise), filter by perfectly circular contours
    if len(bubbles) > EXPECTED_BUBBLES:
        bubbles = sorted(bubbles, key=lambda c: abs(1.0 - (cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]))))
        bubbles = bubbles[:EXPECTED_BUBBLES]

    if len(bubbles) < EXPECTED_BUBBLES:
        return None, len(bubbles), color_paper, thresh

    # 4. Sorting logic (Left-to-Right for 3 Columns)
    bubbles = imutils_contours.sort_contours(bubbles, method="left-to-right")[0]
    cols = [bubbles[0:80], bubbles[80:160], bubbles[160:240]]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    q_idx = 1

    for col in cols:
        col_cnts = imutils_contours.sort_contours(col, method="top-to-bottom")[0]
        
        for i in np.arange(0, len(col_cnts), 4):
            row = imutils_contours.sort_contours(col_cnts[i:i+4], method="left-to-right")[0]
            bubbled_answers = []
            
            for (j, c) in enumerate(row):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                
                # Dynamic Thresholding: Check if > 40% of the bubble area is filled
                (x, y, w, h) = cv2.boundingRect(c)
                bubble_area = w * h
                fill_ratio = total / float(bubble_area)
                
                if fill_ratio > 0.40:  # 40% threshold (Adjustable)
                    bubbled_answers.append(j)

            correct_ans = ANS_KEY.get(q_idx)

            # Grading Logic
            if len(bubbled_answers) == 0:
                results["blank"] += 1
            elif len(bubbled_answers) > 1:
                results["double"] += 1
                results["wrong"] += 1 # Treat double bubble as wrong
                for b in bubbled_answers:
                    cv2.drawContours(color_paper, [row[b]], -1, (0, 255, 255), 3) # Yellow for double
            elif bubbled_answers[0] == correct_ans:
                results["correct"] += 1
                cv2.drawContours(color_paper, [row[bubbled_answers[0]]], -1, (0, 255, 0), 3) # Green
            else:
                results["wrong"] += 1
                cv2.drawContours(color_paper, [row[bubbled_answers[0]]], -1, (0, 0, 255), 3) # Red
                cv2.drawContours(color_paper, [row[correct_ans]], -1, (255, 0, 0), 2) # Blue Correct
            
            q_idx += 1

    return results, len(bubbles), color_paper, thresh

# --- UI ---
st.title("🎓 Yuva Gyan Mahotsav 2026")
st.markdown("Scan OMR sheets instantly. Powered by OpenCV & Streamlit.")

with st.sidebar:
    st.header("⚙️ Settings")
    debug_mode = st.toggle("Enable Debug Mode", help="Show binary thresholding map to debug lighting issues.")
    st.info("Ensure the OMR sheet fills the camera view and lies flat in good lighting.")

col1, col2 = st.columns(2)
with col1:
    camera_img = st.camera_input("Take a photo of the OMR sheet")
with col2:
    upload_img = st.file_uploader("Or upload an image", type=['jpg','png','jpeg'])

input_file = camera_img if camera_img else upload_img

if input_file:
    # Convert file to OpenCV format safely
    img = Image.open(input_file).convert('RGB')
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing OMR Sheet..."):
        try:
            data, count, processed_img, thresh_img = process_omr(img_np, debug=debug_mode)
            
            if data is None:
                st.error(f"⚠️ **Detection Error:** Found {count}/{EXPECTED_BUBBLES} bubbles. Please hold the camera closer/straighter.")
                
                # Debugging visualizer
                st.write("### What the AI sees:")
                debug_col1, debug_col2 = st.columns(2)
                debug_col1.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Perspective View", use_container_width=True)
                debug_col2.image(thresh_img, caption="Binary Map (Ensure bubbles are clear white dots)", use_container_width=True, clamp=True)
                
            else:
                pos = data['correct'] * CORRECT_PTS
                neg = data['wrong'] * WRONG_PTS
                total = pos - neg
                
                st.markdown("---")
                st.markdown("### 📊 OFFICIAL SCORECARD")
                
                # Dashboard style metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Correct", data['correct'], f"+{pos} pts")
                m2.metric("Incorrect", data['wrong'], f"-{neg} pts")
                m3.metric("Unattempted", data['blank'])
                m4.metric("Double Marked", data['double'], help="Counted as incorrect")
                m5.metric("FINAL SCORE", total)

                st.markdown("---")
                
                col_img1, col_img2 = st.columns([1, 1])
                with col_img1:
                    st.markdown("#### 🔍 Visual Verification")
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col_img2:
                    if debug_mode:
                        st.markdown("#### 🛠️ AI Threshold Map")
                        st.image(thresh_img, use_container_width=True, clamp=True)
                    else:
                        st.success("✅ OMR Graded Successfully!")
                        st.write("**Legend:**")
                        st.write("🟢 **Green:** Correct")
                        st.write("🔴 **Red:** Incorrect (Student's Answer)")
                        st.write("🔵 **Blue:** Missed Correct Answer")
                        st.write("🟡 **Yellow:** Invalid (Double Bubbled)")
                        
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.info("Try turning on 'Debug Mode' in the sidebar to see what is failing.")
