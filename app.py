import streamlit as st
import cv2
import numpy as np
import imutils
from imutils import contours as imutils_contours
from PIL import Image

# --- STRICT SCORING SCHEME ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted
UNATTEMPTED_PTS = 0
TOTAL_QUESTIONS = 60

# --- MASTER ANSWER KEY (Extracted directly from your image) ---
# 0=A, 1=B, 2=C, 3=D. Questions marked None were left blank on the master.
master_key = {
    1: 1, 2: 3, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0,
    8: 0, 9: 3, 10: 1, 11: 0, 12: 0, 13: 3, 14: 0,
    15: 1, 16: 0, 17: 3, 18: 1, 19: 1, 20: 2,
    21: 0, 22: 1, 23: 0, 24: 1, 25: 3, 26: 0, 27: 2, 28: 3, 29: 3, 30: 2,
    31: 0, 32: 2, 33: 1, 34: 1, 35: None, 36: 1, 37: 3, 38: 2, 39: 0, 40: 3,
    41: 1, 42: 0, 43: 3, 44: 2, 45: 1, 46: 2, 47: 0, 48: 0, 49: 0, 50: 0,
    51: 0, 52: 1, 53: 1, 54: 3, 55: 1,
    56: 2, 57: 0, 58: None, 59: None, 60: 1
}

def process_and_grade_omr(image_pil):
    # 1. Normalize Image Size (Makes bubble sizes consistent regardless of camera)
    image = np.array(image_pil.convert('RGB'))
    image = image[:, :, ::-1].copy() # RGB to BGR
    image = imutils.resize(image, height=1500)
    annotated_image = image.copy()
    
    # 2. Adaptive Thresholding (Ignores the background watermark)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)

    # 3. Find All Contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # 4. Filter for Bubbles
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # On a 1500px high image, bubbles are roughly 15-40px
        if 15 <= w <= 40 and 15 <= h <= 40 and 0.8 <= ar <= 1.2:
            questionCnts.append(c)

    # SAFETY CHECK: Ensure we found enough bubbles
    if len(questionCnts) < 240:
        return None, len(questionCnts), None, None

    # If we found noise, take the 240 most "circular" contours
    if len(questionCnts) > 240:
        questionCnts = sorted(questionCnts, key=cv2.contourArea, reverse=True)[:240]

    # --- THE 3-COLUMN SORTING ALGORITHM ---
    # Sort all 240 bubbles left-to-right to separate the columns
    questionCnts = imutils_contours.sort_contours(questionCnts, method="left-to-right")[0]
    
    # Split into the 3 columns (80 bubbles per column)
    col1 = questionCnts[0:80]
    col2 = questionCnts[80:160]
    col3 = questionCnts[160:240]
    
    student_answers = {}
    q_offset = 0
    
    correct_count, incorrect_count, unattempted_count = 0, 0, 0

    # Process each column
    for col in [col1, col2, col3]:
        # Sort the column top-to-bottom
        col = imutils_contours.sort_contours(col, method="top-to-bottom")[0]
        
        # Iterate through questions (4 bubbles at a time)
        for (q, i) in enumerate(np.arange(0, len(col), 4)):
            q_num = q_offset + q + 1
            
            # Sort the 4 options left-to-right (A, B, C, D)
            row_cnts = imutils_contours.sort_contours(col[i:i + 4], method="left-to-right")[0]
            
            bubbled = None
            max_pixels = 0
            
            # Check which bubble has the most ink
            for (j, c) in enumerate(row_cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                
                # If a bubble is significantly filled
                if total > 300 and total > max_pixels:
                    max_pixels = total
                    bubbled = j
            
            student_answers[q_num] = bubbled
            correct_ans = master_key.get(q_num)
            
            # Draw visual feedback
            if correct_ans is not None:
                if bubbled == correct_ans:
                    # Correct: Draw Green Box around the bubble
                    cv2.drawContours(annotated_image, [row_cnts[bubbled]], -1, (0, 255, 0), 3)
                    correct_count += 1
                elif bubbled is None:
                    unattempted_count += 1
                else:
                    # Incorrect: Draw Red Box around student's choice
                    cv2.drawContours(annotated_image, [row_cnts[bubbled]], -1, (0, 0, 255), 3)
                    # Draw Blue Box around the true correct answer
                    cv2.drawContours(annotated_image, [row_cnts[correct_ans]], -1, (255, 0, 0), 3)
                    incorrect_count += 1
            else:
                # If the master key is blank (dropped question), ignore
                if bubbled is None:
                    unattempted_count += 1

        q_offset += 20 # Move to the next column's starting question number

    # Calculate final scores
    pos_score = correct_count * CORRECT_PTS
    neg_score = incorrect_count * WRONG_PTS
    total_score = pos_score - neg_score
    
    # Convert annotated image back to RGB for Streamlit
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return student_answers, len(questionCnts), annotated_image, (correct_count, incorrect_count, unattempted_count, pos_score, neg_score, total_score)


# --- STREAMLIT UI ---
st.set_page_config(page_title="Ultra Pro OMR Grader", layout="wide")

st.title("Yuva Gyan Mahotsav 2026 - OMR Auto-Grader")

uploaded_file = st.file_uploader("Upload Student OMR Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    if st.button("SCAN & GRADE", type="primary"):
        with st.spinner("Processing 3-Column OMR Layout..."):
            
            answers, count, annotated_img, metrics = process_and_grade_omr(image)
            
            if answers is None:
                st.error("⚠️ Scan Error: Could not find all 240 bubbles. Ensure the photo is clear, well-lit, and the whole paper is visible.")
            else:
                c, i, u, pos, neg, total = metrics
                
                col1, col2 = st.columns([1, 1.5])
                
                with col1:
                    st.markdown("### 📊 OFFICIAL RESULTS")
                    st.success(f"**CORRECT:** {c}")
                    st.error(f"**INCORRECT:** {i}")
                    st.warning(f"**UNATTEMPTED:** {u}")
                    st.markdown("---")
                    st.info(f"**POS. SCORE (+):** {pos}")
                    st.info(f"**NEG. SCORE (-):** {neg}")
                    st.markdown("---")
                    st.markdown(f"## **TOTAL OBTAINED SCORE: {total}**")
                    
                with col2:
                    st.markdown("### Visual Verification")
                    st.markdown("🟩 **Correct** | 🟥 **Incorrect** | 🟦 **Master Key Answer**")
                    st.image(annotated_img, use_container_width=True)
