import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours as imutils_contours
from PIL import Image

# --- STRICT SCORING SCHEME ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Will be subtracted
UNATTEMPTED_PTS = 0
TOTAL_QUESTIONS = 60

# --- MASTER ANSWER KEY ---
# Updated to exactly match the checked boxes in the Yuva Gyan Mahotsav 2026 document.
# Mapping: A=0, B=1, C=2, D=3
master_key = {
    1: 1, 2: 3, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 
    8: 0, 9: 3, 10: 1, 11: 0, 12: 0, 13: 3, 14: 0, 
    15: 1, 16: 0, 17: 3, 18: None, 19: None, 20: None, 21: 0, 22: 1, 
    23: 0, 24: 1, 25: 3, 26: 0, 27: 2, 28: 3, 29: 3, 30: 2, 
    31: 0, 32: 2, 33: 1, 34: 1, 35: None, 36: 1, 37: 3, 38: None, 39: 0, 40: 3, 
    41: 1, 42: 0, 43: 3, 44: 2, 45: 1, 46: 2, 47: 0, 48: 0, 49: 0, 50: 0, 
    51: 0, 52: 1, 53: 1, 54: 3, 55: 1, 
    56: 2, 57: 0, 58: None, 59: None, 60: None
}

def process_pro_omr(image_pil):
    """
    Advanced OpenCV Pipeline: Detects paper, flattens it, dynamically sorts 
    bubbles into rows, and detects filled marks using pixel intensity.
    """
    image = np.array(image_pil.convert('RGB'))
    image = image[:, :, ::-1].copy() # Convert RGB to BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge Detection & Document Contour Extraction
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

    # 2. Perspective Transform (Flatten the scanned page)
    if docCnt is not None:
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
    else:
        # Fallback if edges aren't clear
        warped = gray

    # 3. Binarization (Isolate the ink)
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 4. Find all circular bubbles
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Identify bubbles based on size and aspect ratio
        if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.2:
            questionCnts.append(c)

    # --- THE "PRO" LOGIC: DYNAMIC SORTING ---
    student_answers = {}
    
    # We expect exactly 240 bubbles (60 questions * 4 options).
    # If the camera didn't pick up exactly 240, we return a failure flag so 
    # the user knows to retake the photo, preventing incorrect grades.
    if len(questionCnts) != 240:
        return None, len(questionCnts)

    # Sort bubbles top-to-bottom
    questionCnts = imutils_contours.sort_contours(questionCnts, method="top-to-bottom")[0]

    # Iterate over bubbles in batches of 4 (Options A, B, C, D)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        # Sort the current batch of 4 from left-to-right
        cnts = imutils_contours.sort_contours(questionCnts[i:i + 4], method="left-to-right")[0]
        
        bubbled = None
        max_pixels = 0

        # Loop over the sorted contours to see which one has the most "ink"
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Count white pixels in the mask (representing filled ink)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            # Threshold for considering a bubble "filled" (adjust if needed)
            if total > 400 and total > max_pixels:
                max_pixels = total
                bubbled = j

        # Store the answer (j represents 0=A, 1=B, 2=C, 3=D)
        student_answers[q + 1] = bubbled

    return student_answers, len(questionCnts)

def grade_exam(student_answers):
    """ Calculates final scores exactly to the Yuva Gyan Mahotsav format """
    correct, incorrect, unattempted = 0, 0, 0

    for q_num in range(1, TOTAL_QUESTIONS + 1):
        correct_ans = master_key.get(q_num)
        student_ans = student_answers.get(q_num)

        # Ignore questions if the master key has them as None (dropped questions)
        if correct_ans is None:
            continue

        if student_ans is None:
            unattempted += 1
        elif student_ans == correct_ans:
            correct += 1
        else:
            incorrect += 1

    pos_score = correct * CORRECT_PTS
    neg_score = incorrect * WRONG_PTS
    total_score = pos_score - neg_score

    return correct, incorrect, unattempted, pos_score, neg_score, total_score

# --- USER INTERFACE ---
st.set_page_config(page_title="Pro OMR Grader", layout="centered")

st.title("Yuva Gyan Mahotsav 2026")
st.subheader("Automated OMR Processing System")
st.write("Upload a clear, flat photo of the student's OMR sheet.")

uploaded_file = st.file_uploader("Upload OMR Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Scanned Document", use_container_width=True)
    
    if st.button("PROCESS & GRADE EXAM", type="primary"):
        with st.spinner("Running Advanced Vision Pipeline..."):
            
            student_answers, bubbles_found = process_pro_omr(image)
            
            if student_answers is None:
                st.error("⚠️ **Scan Error:** The system could not clearly detect all 240 bubbles.")
                st.warning(f"Bubbles found: {bubbles_found} / 240. \n\n"
                           "**How to fix:** Ensure the paper is flat against a dark background, "
                           "the lighting is even, and all four corners are visible.")
            else:
                c, i, u, pos, neg, total = grade_exam(student_answers)
                
                st.divider()
                st.markdown("### 📊 FOR OFFICIAL USE ONLY")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**CORRECT:** {c}")
                    st.info(f"**INCORRECT:** {i}")
                    st.info(f"**UNATTEMPTED:** {u}")
                with col2:
                    st.success(f"**POS. SCORE (+):** {pos}")
                    st.error(f"**NEG. SCORE (-):** {neg}")
                    
                st.markdown("---")
                st.markdown(f"## **TOTAL OBTAINED SCORE: {total}**")
