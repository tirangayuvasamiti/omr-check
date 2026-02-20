import streamlit as st
import cv2
import numpy as np
import imutils
from imutils import contours

# --- CONFIGURATION ---
ANSWER_KEY = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0, 10: 1,
    11: 2, 12: 3, 13: 0, 14: 1, 15: 2, 16: 3, 17: 0, 18: 1, 19: 2, 20: 3,
    21: 0, 22: 1, 23: 2, 24: 3, 25: 0, 26: 1, 27: 2, 28: 3, 29: 0, 30: 1,
    31: 2, 32: 3, 33: 0, 34: 1, 35: 2, 36: 3, 37: 0, 38: 1, 39: 2, 40: 3,
    41: 0, 42: 1, 43: 2, 44: 3, 45: 0, 46: 1, 47: 2, 48: 3, 49: 0, 50: 1,
    51: 2, 52: 3, 53: 0, 54: 1, 55: 2, 56: 3, 57: 0, 58: 1, 59: 2, 60: 3
}

def grade_omr(image_bytes):
    # Convert uploaded web file to an OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    question_cnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 15 and h >= 15 and 0.8 <= ar <= 1.2:
            question_cnts.append(c)

    # Sort into 3 columns
    boxes = [cv2.boundingRect(c) for c in question_cnts]
    min_x = min([b[0] for b in boxes])
    max_x = max([b[0] + b[2] for b in boxes])
    col_width = (max_x - min_x) / 3

    col1, col2, col3 = [], [], []
    for c in question_cnts:
        x = cv2.boundingRect(c)[0]
        if x < min_x + col_width:
            col1.append(c)
        elif x < min_x + (2 * col_width):
            col2.append(c)
        else:
            col3.append(c)

    columns = [col1, col2, col3]
    correct, incorrect, unattempted = 0, 0, 0
    current_question = 1

    for col_cnts in columns:
        if len(col_cnts) == 0: continue
        col_cnts = contours.sort_contours(col_cnts, method="top-to-bottom")[0]

        for (q, i) in enumerate(np.arange(0, len(col_cnts), 4)):
            if i + 4 > len(col_cnts): break
            cnts_row = contours.sort_contours(col_cnts[i:i + 4], method="left-to-right")[0]
            bubbled = None

            for (j, c) in enumerate(cnts_row):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            color = (0, 0, 255) # Red for incorrect
            if bubbled[0] < 200: 
                unattempted += 1
                color = (0, 255, 255) # Yellow for unattempted
            else:
                correct_answer = ANSWER_KEY.get(current_question, -1)
                if bubbled[1] == correct_answer:
                    correct += 1
                    color = (0, 255, 0) # Green for correct
                else:
                    incorrect += 1
                cv2.drawContours(image, [cnts_row[bubbled[1]]], -1, color, 3)
            
            if bubbled[0] >= 200 and bubbled[1] != correct_answer and correct_answer != -1:
                 cv2.drawContours(image, [cnts_row[correct_answer]], -1, (255, 0, 0), 2)
                 
            current_question += 1

    score = (correct * 3) + (incorrect * -1)
    
    # Convert OpenCV image (BGR) to Streamlit image (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return score, correct, incorrect, unattempted, image_rgb

# --- STREAMLIT WEB UI ---
st.set_page_config(page_title="OMR Auto-Grader", layout="centered")
st.title("📄 OMR Auto-Grader")
st.markdown("Upload a scanned OMR sheet to automatically grade it.")

uploaded_file = st.file_uploader("Upload OMR Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Grading OMR Sheet..."):
        # Pass the uploaded file to our OpenCV function
        score, correct, incorrect, unattempted, graded_img = grade_omr(uploaded_file.read())
        
        # Display Results
        st.success("Grading Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Score", f"{score}/180")
        col2.metric("Correct (+3)", correct)
        col3.metric("Incorrect (-1)", incorrect)
        col4.metric("Unattempted", unattempted)
        
        # Display the visually graded image
        st.image(graded_img, caption="Graded OMR Sheet", use_container_width=True)
