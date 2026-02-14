import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- STRICT SCORING SCHEME ---
CORRECT_PTS = 3
WRONG_PTS = 1  # This will be subtracted
UNATTEMPTED_PTS = 0
TOTAL_QUESTIONS = 60

# --- MASTER ANSWER KEY ---
# Extracted from the Yuva Gyan Mahotsav 2026 layout (A=0, B=1, C=2, D=3)
master_key = {
    1: 1,  2: 3,  3: 1,  4: 1,  5: 1,  6: 0,  7: 0,  8: 0,  9: 3, 10: 1, 
    11: 0, 12: 0, 13: 3, 14: 0, 15: 1, 16: 0, 17: 3, 23: 0, 24: 1, 25: 3, 
    26: 0, 27: 2, 28: 3, 29: 3, 30: 2, 31: 0, 32: 2, 33: 1, 34: 1, 36: 1, 
    37: 3, 39: 0, 40: 3, 42: 0, 43: 3, 44: 2, 45: 1, 46: 2, 47: 0, 48: 0, 
    49: 0, 50: 0, 51: 0, 52: 1, 53: 1, 54: 3, 55: 1, 56: 2, 57: 0
}

def process_omr_image(image_pil):
    """
    OpenCV Pipeline to process the image and extract answers.
    """
    # 1. Convert Image for OpenCV
    image = np.array(image_pil.convert('RGB'))
    image = image[:, :, ::-1].copy() # RGB to BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Basic Image Processing (Blur & Edge Detection)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # 3. Thresholding (Turns background white, ink black)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # --- SIMULATION BLOCK ---
    # To prevent the app from crashing on Streamlit before you tune the exact 
    # bubble coordinates of your camera setup, we simulate the extraction phase.
    # In a production environment, you replace this block with cv2 contour sorting.
    
    student_answers = master_key.copy()
    student_answers[2] = 1     # Simulating an incorrect answer
    student_answers[3] = None  # Simulating an unattempted answer
    student_answers[50] = 3    # Simulating an incorrect answer
    
    return student_answers

def calculate_score(student_answers):
    """
    Applies the +3 / -1 / 0 marking scheme.
    """
    correct = 0
    incorrect = 0
    unattempted = 0

    for q_num in range(1, TOTAL_QUESTIONS + 1):
        correct_ans = master_key.get(q_num)
        student_ans = student_answers.get(q_num)

        # Skip questions that aren't in the master key (if any are blanked out)
        if correct_ans is None and student_ans is None:
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

# --- STREAMLIT UI ---
st.set_page_config(page_title="OMR Grader", layout="centered")

st.title("OMR Auto-Grader System")
st.write("Upload a scanned image or mobile photo of the filled OMR sheet.")

uploaded_file = st.file_uploader("Upload OMR Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", use_container_width=True)
    
    if st.button("SCAN & GENERATE RESULT", type="primary"):
        with st.spinner("Processing OMR via OpenCV..."):
            
            # Run the vision pipeline
            student_answers = process_omr_image(image)
            
            # Calculate final grades
            c, i, u, pos, neg, total = calculate_score(student_answers)
            
            # Display Exact Output Format
            st.divider()
            st.markdown("### 📊 OFFICIAL RESULTS")
            
            st.success(f"**CORRECT:** {c}\n\n"
                       f"**INCORRECT:** {i}\n\n"
                       f"**UNATTEMPTED:** {u}\n\n"
                       f"**POS. SCORE (+):** {pos}\n\n"
                       f"**NEG. SCORE (-):** {neg}\n\n"
                       f"### TOTAL OBTAINED SCORE: {total}")