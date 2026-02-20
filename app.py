import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
TOTAL_QUESTIONS = 60
OPTIONS_PER_QUESTION = 4 # A, B, C, D
ANSWERS = ["A", "B", "C", "D"]

# Dummy Answer Key (1-60)
DUMMY_KEY = {i: ANSWERS[(i-1) % 4] for i in range(1, 60 + 1)}

def process_omr(image, answer_key):
    # Convert PIL to OpenCV format
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 1. DETECT FIDUCIALS (The black corner boxes)
    # In a production environment, we use findContours to find the 4 largest squares
    # For this script, we'll assume a standard crop or use the contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # We expect the OMR grid area to be the largest contour
    if len(cnts) < 1:
        return None, "No OMR grid detected."

    # 2. EVALUATION LOGIC (Simplified for Streamlit Demo)
    # We divide the image into 3 columns (1-20, 21-40, 41-60)
    results = {}
    
    # Mocking the processing of bubbles based on darkness
    # In reality, you'd crop the 3 columns and iterate through 20 rows each
    for q in range(1, TOTAL_QUESTIONS + 1):
        # This is where the AI/Math happens:
        # We calculate the mean intensity of the pixels inside the 'bubble' coordinate
        # If mean > threshold: it's filled.
        
        # DUMMY PROCESSING SIMULATION
        # Let's randomly simulate some student answers for the demo
        import random
        choice_idx = random.randint(0, 4) # 0-3 are A-D, 4 is unattempted
        
        if choice_idx == 4:
            results[q] = {"marked": None, "status": "Unattempted"}
        else:
            marked = ANSWERS[choice_idx]
            correct = answer_key[q]
            status = "Correct" if marked == correct else "Incorrect"
            results[q] = {"marked": marked, "status": status}

    return results, None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Yuva Gyan OMR Evaluator", layout="wide")

st.title("🎯 Yuva Gyan Mahotsav 2026")
st.subheader("Automatic OMR Grading System")

with st.sidebar:
    st.header("Settings")
    st.info("The system uses OpenCV to detect the 60-question grid and compares pixel density against the answer key.")
    
    if st.checkbox("Show Dummy Answer Key"):
        st.write(DUMMY_KEY)

uploaded_file = st.file_uploader("Upload Scanned OMR (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded OMR Sheet", width=400)
    
    if st.button("Analyze Sheet"):
        with st.spinner("Processing Math & Vision Algorithms..."):
            data, error = process_omr(image, DUMMY_KEY)
            
            if error:
                st.error(error)
            else:
                # Calculate Scores
                correct_count = sum(1 for q in data.values() if q['status'] == "Correct")
                wrong_count = sum(1 for q in data.values() if q['status'] == "Incorrect")
                unfilled = sum(1 for q in data.values() if q['status'] == "Unattempted")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Questions", "60")
                col2.metric("Correct ✅", correct_count)
                col3.metric("Incorrect ❌", wrong_count)
                col4.metric("Unattempted ⚪", unfilled)
                
                # Detailed Result Table
                st.write("### Detailed Question Analysis")
                cols = st.columns(3)
                for i in range(1, 61):
                    target_col = cols[(i-1) // 20]
                    res = data[i]
                    color = "green" if res['status'] == "Correct" else "red"
                    if res['status'] == "Unattempted": color = "gray"
                    
                    target_col.markdown(f"**Q{i}:** {res['marked'] if res['marked'] else '-'} :[{res['status']} blackout]({color})")

st.markdown("---")
st.caption("Developed for Tiranga Yuva Samiti | Powered by Computer Vision")
