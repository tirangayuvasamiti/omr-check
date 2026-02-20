import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils.perspective import four_point_transform
from pyzbar.pyzbar import decode

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Mahotsav - OMR Grader", layout="wide")

# --- GRADING SCHEME ---
CORRECT_MARKS = 3
WRONG_MARKS = -1
UNATTEMPTED_MARKS = 0

# --- COLOR CODES (BGR format for OpenCV) ---
COLOR_CORRECT = (0, 255, 0)       # Green
COLOR_WRONG = (0, 0, 255)         # Red
COLOR_MULTIFILLED = (0, 165, 255) # Orange
COLOR_MISSED_CORRECT = (255, 0, 0)# Blue (Shows what should have been marked)

def process_omr(image, answer_key):
    """
    Core Computer Vision Engine to process the OMR sheet.
    """
    # 1. Convert PIL image to OpenCV format
    image = np.array(image.convert('RGB'))
    # OpenCV uses BGR natively, but we'll work with RGB for Streamlit output
    # Let's keep a copy for drawing
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Find the document contour (largest 4-point polygon)
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

    # Apply perspective transform if document bounds are found
    if docCnt is not None:
        paper = four_point_transform(image, docCnt.reshape(4, 2))
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
        output_image = paper.copy()
    else:
        # Fallback if the image is already cropped perfectly to the document
        paper = image.copy()
        warped = gray.copy()

    # 3. Read Machine Code (Barcode/QR Code)
    decoded_objects = decode(paper)
    machine_code = decoded_objects[0].data.decode("utf-8") if decoded_objects else "Not Detected"

    # 4. Binarize the image to find bubbles
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 5. Find contours and filter for bubbles
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Filter based on expected bubble dimensions and circularity
        if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.2:
            questionCnts.append(c)

    # Note: A perfect 60-question sheet with A,B,C,D has 240 bubbles.
    # We sort them top-to-bottom, then group them.
    # For a robust 3-column layout (1-20, 21-40, 41-60), we sort dynamically.
    
    # Check if we have enough bubbles detected
    if len(questionCnts) < 240:
        st.warning(f"Only detected {len(questionCnts)}/240 bubbles. Ensure the scan is clear and unshadowed.")
        return output_image, 0, 0, 0, machine_code

    # --- ADVANCED SORTING: 3 COLUMNS ---
    # Sort all bubbles by X coordinate to divide into 3 columns
    questionCnts = sorted(questionCnts, key=lambda c: cv2.boundingRect(c)[0])
    
    col1 = questionCnts[0:80]   # Questions 1-20 (20 * 4 options)
    col2 = questionCnts[80:160] # Questions 21-40
    col3 = questionCnts[160:240]# Questions 41-60

    # Function to sort a column top-to-bottom, then left-to-right per row
    def sort_column(col_cnts):
        # Sort top-to-bottom
        col_cnts = sorted(col_cnts, key=lambda c: cv2.boundingRect(c)[1])
        sorted_col = []
        for i in range(0, len(col_cnts), 4):
            # Sort the 4 bubbles of this question left-to-right (A, B, C, D)
            row = sorted(col_cnts[i:i+4], key=lambda c: cv2.boundingRect(c)[0])
            sorted_col.extend(row)
        return sorted_col

    col1 = sort_column(col1)
    col2 = sort_column(col2)
    col3 = sort_column(col3)

    final_cnts = col1 + col2 + col3

    # 6. Grading Loop
    correct_count = 0
    wrong_count = 0
    unattempted_count = 0
    
    for (q, i) in enumerate(np.arange(0, len(final_cnts), 4)):
        cnts_row = final_cnts[i:i+4]
        bubbled = []

        # Determine which bubbles are filled based on non-zero pixels
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            # Threshold for considering a bubble "filled" (adjust based on resolution)
            if total > 300: 
                bubbled.append(j)

        correct_ans = answer_key.get(q + 1)
        
        if correct_ans is None:
            continue # Skip if no answer key for this question

        # --- EVALUATION AND DRAWING ---
        if len(bubbled) == 0:
            # Unattempted
            unattempted_count += 1
            # Outline the correct answer in Blue
            cv2.drawContours(output_image, [cnts_row[correct_ans]], -1, COLOR_MISSED_CORRECT, 3)
            
        elif len(bubbled) > 1:
            # Multi-filled (Invalid)
            wrong_count += 1
            for b in bubbled:
                cv2.drawContours(output_image, [cnts_row[b]], -1, COLOR_MULTIFILLED, 3)
            # Still show correct one in blue if it wasn't one of the ones they marked
            if correct_ans not in bubbled:
                 cv2.drawContours(output_image, [cnts_row[correct_ans]], -1, COLOR_MISSED_CORRECT, 3)
                 
        else:
            # Single Fill
            student_ans = bubbled[0]
            if student_ans == correct_ans:
                # Correct
                correct_count += 1
                cv2.drawContours(output_image, [cnts_row[student_ans]], -1, COLOR_CORRECT, 3)
            else:
                # Wrong
                wrong_count += 1
                cv2.drawContours(output_image, [cnts_row[student_ans]], -1, COLOR_WRONG, 3)
                # Show what the correct answer was in blue
                cv2.drawContours(output_image, [cnts_row[correct_ans]], -1, COLOR_MISSED_CORRECT, 3)

    return output_image, correct_count, wrong_count, unattempted_count, machine_code

# --- STREAMLIT UI ---
st.title("📝 Yuva Gyan Mahotsav 2026 - AI OMR Grader")
st.markdown("Automated evaluation engine supporting **+3, -1, 0** marking scheme with multi-fill and unattempted detection.")

# Sidebar for Answer Key
st.sidebar.header("⚙️ Answer Key Configuration")
st.sidebar.markdown("Define answers for 60 questions. (0=A, 1=B, 2=C, 3=D)")

# Generate a default answer key for testing (User can modify this logic to upload a CSV)
default_key = {i: np.random.randint(0, 4) for i in range(1, 61)} 
# For production, replace the random generation with actual inputs or file upload.
st.sidebar.write("Using internal answer key for 60 questions.")

uploaded_file = st.file_uploader("Upload Scanned OMR Sheet (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original OMR")
        st.image(image, use_container_width=True)
        
    with st.spinner("Analyzing and Grading Document..."):
        graded_img, correct, wrong, unattempted, m_code = process_omr(image, default_key)
        
        # Calculate Final Score
        total_score = (correct * CORRECT_MARKS) + (wrong * WRONG_MARKS) + (unattempted * UNATTEMPTED_MARKS)
        max_score = 60 * CORRECT_MARKS
        
    with col2:
        st.subheader("Graded OMR")
        st.image(graded_img, use_container_width=True)

    # --- RESULTS DASHBOARD ---
    st.markdown("---")
    st.header("📊 Evaluation Results")
    st.info(f"**Detected Machine Code / ID:** {m_code}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Score", f"{total_score} / {max_score}")
    m2.metric("✅ Correct (+3)", correct)
    m3.metric("❌ Incorrect (-1)", wrong)
    m4.metric("⚪ Unattempted (0)", unattempted)
    
    st.markdown("""
    ### Color Legend:
    * **<span style='color:green'>Green Outline:</span>** Correctly filled bubble.
    * **<span style='color:red'>Red Outline:</span>** Incorrectly filled bubble.
    * **<span style='color:blue'>Blue Outline:</span>** The correct answer (shown if student got it wrong or missed it).
    * **<span style='color:orange'>Orange Outline:</span>** Multi-filled/Invalid bubbles.
    """, unsafe_allow_html=True)
