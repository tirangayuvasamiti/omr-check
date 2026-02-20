import streamlit as st
import cv2
import numpy as np

# --- Configuration & Constants ---
st.set_page_config(page_title="Ultra OMR Grader", layout="wide")

# Marking Scheme
SCORE_CORRECT = 3
SCORE_INCORRECT = -1
SCORE_UNATTEMPTED = 0

# --- EDIT YOUR ANSWER KEY HERE ---
# Use 'A', 'B', 'C', or 'D' for each question.
ANSWER_KEY = {
    1: 'A', 2: 'C', 3: 'D', 4: 'A', 5: 'A', 6: 'A', 7: 'C', 8: 'B', 9: 'D', 10: 'D',
    11: 'D', 12: 'A', 13: 'D', 14: 'D', 15: 'A', 16: 'D', 17: 'D', 18: 'A', 19: 'D', 20: 'C',
    21: 'D', 22: 'B', 23: 'D', 24: 'A', 25: 'C', 26: 'A', 27: 'D', 28: 'A', 29: 'C', 30: 'A',
    31: 'A', 32: 'B', 33: 'A', 34: 'D', 35: 'A', 36: 'C', 37: 'D', 38: 'B', 39: 'A', 40: 'D',
    41: 'A', 42: 'A', 43: 'D', 44: 'A', 45: 'A', 46: 'A', 47: 'C', 48: 'D', 49: 'D', 50: 'D',
    51: 'D', 52: 'A', 53: 'D', 54: 'A', 55: 'D', 56: 'C', 57: 'B', 58: 'D', 59: 'D', 60: 'C'
}

# Helper to map letters to indices
OPT_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Colors for bounding boxes (BGR format for OpenCV)
COLOR_CORRECT = (0, 255, 0)      # Green
COLOR_INCORRECT = (0, 0, 255)    # Red
COLOR_MULTIPLE = (0, 165, 255)   # Orange
COLOR_UNATTEMPTED = (255, 0, 0)  # Blue

def order_points(pts):
    """Order coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_omr_grid(image):
    """Detects the 4 corner fiducials and warps the grid area perfectly."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doc_cnt = None

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

    if doc_cnt is None:
        return None, None

    pts = doc_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

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

def grade_omr(warped):
    """Processes the 3-column warped grid using fill-percentage for high accuracy."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding handles shadows and varying light beautifully
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    height, width = warped.shape[:2]
    
    cols = 3
    rows_per_col = 20
    options_per_q = 4
    
    col_width = width // cols
    row_height = height // rows_per_col

    results = []
    correct_count = 0
    incorrect_count = 0
    unattempted_count = 0

    question_num = 1

    for c in range(cols):
        for r in range(rows_per_col):
            x_start = c * col_width
            y_start = r * row_height
            
            # Options start after the question number (approx 25% of the column width)
            q_x_offset = int(col_width * 0.25) 
            q_width = col_width - q_x_offset
            opt_width = q_width // options_per_q
            
            fill_percentages = []
            bubbles_coords = []

            for o in range(options_per_q):
                bx_start = x_start + q_x_offset + (o * opt_width)
                
                # Extract specific bubble region
                bubble_roi = thresh[y_start:y_start+row_height, bx_start:bx_start+opt_width]
                
                # Calculate fill percentage instead of raw pixel count for higher accuracy
                total_pixels = bubble_roi.shape[0] * bubble_roi.shape[1]
                filled_pixels = cv2.countNonZero(bubble_roi)
                percentage = (filled_pixels / float(total_pixels)) * 100 if total_pixels > 0 else 0
                
                fill_percentages.append(percentage)
                bubbles_coords.append((bx_start, y_start, opt_width, row_height))

            # Threshold for considering a bubble "filled" (e.g., > 30% filled with dark ink)
            fill_threshold = 30.0 
            filled_indices = [i for i, p in enumerate(fill_percentages) if p > fill_threshold]
            
            # Map correct answer letter to index
            actual_ans_letter = ANSWER_KEY.get(question_num, 'A')
            actual_ans_idx = OPT_MAP.get(actual_ans_letter, 0)
            
            status = ""
            color = (0,0,0)

            if len(filled_indices) == 1:
                marked_ans = filled_indices[0]
                if marked_ans == actual_ans_idx:
                    status = "Correct"
                    color = COLOR_CORRECT
                    correct_count += 1
                else:
                    status = "Incorrect"
                    color = COLOR_INCORRECT
                    incorrect_count += 1
            elif len(filled_indices) > 1:
                status = "Multiple"
                color = COLOR_MULTIPLE
                incorrect_count += 1 # Counted as incorrect
            else:
                status = "Unattempted"
                color = COLOR_UNATTEMPTED
                unattempted_count += 1

            # Draw visual feedback
            for i, (bx, by, bw, bh) in enumerate(bubbles_coords):
                if i in filled_indices:
                     # Highlight detected fills
                     cv2.rectangle(warped, (bx, by), (bx+bw, by+bh), color, 2)
                elif i == actual_ans_idx and status != "Correct":
                     # Show what the correct answer should have been
                     cv2.circle(warped, (bx + bw//2, by + bh//2), min(bw,bh)//3, COLOR_CORRECT, 1)

            results.append({"Q": question_num, "Status": status})
            question_num += 1

    # Calculate Scores
    pos_score = correct_count * SCORE_CORRECT
    neg_score = incorrect_count * abs(SCORE_INCORRECT) # Ensuring proper subtraction display
    total_score = (correct_count * SCORE_CORRECT) + (incorrect_count * SCORE_INCORRECT)

    metrics = {
        "Correct": correct_count,
        "Incorrect": incorrect_count,
        "Unattempted": unattempted_count,
        "Positive": pos_score,
        "Negative": neg_score,
        "Total": total_score
    }

    return warped, metrics

# --- Streamlit UI ---
st.title("📝 Yuva Gyan Mahotsav 2026 - Ultra Accurate OMR Grader")
st.markdown("Upload a scanned or photographed OMR sheet. The system will auto-align and grade it based on the hardcoded Answer Key.")

uploaded_file = st.file_uploader("Upload OMR Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.write("### Processing...")
    
    warped_img = get_omr_grid(image)
    
    if warped_img is None:
         st.error("Could not detect the 4 corners of the OMR grid. Ensure the image captures the black corner markers clearly.")
    else:
        annotated_img, metrics = grade_omr(warped_img)
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Processed & Graded OMR", use_container_width=True)
            st.caption("🟢 Correct | 🔴 Incorrect | 🟠 Multiple | 🔵 Unattempted")
            
        with col2:
            st.subheader("📊 Final Results & Marking")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Correct Questions", metrics["Correct"])
            m2.metric("Incorrect / Multiple", metrics["Incorrect"])
            m3.metric("Unattempted", metrics["Unattempted"])
            
            st.divider()
            
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Positive Score (+3)", f"+{metrics['Positive']}")
            sm2.metric("Negative Score (-1)", f"-{metrics['Negative']}")
            sm3.metric("🏆 TOTAL SCORE", metrics["Total"])
            
            st.success(f"Final Score: {metrics['Total']} / 180")
