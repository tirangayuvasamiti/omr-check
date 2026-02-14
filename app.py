import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Enterprise Grader", layout="wide", page_icon="📝")

# --- RULES & ANSWER KEY ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted

ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}

# --- BGR COLORS FOR DRAWING ---
COLOR_GREEN = (0, 220, 0)     # Correct
COLOR_RED = (0, 0, 255)       # Incorrect
COLOR_BLUE = (255, 0, 0)      # Missed
COLOR_YELLOW = (0, 255, 255)  # Double Marked
COLOR_GRID = (200, 200, 200)  # Debug Grid

def load_document(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_document_corners(image):
    """Finds the 4 paper corners to flatten the document."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    doc_cnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > (image.shape[0] * image.shape[1] * 0.2):
                doc_cnt = approx
                break

    if doc_cnt is None:
        h, w = image.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
        return corners

    return doc_cnt.reshape(4, 2)

def process_omr_engine(image_np, show_missed=True):
    # 1. Base Alignment
    orig = imutils.resize(image_np, height=1500)
    corner_pts = get_document_corners(orig)

    # Standardize to an EXACT 1200x1600 matrix. 
    # This makes all internal coordinates predictable via strict math.
    warped_gray = four_point_transform(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    # =========================================================================
    # ULTRA PRO MATH: DETERMINISTIC GRID CONFIGURATION
    # Adjust these exact pixel coordinates based on the original template URL
    # Canvas Size is exactly 1200 (X) by 1600 (Y)
    # =========================================================================
    
    # Vertical Math (Y-Axis)
    Y_START = 380            # Y-coordinate of the first row (Q1, Q21, Q41)
    Y_END = 1520             # Y-coordinate of the last row (Q20, Q40, Q60)
    NUM_ROWS = 20
    Y_STEP = (Y_END - Y_START) / (NUM_ROWS - 1) # Mathematical gap between rows
    
    # Horizontal Math (X-Axis)
    # The starting X-coordinate for Option 'A' in Columns 1, 2, and 3
    COL_STARTS_X = [180, 560, 940] 
    
    # Distance between Option A, B, C, D
    OPT_STEP_X = 55 
    
    # Size of the bubble bounding box to extract
    BUBBLE_W = 40
    BUBBLE_H = 40
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown_data = []

    # Helper function to draw boxes
    def draw_box(b_box, color, thickness=3):
        cv2.rectangle(warped_color, (b_box[0], b_box[1]), (b_box[0]+b_box[2], b_box[1]+b_box[3]), color, thickness)

    # 2. Mathematical Extraction & Grading Loop
    for q_idx in range(60):
        q_number = q_idx + 1
        
        # Determine column (0, 1, 2) and row (0 to 19) using modulo math
        col_idx = q_idx // 20
        row_idx = q_idx % 20
        
        # Calculate exact Base (X, Y) for this question
        base_x = COL_STARTS_X[col_idx]
        base_y = int(Y_START + (row_idx * Y_STEP))
        
        # Calculate the 4 bubble boxes mathematically
        row_boxes = []
        fill_data = []
        
        for opt_idx in range(4):
            bx = int(base_x + (opt_idx * OPT_STEP_X))
            by = base_y
            
            # Draw a faint gray box so you can visually verify the grid aligns perfectly
            cv2.rectangle(warped_color, (bx, by), (bx+BUBBLE_W, by+BUBBLE_H), COLOR_GRID, 1)
            
            box = (bx, by, BUBBLE_W, BUBBLE_H)
            row_boxes.append(box)
            
            # Extract ROI mathematically (no contours needed!)
            roi = warped_gray[by:by+BUBBLE_H, bx:bx+BUBBLE_W]
            
            # Calculate intensity
            mask = np.zeros(roi.shape, dtype="uint8")
            cv2.circle(mask, (BUBBLE_W//2, BUBBLE_H//2), int(min(BUBBLE_W, BUBBLE_H) * 0.40), 255, -1)
            mean_intensity = cv2.mean(roi, mask=mask)[0] # 0 = Black, 255 = White
            
            fill_data.append((mean_intensity, opt_idx, box))
            
        # Determine darkest bubble
        fill_data.sort(key=lambda x: x[0])
        lightest_val = fill_data[-1][0]
        
        marked_indices = []
        marked_boxes = []
        
        for data in fill_data:
            intensity, idx, box = data
            # Contrast threshold: Must be noticeably darker than the blank paper
            if intensity < (lightest_val * 0.82):
                marked_indices.append(idx)
                marked_boxes.append(box)

        correct_ans_ai = ANS_KEY.get(q_number) - 1 
        status = ""
        selected_human = "-"

        # --- STRICT MUTUALLY EXCLUSIVE GRADING ---
        if len(marked_indices) == 0:
            results["blank"] += 1
            status = "Blank"
            if show_missed:
                draw_box(row_boxes[correct_ans_ai], COLOR_BLUE, 2)
                
        elif len(marked_indices) > 1:
            results["double"] += 1
            results["wrong"] += 1
            status = "Double Marked"
            selected_human = "Multiple"
            for box in marked_boxes:
                draw_box(box, COLOR_YELLOW, 3)
            if show_missed and correct_ans_ai not in marked_indices:
                draw_box(row_boxes[correct_ans_ai], COLOR_BLUE, 2)
                
        elif len(marked_indices) == 1:
            student_ans = marked_indices[0]
            student_box = marked_boxes[0]
            selected_human = OPTS.get(student_ans, "-")
            
            if student_ans == correct_ans_ai:
                results["correct"] += 1
                status = "Correct"
                draw_box(student_box, COLOR_GREEN, 3) 
            else:
                results["wrong"] += 1
                status = "Incorrect"
                draw_box(student_box, COLOR_RED, 3) 
                if show_missed:
                    draw_box(row_boxes[correct_ans_ai], COLOR_BLUE, 2)
        
        breakdown_data.append({
            "Q No.": str(q_number),
            "Selected": selected_human,
            "Correct Answer": OPTS.get(correct_ans_ai, "-"),
            "Status": status
        })

    return results, warped_color, breakdown_data, "Success"


# --- STREAMLIT UI ---
st.title("🏆 Mathematical OMR Grader")
st.markdown("Fully deterministic grid-based pipeline for 60 Questions. 0% AI guessing.")

with st.sidebar:
    st.header("⚙️ Configuration")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=True)
    st.divider()
    st.info("💡 **Math-Based Engine:** Bypasses unpredictable contour detection. Uses coordinate geometry locked to a 1200x1600 matrix.")

uploaded_file = st.file_uploader("Upload OMR Document (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            img_np = load_document(uploaded_file)
            output = process_omr_engine(img_np, show_missed=show_missed)
            
            data, processed_img, breakdown, msg = output
            
            pos = data['correct'] * CORRECT_PTS
            neg = data['wrong'] * WRONG_PTS
            total = pos - neg
            acc_percent = (data['correct'] / 60) * 100
            
            st.markdown("### 📊 Official Scorecard")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Correct", data['correct'], f"+{pos} pts")
            m2.metric("Incorrect", data['wrong'], f"-{neg} pts")
            m3.metric("Blank", data['blank'])
            m4.metric("Double Marked", data['double'], help="Counts as Incorrect")
            m5.metric("FINAL SCORE", total)
            st.progress(acc_percent / 100, text=f"Overall Accuracy: {acc_percent:.1f}%")
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs(["📝 Graded Sheet", "📈 Analytics", "⚙️ Data Table"])
            
            with tab1:
                col_img, col_leg = st.columns([3, 1])
                with col_img:
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_leg:
                    st.write("### Color Key")
                    st.success("🟩 **Correct**")
                    st.error("🟥 **Incorrect**")
                    st.info("🟦 **Missed Answer**")
                    st.warning("🟨 **Double Mark**")
                    st.write("⬜ **Gray Box**: Grid Calibration")
                    
            with tab2:
                chart_data = pd.DataFrame({
                    "Category": ["Correct", "Incorrect", "Blank/Double"],
                    "Count": [data['correct'], data['wrong'], data['blank'] + data['double']]
                })
                st.bar_chart(chart_data, x="Category", y="Count", color="#2a9df4")
                
            with tab3:
                df = pd.DataFrame(breakdown)
                def color_status(val):
                    if val == 'Correct': return 'color: #28a745; font-weight: bold;'
                    if val == 'Incorrect': return 'color: #dc3545; font-weight: bold;'
                    if val == 'Double Marked': return 'color: #ffc107; font-weight: bold;'
                    return 'color: gray;'
                styled_df = df.style.map(color_status, subset=['Status'])
                st.dataframe(styled_df, hide_index=True, use_container_width=True, height=600)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download CSV", data=csv, file_name="OMR_Results.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"Critical System Exception: {str(e)}")
            st.exception(e)
