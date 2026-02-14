import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Enterprise Grader", layout="wide", page_icon="🏅")

# --- RULES & ANSWER KEY ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted

# Format: Question Number (1-60) : Correct Option (1=A, 2=B, 3=C, 4=D)
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}

# --- COLORS (BGR format for OpenCV) ---
COLOR_GREEN = (0, 200, 0)     # Correct
COLOR_RED = (0, 0, 255)       # Incorrect
COLOR_BLUE = (255, 0, 0)      # Missed Correct Answer
COLOR_YELLOW = (0, 255, 255)  # Double Marked

def load_document(uploaded_file):
    """Safely extracts the first page of a PDF or loads an image."""
    if uploaded_file.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img = Image.open(uploaded_file).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_quadrant_fiducials(gray):
    """NEW LOGIC: Forces the AI to find exactly one marker per quadrant."""
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    
    # Adaptive threshold specifically for black printed ink
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Define the 4 geometric quadrants
    quads = {"TL": [], "TR": [], "BL": [], "BR": []}
    
    for c in cnts:
        area = cv2.contourArea(c)
        if 50 < area < 20000:
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / float(bh)
            if 0.5 <= ar <= 1.5:  # Roughly square/circular
                M = cv2.moments(c)
                if M["m00"] > 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    
                    # Sort into quadrants
                    if px < cx and py < cy: quads["TL"].append((area, [px, py]))
                    elif px >= cx and py < cy: quads["TR"].append((area, [px, py]))
                    elif px < cx and py >= cy: quads["BL"].append((area, [px, py]))
                    elif px >= cx and py >= cy: quads["BR"].append((area, [px, py]))

    # Grab the largest contour in each quadrant
    try:
        tl = sorted(quads["TL"], key=lambda x: x[0], reverse=True)[0][1]
        tr = sorted(quads["TR"], key=lambda x: x[0], reverse=True)[0][1]
        bl = sorted(quads["BL"], key=lambda x: x[0], reverse=True)[0][1]
        br = sorted(quads["BR"], key=lambda x: x[0], reverse=True)[0][1]
        return np.array([tl, tr, br, bl], dtype="float32")
    except IndexError:
        return None  # Failed to find a marker in one or more quadrants

def process_omr_engine(image_np, show_missed=True):
    # 1. Base Alignment
    orig = imutils.resize(image_np, height=1500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    corner_pts = get_quadrant_fiducials(gray)
    if corner_pts is None:
        return None, "Error: Could not detect the 4 corner alignment markers. Ensure the page is flat and fully visible."

    # Standardize image to an exact high-res canvas (1200x1600)
    warped_gray = four_point_transform(gray, corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    # Crop out the top header (Name, Roll No, etc.) to isolate questions
    ROI_TOP = 350
    ROI_BOTTOM = 1550
    roi_gray = warped_gray[ROI_TOP:ROI_BOTTOM, :]
    
    # Threshold strictly to find bubble outlines
    thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Extract strictly 240 Bubbles
    bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        # Filters to find perfectly sized circular bubbles
        if 15 <= w <= 55 and 15 <= h <= 55 and 0.6 <= ar <= 1.4:
            # Shift Y coordinate back to match the original 1200x1600 image
            bubbles.append((x, y + ROI_TOP, w, h))
            
    if len(bubbles) > 240:
        # Sort by circularity to drop any non-bubble noise
        bubbles = sorted(bubbles, key=lambda b: abs(1.0 - (b[2]/float(b[3]))))[:240]
    elif len(bubbles) < 240:
        return None, f"Detection Error: Only found {len(bubbles)}/240 bubbles. Ensure the paper is well lit and not blurry."

    # 3. Dynamic Macro-Sorting (No fixed pixel columns)
    # Sort all 240 bubbles Left-to-Right
    bubbles = sorted(bubbles, key=lambda b: b[0])
    
    # Split cleanly into 3 columns (80 bubbles each)
    col1 = bubbles[0:80]
    col2 = bubbles[80:160]
    col3 = bubbles[160:240]
    
    columns = [col1, col2, col3]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown_data = []
    q_number = 1
    
    # 4. Grading Loop
    for col in columns:
        # Sort the 80 bubbles in this column Top-to-Bottom
        col = sorted(col, key=lambda b: b[1])
        
        # Process 4 bubbles at a time (One Question)
        for i in range(0, 80, 4):
            # Grab the 4 bubbles and ensure they are sorted Left-to-Right (A, B, C, D)
            row = sorted(col[i:i+4], key=lambda b: b[0])
            
            fill_data = []
            for j, (bx, by, bw, bh) in enumerate(row):
                # Isolate the exact core of the bubble on the raw grayscale image
                roi = warped_gray[by:by+bh, bx:bx+bw]
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.circle(mask, (bw//2, bh//2), int(min(bw, bh) * 0.35), 255, -1)
                
                # Calculate darkness (0 = Black, 255 = White)
                mean_intensity = cv2.mean(roi, mask=mask)[0]
                fill_data.append((mean_intensity, j, (bx, by, bw, bh)))

            # Sort by darkest value (lowest number)
            fill_data.sort(key=lambda x: x[0])
            
            darkest_val = fill_data[0][0]
            lightest_val = fill_data[-1][0]
            
            # Determine which bubbles are actually filled by the student
            marked_indices = []
            marked_boxes = []
            
            for data in fill_data:
                intensity, idx, box = data
                # Dynamic Threshold: If an option is significantly darker than the empty paper, it's marked
                # Formula: It must be at least 20% darker than the lightest bubble in the row
                if intensity < (lightest_val * 0.80):
                    marked_indices.append(idx)
                    marked_boxes.append(box)

            # Get Correct Answer (Convert 1-4 to 0-3 index)
            correct_ans_ai = ANS_KEY.get(q_number) - 1 
            
            # Helper to draw boxes
            def draw_box(b_box, color, thickness=3):
                cv2.rectangle(warped_color, (b_box[0], b_box[1]), (b_box[0]+b_box[2], b_box[1]+b_box[3]), color, thickness)

            status = ""
            selected_human = "-"

            # --- STRICT MUTUALLY EXCLUSIVE GRADING ---
            if len(marked_indices) == 0:
                # 1. BLANK
                results["blank"] += 1
                status = "Blank"
                if show_missed:
                    draw_box(row[correct_ans_ai], COLOR_BLUE, 2)
                    
            elif len(marked_indices) > 1:
                # 2. DOUBLE MARKED
                results["double"] += 1
                results["wrong"] += 1
                status = "Double Marked"
                selected_human = "Multiple"
                for box in marked_boxes:
                    draw_box(box, COLOR_YELLOW, 3)
                if show_missed and correct_ans_ai not in marked_indices:
                    draw_box(row[correct_ans_ai], COLOR_BLUE, 2)
                    
            elif len(marked_indices) == 1:
                student_ans = marked_indices[0]
                student_box = marked_boxes[0]
                selected_human = OPTS.get(student_ans, "-")
                
                if student_ans == correct_ans_ai:
                    # 3. CORRECT
                    results["correct"] += 1
                    status = "Correct"
                    draw_box(student_box, COLOR_GREEN, 3) # STRICTLY ONLY GREEN
                else:
                    # 4. INCORRECT
                    results["wrong"] += 1
                    status = "Incorrect"
                    draw_box(student_box, COLOR_RED, 3) # STRICTLY ONLY RED
                    if show_missed:
                        draw_box(row[correct_ans_ai], COLOR_BLUE, 2)
            
            breakdown_data.append({
                "Q No.": str(q_number),
                "Selected": selected_human,
                "Correct Answer": OPTS.get(correct_ans_ai, "-"),
                "Status": status
            })
            
            q_number += 1

    return results, warped_color, breakdown_data, "Success"


# --- STREAMLIT UI ---
st.title("🏅 Yuva Gyan Enterprise Grader")
st.markdown("Fully Autonomous. Mutually Exclusive Logic. Zero Hardcoding.")

with st.sidebar:
    st.header("⚙️ Configuration")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=True, help="Draws a blue box on the correct answer if the student was wrong or left it blank.")
    
    st.divider()
    st.info("💡 **How it works:**\n1. Upload a PDF or Image.\n2. The AI natively finds the 4 corner marks.\n3. It isolates all 240 bubbles dynamically without sliders.\n4. It evaluates purely on mathematical grayscale thresholding.")

uploaded_file = st.file_uploader("Upload OMR Document (PDF, JPG, PNG)", type=['pdf', 'jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            img_np = load_document(uploaded_file)
            output = process_omr_engine(img_np, show_missed=show_missed)
            
            if output[0] is None:
                st.error(f"⚠️ **Engine Failure:** {output[1]}")
            else:
                data, processed_img, breakdown, _ = output
                pos = data['correct'] * CORRECT_PTS
                neg = data['wrong'] * WRONG_PTS
                total = pos - neg
                acc_percent = (data['correct'] / 60) * 100
                
                # --- SCORECARD METRICS ---
                st.markdown("### 📊 Official Scorecard")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Correct", data['correct'], f"+{pos} pts")
                m2.metric("Incorrect", data['wrong'], f"-{neg} pts")
                m3.metric("Blank", data['blank'])
                m4.metric("Double Marked", data['double'], help="Counts as Incorrect")
                m5.metric("FINAL SCORE", total)
                st.progress(acc_percent / 100, text=f"Overall Accuracy: {acc_percent:.1f}%")
                st.markdown("---")
                
                # --- TABS ---
                tab1, tab2, tab3 = st.tabs(["📝 Graded Sheet", "📈 Visual Analytics", "⚙️ Data Table & Export"])
                
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
                        
                with tab2:
                    st.write("#### Performance Analytics")
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
                    st.download_button(label="📥 Download Results as CSV", data=csv, file_name="OMR_Results.csv", mime="text/csv")
                    
        except Exception as e:
            st.error(f"Critical System Exception: {str(e)}")
            st.exception(e)
