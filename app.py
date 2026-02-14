import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Enterprise Grader", layout="wide", page_icon="📝")

# --- RULES & ANSWER KEY ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted

# NORMAL FORMAT: Question No: Correct Option (1=A, 2=B, 3=C, 4=D)
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

def get_document_corners(image):
    """
    Finds the 4 corners of the paper.
    If it's a perfect PDF upload with no background, it uses the image corners itself!
    """
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
            # Must have 4 points and take up at least 20% of the image area
            if len(approx) == 4 and cv2.contourArea(approx) > (image.shape[0] * image.shape[1] * 0.2):
                doc_cnt = approx
                break

    # FALLBACK: If we can't find a paper edge (like in a PDF upload), use the corners of the image itself!
    if doc_cnt is None:
        h, w = image.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
        return corners

    return doc_cnt.reshape(4, 2)

def process_omr_engine(image_np, show_missed=True):
    # 1. Base Alignment using the 4 Page Corners
    orig = imutils.resize(image_np, height=1500)
    corner_pts = get_document_corners(orig)

    # Standardize image to an exact high-res canvas (1200x1600)
    warped_gray = four_point_transform(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    # 2. Extract Questions Area (Skips the Roll No header automatically)
    ROI_TOP = 350
    ROI_BOTTOM = 1580
    roi_gray = warped_gray[ROI_TOP:ROI_BOTTOM, :]
    
    thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Find exactly 240 Bubbles
    bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        # Search for circular options
        if 15 <= w <= 55 and 15 <= h <= 55 and 0.6 <= ar <= 1.4:
            # Shift Y coordinate back to match the original 1200x1600 image
            bubbles.append((x, y + ROI_TOP, w, h))
            
    # Filter Bubbles
    if len(bubbles) > 240:
        bubbles = sorted(bubbles, key=lambda b: abs(1.0 - (b[2]/float(b[3]))))[:240]
    
    if len(bubbles) != 240:
        # ALWAYS SHOW THE IMAGE, even on failure, so user can see what went wrong
        for (bx, by, bw, bh) in bubbles:
            cv2.rectangle(warped_color, (bx, by), (bx+bw, by+bh), (255, 0, 255), 2)
        return None, warped_color, None, f"Detection Error: Found {len(bubbles)}/240 bubbles. Please ensure the scan is clear."

    # 4. Smart Dynamic Macro-Sorting
    # Sort all 240 bubbles Left-to-Right to split into the 3 columns
    bubbles = sorted(bubbles, key=lambda b: b[0])
    columns = [bubbles[0:80], bubbles[80:160], bubbles[160:240]]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown_data = []
    q_number = 1
    
    # 5. Grading Loop
    for col in columns:
        # Sort Top-to-Bottom for this column
        col = sorted(col, key=lambda b: b[1])
        
        # Process 4 bubbles at a time (One Question)
        for i in range(0, 80, 4):
            # Sort the 4 bubbles Left-to-Right (A, B, C, D)
            row = sorted(col[i:i+4], key=lambda b: b[0])
            
            fill_data = []
            for j, (bx, by, bw, bh) in enumerate(row):
                # Isolate the core 35% of the bubble to ignore outlines
                roi = warped_gray[by:by+bh, bx:bx+bw]
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.circle(mask, (bw//2, bh//2), int(min(bw, bh) * 0.35), 255, -1)
                
                mean_intensity = cv2.mean(roi, mask=mask)[0] # 0 = Black, 255 = White
                fill_data.append((mean_intensity, j, (bx, by, bw, bh)))

            fill_data.sort(key=lambda x: x[0])
            darkest_val = fill_data[0][0]
            lightest_val = fill_data[-1][0]
            
            # Determine marked bubbles based on contrast ratio
            marked_indices = []
            marked_boxes = []
            
            for data in fill_data:
                intensity, idx, box = data
                # It must be substantially darker than the blank paper to count
                if intensity < (lightest_val * 0.82):
                    marked_indices.append(idx)
                    marked_boxes.append(box)

            correct_ans_ai = ANS_KEY.get(q_number) - 1 
            
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
                    draw_box(student_box, COLOR_GREEN, 3) 
                    # Note: No blue box will ever be drawn here
                else:
                    # 4. INCORRECT
                    results["wrong"] += 1
                    status = "Incorrect"
                    draw_box(student_box, COLOR_RED, 3) 
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
st.title("🏆 Yuva Gyan Enterprise Grader")
st.markdown("Fully Automated OMR Pipeline for 60 Questions.")

with st.sidebar:
    st.header("⚙️ Configuration")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=True, help="Draws a blue box on the correct answer if the student was wrong or left it blank.")
    
    st.divider()
    st.info("💡 **How it works:**\n1. Upload a PDF or Image.\n2. The AI natively uses the 4 page corners to flatten the document.\n3. It isolates all 240 bubbles automatically.\n4. Evaluates via Mutually Exclusive logic.")

uploaded_file = st.file_uploader("Upload OMR Document (PDF, JPG, PNG)", type=['pdf', 'jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            img_np = load_document(uploaded_file)
            output = process_omr_engine(img_np, show_missed=show_missed)
            
            # Unpack the safe output (which always contains the image now)
            data, processed_img, breakdown, msg = output
            
            if data is None:
                st.error(f"⚠️ **{msg}**")
                st.write("### 🔍 AI Diagnostic View")
                st.write("The purple boxes below show what the AI *thought* were bubbles. If there aren't exactly 240, check the image quality.")
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
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
