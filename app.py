import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import fitz  # PyMuPDF for PDF processing
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Enterprise Grader", layout="wide", page_icon="🏢")

# --- RULES & KEY ---
CORRECT_PTS = 3
WRONG_PTS = 1  # Subtracted

# NORMAL FORMAT: A=1, B=2, C=3, D=4
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D", -1: "BLANK", -2: "DOUBLE"}

def load_file_as_image(uploaded_file):
    """Handles both Image files and PDF files gracefully."""
    if uploaded_file.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: # RGBA to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def find_fiducials(gray):
    """Enterprise Fiducial Scanner with CLAHE Illumination Correction."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 100 < area < 30000: 
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if 0.5 <= ar <= 1.5: 
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) > 0:
                    solidity = area / cv2.contourArea(hull)
                    if solidity > 0.75:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            candidates.append([cx, cy])
                            
    if len(candidates) < 4:
        return None

    pts = np.array(candidates)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def process_omr_enterprise(image_np, show_missed=False, crop_top=400, crop_bottom=1500):
    orig = imutils.resize(image_np, height=1500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # 1. Align via Fiducials
    corner_pts = find_fiducials(gray)
    if corner_pts is None:
        return None, "Engine Failed: Could not locate the 4 dark corner marks. Please ensure the page is not cut off."

    # Standardize output resolution exactly to 1200x1600
    warped_gray = four_point_transform(gray, corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)

    # Vertical Slicing into 3 Columns
    slices = [
        {"x_start": 0, "x_end": 400, "q_start": 1, "q_end": 20},
        {"x_start": 400, "x_end": 800, "q_start": 21, "q_end": 40},
        {"x_start": 800, "x_end": 1200, "q_start": 41, "q_end": 60}
    ]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown_data = [] 
    
    debug_img = warped_color.copy()
    
    # Draw Green Target Zone for Diagnostics
    cv2.rectangle(debug_img, (0, crop_top), (1200, crop_bottom), (0, 255, 0), 4)
    
    for s_idx, sl in enumerate(slices):
        # NEW: Isolate strictly the region of interest (crop_top to crop_bottom)
        slice_thresh = thresh[crop_top:crop_bottom, sl["x_start"]:sl["x_end"]]
        cnts, _ = cv2.findContours(slice_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if 15 <= w <= 55 and 15 <= h <= 55 and 0.6 <= ar <= 1.4:
                # Add offset back so coordinates align with the main 1200x1600 image
                bubbles.append((x + sl["x_start"], y + crop_top, w, h, c))
                
        if len(bubbles) > 80:
            bubbles = sorted(bubbles, key=lambda b: abs(1.0 - (b[2]/float(b[3]))))[:80]
        elif len(bubbles) < 80:
            return None, f"Read Error in Column {s_idx + 1}: Found {len(bubbles)}/80 bubbles inside the green scanning zone. Adjust the Top/Bottom Crop sliders in the sidebar."

        bubbles = sorted(bubbles, key=lambda b: b[1])
        current_q = sl["q_start"]
        
        for i in range(0, 80, 4):
            # Sort 4 options left to right
            row = sorted(bubbles[i:i+4], key=lambda b: b[0])
            pixel_counts = []
            
            for j, (bx, by, bw, bh, bc) in enumerate(row):
                mask = np.zeros(warped_gray.shape, dtype="uint8")
                # Enterprise Inner-Core Masking (35% of bubble center)
                cv2.circle(mask, (int(bx + bw/2), int(by + bh/2)), int(min(bw, bh) * 0.35), 255, -1)
                
                core_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
                total_pixels = cv2.countNonZero(core_thresh)
                pixel_counts.append((total_pixels, j, (bx, by, bw, bh)))
                
                # Draw small purple dot to show where AI tested for ink
                cv2.circle(debug_img, (int(bx + bw/2), int(by + bh/2)), int(min(bw, bh) * 0.35), (255, 0, 255), 1)

            pixel_counts.sort(key=lambda x: x[0], reverse=True)
            darkest_val, darkest_idx, darkest_box = pixel_counts[0]
            second_darkest_val = pixel_counts[1][0]
            
            correct_ans_human = ANS_KEY.get(current_q)
            correct_ans_ai = correct_ans_human - 1 
            
            def draw_box(b_box, color, thickness=3):
                cv2.rectangle(warped_color, (b_box[0], b_box[1]), (b_box[0]+b_box[2], b_box[1]+b_box[3]), color, thickness)

            status = ""
            selected_human = OPTS[darkest_idx]

            if darkest_val < 50: 
                results["blank"] += 1
                status = "Blank"
                selected_human = "-"
                if show_missed:
                    draw_box(row[correct_ans_ai][0:4], (255, 0, 0), 2)
                    
            elif second_darkest_val > (darkest_val * 0.65): 
                results["double"] += 1
                results["wrong"] += 1
                status = "Double Marked"
                selected_human = "Multiple"
                draw_box(darkest_box, (0, 255, 255))
                draw_box(pixel_counts[1][2], (0, 255, 255))
                if show_missed:
                    draw_box(row[correct_ans_ai][0:4], (255, 0, 0), 2)
                    
            elif darkest_idx == correct_ans_ai:
                results["correct"] += 1
                status = "Correct"
                draw_box(darkest_box, (0, 255, 0))
                
            else:
                results["wrong"] += 1
                status = "Incorrect"
                draw_box(darkest_box, (0, 0, 255)) 
                if show_missed:
                    draw_box(row[correct_ans_ai][0:4], (255, 0, 0), 2)
            
            breakdown_data.append({
                "Q No.": current_q,
                "Selected": selected_human,
                "Correct Answer": OPTS[correct_ans_ai],
                "Status": status
            })
            
            current_q += 1

    return results, warped_color, debug_img, breakdown_data, "Success"

# --- UI ---
st.title("🏢 Yuva Gyan Enterprise Grader")
st.markdown("Industry-grade Optical Mark Recognition with PDF support, Area Masking, and CLAHE illumination correction.")

with st.sidebar:
    st.header("⚙️ Target Area Settings")
    st.write("Adjust these sliders to ensure the AI *only* scans the question area and ignores the top header / Roll No bubbles.")
    crop_top = st.slider("Top Crop Boundary", min_value=100, max_value=800, value=350, step=10, help="Drag down to exclude the Roll No/Header from the scan.")
    crop_bottom = st.slider("Bottom Crop Boundary", min_value=1200, max_value=1600, value=1550, step=10, help="Drag up to exclude footers/signatures.")
    
    st.divider()
    st.header("⚙️ Output Settings")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=True, help="Highlights the correct answer if the student was wrong.")
    show_diagnostics = st.toggle("Show AI Diagnostics Map", value=True, help="Shows the Green Target Zone and the exact spots the AI tested.")
    
    st.divider()
    st.info("💡 **Supported Formats:**\n\nPDF, JPG, PNG, JPEG.\n\n*Note: If a multi-page PDF is uploaded, only the first page is processed.*")

uploaded_file = st.file_uploader("Upload Scanned OMR Sheet (PDF or Image)", type=['pdf', 'jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing document via Enterprise Engine..."):
        try:
            img_np = load_file_as_image(uploaded_file)
            
            # Pass crop boundaries to the engine
            output = process_omr_enterprise(img_np, show_missed=show_missed, crop_top=crop_top, crop_bottom=crop_bottom)
            
            if output[0] is None:
                st.error(f"⚠️ **Scan Failed:** {output[1]}")
                if "Read Error" in output[1] and len(output) > 2:
                    st.write("### 🔍 AI Diagnostics Map")
                    st.warning("Look at the Bright Green Box below. Make sure it ONLY covers the 60 questions. Adjust the sliders on the left and try again!")
                    st.image(cv2.cvtColor(output[2], cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                data, processed_img, debug_img, breakdown, _ = output
                pos = data['correct'] * CORRECT_PTS
                neg = data['wrong'] * WRONG_PTS
                total = pos - neg
                
                # --- DASHBOARD METRICS ---
                st.markdown("### 📊 Official Scorecard")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Correct", data['correct'], f"+{pos} pts")
                m2.metric("Incorrect", data['wrong'], f"-{neg} pts")
                m3.metric("Blank", data['blank'])
                m4.metric("Double Marked", data['double'], help="-1 pt penalty")
                m5.metric("FINAL SCORE", total)
                st.markdown("---")
                
                # --- LAYOUT: Image vs Data Table ---
                col_img, col_data = st.columns([1.5, 1])
                
                with col_img:
                    st.write("#### 📝 Graded Sheet Verification")
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    if show_diagnostics:
                        st.write("#### 🔍 AI Diagnostics Map")
                        st.write("*(The Green Box shows the active scanning area)*")
                        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                with col_data:
                    st.write("#### 📈 Question Breakdown")
                    df = pd.DataFrame(breakdown)
                    
                    def color_status(val):
                        if val == 'Correct': return 'color: #28a745; font-weight: bold;'
                        if val == 'Incorrect': return 'color: #dc3545; font-weight: bold;'
                        if val == 'Double Marked': return 'color: #ffc107; font-weight: bold;'
                        return 'color: gray;'
                        
                    styled_df = df.style.map(color_status, subset=['Status'])
                    st.dataframe(styled_df, height=600, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="OMR_Results.csv",
                        mime="text/csv",
                    )
                    
        except Exception as e:
            st.error(f"Critical Engine Failure: {str(e)}")
            st.exception(e)
