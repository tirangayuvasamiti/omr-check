import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Enterprise Grader", layout="wide", page_icon="🚀")

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
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}

# BGR Colors for OpenCV
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)

def load_file_as_image(uploaded_file):
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
        img_np = np.array(img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def find_fiducials(gray):
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
                            candidates.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
                            
    if len(candidates) < 4: return None
    pts = np.array(candidates)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

def process_omr_enterprise(image_np, show_missed=False, crop_top=350, crop_bottom=1550):
    orig = imutils.resize(image_np, height=1500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    corner_pts = find_fiducials(gray)
    if corner_pts is None:
        return None, "Could not locate the 4 dark corner marks. Ensure the page is fully visible."

    warped_gray = four_point_transform(gray, corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    # Threshold strictly used for finding bubble outlines
    thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)

    slices = [
        {"x_start": 0, "x_end": 400, "q_start": 1, "q_end": 20},
        {"x_start": 400, "x_end": 800, "q_start": 21, "q_end": 40},
        {"x_start": 800, "x_end": 1200, "q_start": 41, "q_end": 60}
    ]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown_data = [] 
    
    for s_idx, sl in enumerate(slices):
        slice_thresh = thresh[crop_top:crop_bottom, sl["x_start"]:sl["x_end"]]
        cnts, _ = cv2.findContours(slice_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            # Generous bounding box to catch all bubbles perfectly
            if 12 <= w <= 65 and 12 <= h <= 65 and 0.5 <= ar <= 1.5:
                bubbles.append((x + sl["x_start"], y + crop_top, w, h))
                
        # Group bubbles into strict rows based on Y-coordinates
        bubbles = sorted(bubbles, key=lambda b: b[1])
        if len(bubbles) == 0:
            return None, f"No bubbles found in Column {s_idx + 1}. Check crop sliders."
            
        rows = []
        current_row = [bubbles[0]]
        for b in bubbles[1:]:
            if abs(b[1] - current_row[-1][1]) < 20: # Belongs to same row
                current_row.append(b)
            else:
                rows.append(sorted(current_row, key=lambda x: x[0]))
                current_row = [b]
        rows.append(sorted(current_row, key=lambda x: x[0]))

        # Filter strictly to rows containing exactly 4 options
        valid_rows = [r for r in rows if len(r) == 4]
        
        if len(valid_rows) != 20:
            return None, f"Found {len(valid_rows)}/20 complete questions in Column {s_idx + 1}. Please adjust Top/Bottom Crop sliders."

        current_q = sl["q_start"]
        
        for row in valid_rows:
            intensities = []
            for j, (bx, by, bw, bh) in enumerate(row):
                # 200% Accuracy: Check Inner Core Grayscale Darkness
                roi = warped_gray[by:by+bh, bx:bx+bw]
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.circle(mask, (bw//2, bh//2), int(min(bw, bh) * 0.35), 255, -1)
                
                # Mean intensity: 0 is Pure Black (Filled), 255 is Pure White (Empty)
                mean_intensity = cv2.mean(roi, mask=mask)[0]
                intensities.append((mean_intensity, j, (bx, by, bw, bh)))

            # Sort by darkest value first (lowest number)
            intensities.sort(key=lambda x: x[0])
            darkest_val, darkest_idx, darkest_box = intensities[0]
            second_val, second_idx, second_box = intensities[1]
            lightest_val = intensities[-1][0]
            
            # Grayscale Contrast Ratio
            ratio_1 = darkest_val / lightest_val if lightest_val > 0 else 1.0
            ratio_2 = second_val / lightest_val if lightest_val > 0 else 1.0
            
            correct_ans_ai = ANS_KEY.get(current_q) - 1 
            
            def draw_box(b_box, color, thickness=3):
                cv2.rectangle(warped_color, (b_box[0], b_box[1]), (b_box[0]+b_box[2], b_box[1]+b_box[3]), color, thickness)

            status = ""
            selected_human = OPTS[darkest_idx]

            # --- MUTUALLY EXCLUSIVE DRAWING LOGIC ---
            if ratio_1 > 0.82: 
                # Blank (Darkest bubble is almost as bright as the lightest)
                results["blank"] += 1
                status = "Blank"
                selected_human = "-"
                if show_missed:
                    draw_box(row[correct_ans_ai], COLOR_BLUE)
                    
            elif ratio_2 < 0.85 and (ratio_2 - ratio_1) < 0.12: 
                # Double Marked (Both bubbles are dark, and very close in darkness)
                results["double"] += 1
                results["wrong"] += 1
                status = "Double Marked"
                selected_human = "Multiple"
                draw_box(darkest_box, COLOR_YELLOW)
                draw_box(second_box, COLOR_YELLOW)
                # Only draw blue if the correct answer wasn't one of the yellow ones
                if show_missed and correct_ans_ai not in [darkest_idx, second_idx]:
                    draw_box(row[correct_ans_ai], COLOR_BLUE)
                    
            elif darkest_idx == correct_ans_ai:
                # Correct Answer
                results["correct"] += 1
                status = "Correct"
                draw_box(darkest_box, COLOR_GREEN) # ONLY Green is drawn here
                
            else:
                # Incorrect Answer
                results["wrong"] += 1
                status = "Incorrect"
                draw_box(darkest_box, COLOR_RED) 
                if show_missed:
                    draw_box(row[correct_ans_ai], COLOR_BLUE) 
            
            breakdown_data.append({
                "Q No.": current_q,
                "Selected": selected_human,
                "Correct": OPTS[correct_ans_ai],
                "Status": status
            })
            
            current_q += 1

    return results, warped_color, breakdown_data, "Success"

# --- UI ---
st.title("🚀 Yuva Gyan Enterprise Grader")
st.markdown("Powered by Grayscale Contrast Engine & Strict Row Alignment.")

with st.sidebar:
    st.header("⚙️ Target Area Settings")
    st.write("Adjust sliders to ensure the AI *only* scans the question area.")
    crop_top = st.slider("Top Crop Boundary", min_value=100, max_value=800, value=350, step=10)
    crop_bottom = st.slider("Bottom Crop Boundary", min_value=1200, max_value=1600, value=1550, step=10)
    
    st.divider()
    st.header("⚙️ Output Settings")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=True, help="Draws blue on correct answers if student was wrong.")

uploaded_file = st.file_uploader("Upload Scanned OMR Sheet (PDF or Image)", type=['pdf', 'jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing document via Enterprise Engine..."):
        try:
            img_np = load_file_as_image(uploaded_file)
            output = process_omr_enterprise(img_np, show_missed=show_missed, crop_top=crop_top, crop_bottom=crop_bottom)
            
            if output[0] is None:
                st.error(f"⚠️ **Scan Failed:** {output[1]}")
                st.info("Try adjusting the Crop Sliders on the left menu.")
            else:
                data, processed_img, breakdown, _ = output
                pos = data['correct'] * CORRECT_PTS
                neg = data['wrong'] * WRONG_PTS
                total = pos - neg
                acc_percent = (data['correct'] / 60) * 100
                
                # --- DASHBOARD METRICS ---
                st.markdown("### 📊 Official Scorecard")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Correct", data['correct'], f"+{pos} pts")
                m2.metric("Incorrect", data['wrong'], f"-{neg} pts")
                m3.metric("Blank", data['blank'])
                m4.metric("Double Marked", data['double'], help="Counts as incorrect")
                m5.metric("FINAL SCORE", total)
                st.progress(acc_percent / 100, text=f"Overall Accuracy: {acc_percent:.1f}%")
                st.markdown("---")
                
                # --- TABBED LAYOUT ---
                tab1, tab2, tab3 = st.tabs(["📝 Graded Sheet", "📈 Analytics", "⚙️ Raw Data & Export"])
                
                with tab1:
                    col_img, col_leg = st.columns([3, 1])
                    with col_img:
                        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with col_leg:
                        st.write("### Legend")
                        st.info("🟢 Correct\n\n🔴 Incorrect\n\n🔵 Missed Answer\n\n🟡 Double Mark")
                        
                with tab2:
                    st.write("#### Performance Analytics")
                    chart_data = pd.DataFrame({
                        "Category": ["Correct", "Incorrect", "Blank/Double"],
                        "Count": [data['correct'], data['wrong'], data['blank'] + data['double']]
                    })
                    st.bar_chart(chart_data, x="Category", y="Count", color="#0099ff")
                    
                with tab3:
                    df = pd.DataFrame(breakdown)
                    # Force 'Q No.' to be a string so it displays cleanly without commas
                    df['Q No.'] = df['Q No.'].astype(str)
                    
                    def color_status(val):
                        if val == 'Correct': return 'color: #28a745; font-weight: bold;'
                        if val == 'Incorrect': return 'color: #dc3545; font-weight: bold;'
                        if val == 'Double Marked': return 'color: #ffc107; font-weight: bold;'
                        return 'color: gray;'
                        
                    styled_df = df.style.map(color_status, subset=['Status'])
                    
                    # hide_index=True instantly removes the "0, 1, 2" column on the left!
                    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=600)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Results as CSV", data=csv, file_name="OMR_Results.csv", mime="text/csv")
                    
        except Exception as e:
            st.error(f"Critical Engine Failure: {str(e)}")
            st.exception(e)
