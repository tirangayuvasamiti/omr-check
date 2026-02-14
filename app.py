import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Yuva Gyan Ultra Grader", layout="wide", page_icon="⚡")

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

def find_corner_fiducials(gray):
    """New Engine: Finds the 4 corner machine marks by searching the 4 extremes."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Aggressive thresholding for black marks on white paper
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 50 < area < 25000:  # Broad range to catch big/small markers
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if 0.5 <= ar <= 1.5:  # Roughly square/circular
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) > 0:
                    solidity = area / cv2.contourArea(hull)
                    if solidity > 0.7:  # Solid black filled shape
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            candidates.append((cx, cy))
                            
    if len(candidates) < 4:
        return None

    # Sort candidates into the 4 corners mathematically
    h, w = gray.shape
    corners = [(0, 0), (w, 0), (w, h), (0, h)] # TL, TR, BR, BL
    final_pts = []
    
    for corner in corners:
        # Find the candidate with the shortest geometric distance to the actual image corner
        best_pt = min(candidates, key=lambda p: (p[0] - corner[0])**2 + (p[1] - corner[1])**2)
        final_pts.append([best_pt[0], best_pt[1]])
        
    return np.array(final_pts, dtype="float32")

def process_omr_ultra(image_np, show_missed=False):
    # 1. Image Prep
    orig = imutils.resize(image_np, height=1200)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # 2. Fiducial Warp (Corner Marks)
    corner_pts = find_corner_fiducials(gray)
    if corner_pts is None:
        return None, "Error: Could not find all 4 corner machine marks. Ensure they are visible."

    # Warp to a strict, standardized size (Width: 1200, Height: 1600)
    warped_gray = four_point_transform(gray, corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    # 3. Thresholding for Bubbles
    thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)

    # 4. Vertical Slicing Mechanism
    # The sheet has 3 columns. We slice the image into 3 vertical zones to prevent row cross-contamination.
    slices = [
        {"x_start": 0, "x_end": 400, "q_start": 1, "q_end": 20},
        {"x_start": 400, "x_end": 800, "q_start": 21, "q_end": 40},
        {"x_start": 800, "x_end": 1200, "q_start": 41, "q_end": 60}
    ]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    debug_img = warped_color.copy()
    
    for s_idx, sl in enumerate(slices):
        slice_thresh = thresh[:, sl["x_start"]:sl["x_end"]]
        
        # Find bubbles in this specific slice
        cnts, _ = cv2.findContours(slice_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h)
            if 15 <= w <= 55 and 15 <= h <= 55 and 0.6 <= ar <= 1.4:
                # Add offset back so coordinates match the main image
                bubbles.append((x + sl["x_start"], y, w, h, c))
                
        # Filter to exact 80 bubbles (20 questions * 4 options)
        if len(bubbles) > 80:
            # Sort by circularity to drop noise
            bubbles = sorted(bubbles, key=lambda b: abs(1.0 - (b[2]/float(b[3]))))[:80]
        elif len(bubbles) < 80:
            return None, f"Error in Column {s_idx + 1}: Found {len(bubbles)}/80 bubbles. Ensure column is clear."

        # Sort top-to-bottom
        bubbles = sorted(bubbles, key=lambda b: b[1])
        
        # Grade Row by Row
        current_q = sl["q_start"]
        for i in range(0, 80, 4):
            # Take a row of 4 options and sort left-to-right
            row = bubbles[i:i+4]
            row = sorted(row, key=lambda b: b[0])
            
            pixel_counts = []
            for j, (bx, by, bw, bh, bc) in enumerate(row):
                # Inner Core Mask: Only look at the very center of the bubble!
                mask = np.zeros(warped_gray.shape, dtype="uint8")
                # Draw a circle 40% the size of the bounding box at the center
                cv2.circle(mask, (int(bx + bw/2), int(by + bh/2)), int(min(bw, bh) * 0.35), 255, -1)
                
                core_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
                total_pixels = cv2.countNonZero(core_thresh)
                pixel_counts.append((total_pixels, j, (bx, by, bw, bh)))
                
                # Draw diagnostics
                cv2.circle(debug_img, (int(bx + bw/2), int(by + bh/2)), int(min(bw, bh) * 0.35), (255, 0, 255), 1)

            pixel_counts.sort(key=lambda x: x[0], reverse=True)
            darkest_val, darkest_idx, darkest_box = pixel_counts[0]
            second_darkest_val = pixel_counts[1][0]
            
            correct_ans_human = ANS_KEY.get(current_q)
            correct_ans_ai = correct_ans_human - 1 # Convert to 0-3 index
            
            # Draw contour helper
            def draw_box(b_box, color, thickness=3):
                cv2.rectangle(warped_color, (b_box[0], b_box[1]), (b_box[0]+b_box[2], b_box[1]+b_box[3]), color, thickness)

            # Grading Logic
            if darkest_val < 50: 
                # Inner core has less than 50 filled pixels = BLANK
                results["blank"] += 1
                if show_missed:
                    draw_box(row[correct_ans_ai][0:4], (255, 0, 0), 2) # Blue
                    
            elif second_darkest_val > (darkest_val * 0.65): 
                # DOUBLE BUBBLE
                results["double"] += 1
                results["wrong"] += 1
                draw_box(darkest_box, (0, 255, 255)) # Yellow
                draw_box(pixel_counts[1][2], (0, 255, 255))
                if show_missed:
                    draw_box(row[correct_ans_ai][0:4], (255, 0, 0), 2)
                    
            elif darkest_idx == correct_ans_ai:
                results["correct"] += 1
                draw_box(darkest_box, (0, 255, 0)) # Green
                
            else:
                results["wrong"] += 1
                draw_box(darkest_box, (0, 0, 255)) # Red
                if show_missed:
                    draw_box(row[correct_ans_ai][0:4], (255, 0, 0), 2) # Blue
            
            current_q += 1

    return results, warped_color, debug_img, "Success"

# --- UI ---
st.title("⚡ Yuva Gyan Ultra Grader")

with st.sidebar:
    st.header("⚙️ Output Settings")
    show_missed = st.toggle("Show Missed Answers (Blue)", value=True, help="Draws a blue box to show the right answer on mistakes.")
    st.divider()
    st.info("💡 **Instructions:**\n1. Use the Mobile Camera tab.\n2. Ensure the **4 black corner dots/squares** are clearly in the frame.\n3. The app will automatically isolate, slice, and read the inner cores of the bubbles.")

tab1, tab2 = st.tabs(["📱 Native Mobile Camera (Tap to focus)", "📸 Web Scanner"])

with tab1:
    upload_img = st.file_uploader("Take Photo or Upload Image", type=['jpg','png','jpeg'])

with tab2:
    camera_img = st.camera_input("Live Camera", label_visibility="collapsed")

input_file = upload_img if upload_img is not None else camera_img

if input_file:
    img = Image.open(input_file).convert('RGB')
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Executing Ultra Engine Processing..."):
        try:
            output = process_omr_ultra(img_np, show_missed=show_missed)
            
            if output[0] is None:
                err_msg = output[1]
                st.error(f"⚠️ **Scan Failed:** {err_msg}")
            else:
                data, processed_img, debug_img, _ = output
                pos = data['correct'] * CORRECT_PTS
                neg = data['wrong'] * WRONG_PTS
                total = pos - neg
                
                # SCORECARD
                st.markdown("---")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Correct", data['correct'], f"+{pos}")
                m2.metric("Incorrect", data['wrong'], f"-{neg}")
                m3.metric("Blank", data['blank'])
                m4.metric("Double Marked", data['double'])
                m5.metric("FINAL SCORE", total)
                st.markdown("---")
                
                col_res, col_diag = st.columns([1, 1])
                with col_res:
                    st.write("### 📝 Graded Sheet")
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_diag:
                    st.write("### 🔍 AI Inner-Core Diagnostics")
                    st.write("*(The purple dots show the exact center where the AI checked for ink)*")
                    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Critical Engine Failure: {str(e)}")
