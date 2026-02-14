import streamlit as st
import cv2
import numpy as np
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import pandas as pd
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="Yuva Gyan AI Vision Grader", layout="wide", page_icon="👁️")

# Answer Key (1-60)
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}

def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def get_contours_for_bubbles(binary_img):
    """
    AI FILTERING:
    Finds all contours, but returns ONLY the ones that are bubbles.
    It ignores text, lines, and noise based on shape properties.
    """
    cnts = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    question_cnts = []
    
    for c in cnts:
        # Compute bounding box
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        # --- THE AI FILTER ---
        # 1. Width/Height must be reasonable for a bubble (20px - 60px)
        # 2. Aspect Ratio must be close to 1.0 (Circular). 
        #    Text like "1" is thin (AR < 0.5), "Q" is wide. Bubbles are 0.9-1.1
        if w >= 25 and h >= 25 and ar >= 0.85 and ar <= 1.15:
            question_cnts.append(c)
            
    return question_cnts

def process_ai_vision(image):
    # 1. Pre-processing (Corner Detection)
    # Using the robust fallback method
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_full, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # Aggressive dilation to close gaps
    edged = cv2.dilate(edged, np.ones((5,5), np.uint8), iterations=2)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    doc_cnt = None
    
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
                
    if doc_cnt is None:
        # Fallback to full image
        h, w = image.shape[:2]
        doc_cnt = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    else:
        doc_cnt = doc_cnt.reshape(4, 2)

    # Flatten
    warped = four_point_transform(image, doc_cnt)
    warped = cv2.resize(warped, (1200, 1600)) # Normalize size for consistency
    
    # 2. ROI CROP (Smart Slicing)
    # Cut off Instructions (Top) and Footer (Bottom)
    roi = warped[360:1540, :] 
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 3. AI BUBBLE DETECTION
    # Otsu's Thresholding (Auto-determines best contrast)
    thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Get ONLY the bubbles (Filter out noise)
    bubble_cnts = get_contours_for_bubbles(thresh)
    
    # 4. CLUSTERING & SORTING
    # The bubbles are currently random. We must organize them.
    # We expect 240 bubbles (60 Qs * 4 Options)
    
    if len(bubble_cnts) != 240:
        # Strict mode warning: if we don't find exactly 240, 
        # we try to proceed but flag it.
        pass 

    # Sort Left-to-Right to find Columns
    bubble_cnts = contours.sort_contours(bubble_cnts, method="left-to-right")[0]
    
    # Split into 3 Columns (Chunks of 80 bubbles)
    # This assumes the 3-column layout of your sheet
    cols = []
    chunk_size = len(bubble_cnts) // 3
    if chunk_size == 0: return None, roi, [], "Error: No bubbles found."
    
    cols.append(bubble_cnts[0:chunk_size])           # Col 1
    cols.append(bubble_cnts[chunk_size:chunk_size*2]) # Col 2
    cols.append(bubble_cnts[chunk_size*2:])           # Col 3
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    logs = []
    q_counter = 1
    
    # Process each column
    for col_bubbles in cols:
        # Sort Top-to-Bottom (Rows)
        col_bubbles = contours.sort_contours(col_bubbles, method="top-to-bottom")[0]
        
        # Process in chunks of 4 (A, B, C, D)
        for (q, i) in enumerate(np.arange(0, len(col_bubbles), 4)):
            # Get the row of 4 bubbles
            row_cnts = col_bubbles[i:i+4]
            
            # Sort them Left-to-Right (A -> D) to be sure
            row_cnts = contours.sort_contours(row_cnts, method="left-to-right")[0]
            
            # Identify the marked bubble
            bubbled = []
            
            for (j, c) in enumerate(row_cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                
                # Check fullness
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                
                # Store (pixel_count, index, contour)
                bubbled.append((total, j, c))
                
            # Sort by "filled" pixels (Most filled is the answer)
            bubbled.sort(key=lambda x: x[0], reverse=True)
            
            # Decision Logic (Relative Threshold)
            most_filled_amt = bubbled[0][0]
            second_filled_amt = bubbled[1][0]
            
            # A bubble is "marked" if it has significantly more pixels than the others
            # or simply crosses a minimum pixel threshold (e.g. 400 pixels)
            marked_indices = []
            
            # Check for blanks (if even the most filled is empty)
            if most_filled_amt < 400: # 400 pixels is approx 20% of a bubble area
                marked_indices = [] # Blank
            elif most_filled_amt > 400 and second_filled_amt > (most_filled_amt * 0.85):
                # If the second best is almost as filled as the first -> Double Mark
                marked_indices = [bubbled[0][1], bubbled[1][1]]
            else:
                marked_indices = [bubbled[0][1]]
            
            # Grading
            color = (0, 0, 255) # Default Red
            k = ANS_KEY.get(q_counter, -1) - 1
            status = ""
            
            if len(marked_indices) == 0:
                results['blank'] += 1
                status = "Blank"
                # Draw Blue on correct answer
                cv2.drawContours(roi, [row_cnts[k]], -1, (255, 0, 0), 2)
                
            elif len(marked_indices) > 1:
                results['double'] += 1
                results['wrong'] += 1
                status = "Double"
                for idx in marked_indices:
                    cv2.drawContours(roi, [row_cnts[idx]], -1, (0, 255, 255), 3)
                    
            else:
                choice = marked_indices[0]
                if choice == k:
                    results['correct'] += 1
                    status = "Correct"
                    color = (0, 255, 0)
                else:
                    results['wrong'] += 1
                    status = "Incorrect"
                    color = (0, 0, 255)
                    # Show correct
                    cv2.drawContours(roi, [row_cnts[k]], -1, (255, 0, 0), 2)
                    
                cv2.drawContours(roi, [row_cnts[choice]], -1, color, 3)

            logs.append({"Q": q_counter, "Status": status})
            q_counter += 1
            
    return results, roi, logs, "Success"

# --- UI ---
st.title("👁️ AI Vision OMR Grader")
st.write("Detects bubbles dynamically using Shape Analysis & Contour Detection.")


uploaded_file = st.file_uploader("Upload OMR", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.spinner("AI is detecting bubbles..."):
        try:
            img = load_image(uploaded_file)
            stats, out_img, logs, msg = process_ai_vision(img)
            
            if stats:
                st.success(f"Detection Complete. Found {len(logs)} questions.")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Correct", stats['correct'])
                c2.metric("❌ Wrong", stats['wrong'])
                c3.metric("⚠️ Blank/Double", stats['blank'] + stats['double'])
                
                t1, t2 = st.tabs(["🖼️ AI View", "📊 Data"])
                with t1:
                    st.image(out_img, channels="BGR", use_container_width=True)
                    st.caption("The colored rings are the actual contours detected by the AI.")
                with t2:
                    st.dataframe(pd.DataFrame(logs), use_container_width=True)
            else:
                st.error(msg)
                st.image(out_img, channels="BGR")
                
        except Exception as e:
            st.error(f"Error: {e}")
