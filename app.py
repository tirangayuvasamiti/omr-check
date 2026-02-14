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
WRONG_PTS = 1 

# Normal Format: A=1, B=2, C=3, D=4
# Key extracted from provided Answer Key [cite: 20-300]
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}

def get_document_corners(image):
    """Detects the 4 physical corners of the page using edge contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
    # Default to image boundary if edge detection fails
    h, w = image.shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")

def process_omr_engine(image_np, show_missed=True):
    # 1. Image Alignment & Standardized Warp
    orig = imutils.resize(image_np, height=1500)
    corner_pts = get_document_corners(orig)
    warped_gray = four_point_transform(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), corner_pts)
    warped_color = four_point_transform(orig, corner_pts)
    warped_gray = cv2.resize(warped_gray, (1200, 1600))
    warped_color = cv2.resize(warped_color, (1200, 1600))
    
    # 2. Extract Question Region [cite: 328-581]
    ROI_TOP, ROI_BOTTOM = 350, 1580
    roi_gray = warped_gray[ROI_TOP:ROI_BOTTOM, :]
    thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 11)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Locate & Sort Bubbles
    bubbles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 15 <= w <= 55 and 15 <= h <= 55 and 0.6 <= (w/h) <= 1.4:
            bubbles.append((x, y + ROI_TOP, w, h))
    
    if len(bubbles) != 240:
        return None, warped_color, None, f"Found {len(bubbles)}/240 bubbles. Check image quality."

    bubbles = sorted(bubbles, key=lambda b: b[0])
    columns = [bubbles[0:80], bubbles[80:160], bubbles[160:240]]
    
    results = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
    breakdown = []
    q_num = 1
    
    # 4. Mathematical Intensity Engine
    for col in columns:
        col = sorted(col, key=lambda b: b[1])
        for i in range(0, 80, 4):
            row = sorted(col[i:i+4], key=lambda b: b[0])
            intensities = []
            for j, (bx, by, bw, bh) in enumerate(row):
                roi = warped_gray[by:by+bh, bx:bx+bw]
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.circle(mask, (bw//2, bh//2), int(min(bw, bh) * 0.35), 255, -1)
                intensities.append((cv2.mean(roi, mask=mask)[0], j, (bx, by, bw, bh)))

            intensities.sort(key=lambda x: x[0])
            darkest_val, second_val = intensities[0][0], intensities[1][0]
            lightest_val = intensities[-1][0]
            
            # MATH RULE: Must be at least 15% darker than background to be "filled" [cite: 323]
            fill_cutoff = lightest_val * 0.85
            marked = [idx for val, idx, box in intensities if val < fill_cutoff]
            
            ans_idx = ANS_KEY.get(q_num) - 1
            status, selected = "", "-"

            # --- MUTUALLY EXCLUSIVE GRADING ---
            if not marked:
                results["blank"] += 1
                status = "Blank"
                if show_missed: cv2.rectangle(warped_color, (row[ans_idx][0], row[ans_idx][1]), (row[ans_idx][0]+row[ans_idx][2], row[ans_idx][1]+row[ans_idx][3]), (255,0,0), 2)
            elif len(marked) > 1 and (second_val < darkest_val + 20):
                # Only flag as double if both are DARK 
                results["double"] += 1
                results["wrong"] += 1
                status = "Double"
                selected = "Multiple"
                for idx in marked: cv2.rectangle(warped_color, (row[idx][0], row[idx][1]), (row[idx][0]+row[idx][2], row[idx][1]+row[idx][3]), (0,255,255), 3)
            elif marked[0] == ans_idx:
                results["correct"] += 1
                status = "Correct"
                selected = OPTS[marked[0]]
                cv2.rectangle(warped_color, (row[marked[0]][0], row[marked[0]][1]), (row[marked[0]][0]+row[marked[0]][2], row[marked[0]][1]+row[marked[0]][3]), (0,220,0), 3)
            else:
                results["wrong"] += 1
                status = "Incorrect"
                selected = OPTS[marked[0]]
                cv2.rectangle(warped_color, (row[marked[0]][0], row[marked[0]][1]), (row[marked[0]][0]+row[marked[0]][2], row[marked[0]][1]+row[marked[0]][3]), (0,0,255), 3)
                if show_missed: cv2.rectangle(warped_color, (row[ans_idx][0], row[ans_idx][1]), (row[ans_idx][0]+row[ans_idx][2], row[ans_idx][1]+row[ans_idx][3]), (255,0,0), 2)

            breakdown.append({"Q": q_num, "Pick": selected, "Ans": OPTS[ans_idx], "Status": status})
            q_num += 1

    return results, warped_color, breakdown, "Success"

# --- STREAMLIT UI ---
st.title("🏆 Yuva Gyan Ultra-Accuracy Grader")
uploaded_file = st.file_uploader("Upload OMR Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    res, processed, table, msg = process_omr_engine(img_np)
    
    if res:
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.dataframe(pd.DataFrame(table), hide_index=True)
    else:
        st.error(msg)
