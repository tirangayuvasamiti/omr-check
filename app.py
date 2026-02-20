# Streamlit OMR Auto-Grader for YUVA GYAN MAHOTSAV 2026
# Single-file app.py — Upload a scanned OMR image to auto-grade

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="OMR Auto Grader", layout="wide")

st.title("OMR Auto Grader – YGM 2026")

# ---------------- ANSWER KEY (EDIT IF NEEDED) ----------------
# 60 questions, options: A,B,C,D
ANSWER_KEY = [
    'A','C','D','A','C','A','C','B','D','D','C','A','D','D','A','D','A','A','C','B',
    'D','B','D','C','B','A','D','A','B','A','C','B','C','D','A','C','D','B','A','D',
    'A','C','D','A','B','A','A','C','D','D','D','A','D','A','D','C','B','C','D','C'
]

POS_MARK = 3
NEG_MARK = -1

# ------------- IMAGE HELPERS -------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_paper(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(image, approx.reshape(4,2))
    return image


def threshold_ultra(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,15,5)
    return thr

# --------- BUBBLE GRID MAPPING ---------
# 3 columns × 20 rows × 4 options

def get_answer_region(warped):
    h, w = warped.shape[:2]
    top = int(h*0.22)
    bottom = int(h*0.84)
    left = int(w*0.05)
    right = int(w*0.95)
    return warped[top:bottom, left:right]


def split_columns(region):
    h, w = region.shape[:2]
    col_w = w//3
    return [region[:, i*col_w:(i+1)*col_w] for i in range(3)]


def detect_bubbles(col_img, thr_col):
    h, w = col_img.shape[:2]
    rows = 20
    opts = 4
    row_h = h//rows
    opt_w = w//opts

    responses = []
    for r in range(rows):
        row_vals = []
        for o in range(opts):
            y1 = r*row_h
            y2 = (r+1)*row_h
            x1 = o*opt_w
            x2 = (o+1)*opt_w

            cell = thr_col[y1:y2, x1:x2]
            total = cv2.countNonZero(cell)
            area = cell.shape[0]*cell.shape[1]
            fill_ratio = total/area
            row_vals.append(fill_ratio)
        responses.append(row_vals)
    return responses


def interpret(row_vals, fill_thr=0.22, multi_margin=0.06):
    filled = [i for i,v in enumerate(row_vals) if v>fill_thr]
    if len(filled)==0:
        return None, "UNATTEMPTED"
    if len(filled)>1:
        mx = max(row_vals)
        top = [i for i,v in enumerate(row_vals) if mx-v<multi_margin]
        if len(top)>1:
            return None, "MULTIPLE"
        return top[0], "OK"
    return filled[0], "OK"

# ------------- MAIN -------------
upload = st.file_uploader("Upload OMR Image", type=["jpg","png","jpeg"])

if upload:
    img = Image.open(upload).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    warped = detect_paper(img_cv)
    thr = threshold_ultra(warped)

    ans_region = get_answer_region(warped)
    thr_region = get_answer_region(thr)

    cols = split_columns(ans_region)
    thr_cols = split_columns(thr_region)

    answers = []
    status = []

    for i in range(3):
        res = detect_bubbles(cols[i], thr_cols[i])
        for row in res:
            idx, stt = interpret(row)
            answers.append(idx)
            status.append(stt)

    option_map = ['A','B','C','D']

    correct=0
    incorrect=0
    unattempted=0
    multiple=0

    for i,a in enumerate(answers[:60]):
        if status[i]=="UNATTEMPTED":
            unattempted+=1
        elif status[i]=="MULTIPLE":
            incorrect+=1
            multiple+=1
        else:
            if option_map[a]==ANSWER_KEY[i]:
                correct+=1
            else:
                incorrect+=1

    pos = correct*POS_MARK
    neg = incorrect*(-NEG_MARK)
    total = pos - neg

    c1,c2,c3 = st.columns(3)
    c1.metric("Correct", correct)
    c2.metric("Incorrect", incorrect)
    c3.metric("Unattempted", unattempted)

    c4,c5,c6 = st.columns(3)
    c4.metric("Positive Score", pos)
    c5.metric("Negative Score", neg)
    c6.metric("Total Score", total)

    st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Detected OMR", use_column_width=True)

st.markdown("---")
st.write("Scoring: +3 Correct, -1 Wrong, 0 Unattempted")
