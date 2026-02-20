import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("YUVA GYAN MAHOTSAV 2026 - OMR AUTO GRADER")

# ================= ANSWER KEY =================
ANSWER_KEY = [
'A','C','D','A','C','A','C','B','D','D','C','A','D','D','A','D','A','A','C','B',
'D','B','D','C','B','A','D','A','B','A','C','B','C','D','A','C','D','B','A','D',
'A','C','D','A','B','A','A','C','D','D','D','A','D','A','D','C','B','C','D','C'
]

POS_MARK = 3
NEG_MARK = -1
OPTIONS = ['A','B','C','D']

# ================ PERSPECTIVE =================
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA,widthB))
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA,heightB))
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect,dst)
    return cv2.warpPerspective(image,M,(maxWidth,maxHeight))

# ============== ANCHOR DETECTION ==============
def detect_anchors_and_warp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 80,255,cv2.THRESH_BINARY_INV)[1]
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anchors = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 500 < area < 5000:
            x,y,w,h = cv2.boundingRect(c)
            aspect = w/float(h)
            if 0.8 < aspect < 1.2:
                anchors.append((x,y,w,h))

    if len(anchors) < 4:
        return image

    anchors = sorted(anchors, key=lambda b:(b[1],b[0]))

    tl = anchors[0]
    tr = anchors[1]
    bl = anchors[-2]
    br = anchors[-1]

    pts = np.array([
        [tl[0], tl[1]],
        [tr[0]+tr[2], tr[1]],
        [br[0]+br[2], br[1]+br[3]],
        [bl[0], bl[1]+bl[3]]
    ], dtype="float32")

    return four_point_transform(image, pts)

# =============== MCQ REGION ===================
def get_answer_region(warped):
    h, w = warped.shape[:2]
    top    = int(h*0.26)
    bottom = int(h*0.83)
    left   = int(w*0.06)
    right  = int(w*0.94)
    return warped[top:bottom, left:right]

# =============== THRESHOLD ====================
def preprocess(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5)
    
    # Apply morphological closing to solidify grainy pencil marks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    
    return thr

# ============ GRID SPLIT ======================
def split_columns(region):
    h,w = region.shape[:2]
    col_w = w//3
    return [region[:,i*col_w:(i+1)*col_w] for i in range(3)]

def detect_bubbles(col):
    h, w = col.shape
    rows = 20
    opts = 4
    row_h = h // rows
    opt_w = w // opts

    answers = []
    status = []

    for r in range(rows):
        vals = []
        for o in range(opts):
            y1 = r * row_h
            y2 = (r + 1) * row_h
            x1 = o * opt_w
            x2 = (o + 1) * opt_w

            # ADD MARGINS: Focus on the center of the cell
            margin_y = int(row_h * 0.15)
            margin_x = int(opt_w * 0.15)

            cell = col[y1+margin_y : y2-margin_y, x1+margin_x : x2-margin_x]
            
            # Safety check for extremely small crops
            if cell.size == 0:
                vals.append(0)
                continue

            total = cv2.countNonZero(cell)
            area = cell.shape[0] * cell.shape[1]
            ratio = total / float(area)

            vals.append(ratio)

        # RELATIVE THRESHOLDING
        MIN_ATTEMPT_THRESH = 0.20 
        filled = [i for i, v in enumerate(vals) if v > MIN_ATTEMPT_THRESH]

        if len(filled) == 0:
            answers.append(None)
            status.append("UNATTEMPTED")
        elif len(filled) == 1:
            answers.append(filled[0])
            status.append("OK")
        else:
            # Check for poor erasures by comparing densities
            sorted_indices = np.argsort(vals)[::-1]
            top_val = vals[sorted_indices[0]]
            second_val = vals[sorted_indices[1]]

            # If the second highest is >60% as dense as the highest, it's a true double mark
            if second_val > (top_val * 0.60):
                answers.append(None)
                status.append("MULTIPLE")
            else:
                answers.append(sorted_indices[0])
                status.append("OK")

    return answers, status

# ================= STREAMLIT ==================
upload = st.file_uploader("Upload OMR Image", type=['jpg','jpeg','png'])

if upload:
    img = Image.open(upload).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    warped = detect_anchors_and_warp(img_cv)

    region = get_answer_region(warped)
    thr = preprocess(region)

    cols = split_columns(thr)

    all_ans = []
    all_stat = []

    for c in cols:
        a, s = detect_bubbles(c)
        all_ans.extend(a)
        all_stat.extend(s)

    correct = 0
    incorrect = 0
    unattempted = 0

    for i in range(60):
        if all_stat[i] == "UNATTEMPTED":
            unattempted += 1
        elif all_stat[i] == "MULTIPLE":
            incorrect += 1
        else:
            if OPTIONS[all_ans[i]] == ANSWER_KEY[i]:
                correct += 1
            else:
                incorrect += 1

    pos = correct * POS_MARK
    neg = incorrect * abs(NEG_MARK)
    total = pos - neg

    st.subheader("RESULT")
    c1, c2, c3 = st.columns(3)
    c1.metric("Correct", correct)
    c2.metric("Incorrect", incorrect)
    c3.metric("Unattempted", unattempted)

    c4, c5, c6 = st.columns(3)
    c4.metric("Positive Score", pos)
    c5.metric("Negative Score", neg)
    c6.metric("Total Score", total)

    st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB),
             caption="Aligned OMR", use_container_width=True)
