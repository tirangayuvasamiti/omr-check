import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# =========================================================
# YUVA GYAN MAHOTSAV — OMR GRADING TOOL (Ultra Pro UI)
# =========================================================

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Yuva Gyan Mahotsav • OMR Grading Tool",
    layout="wide",
)

# -----------------------------
# ANSWER KEY (1..60) values: 1..4 (A..D)
# -----------------------------
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}
OPTS = {0: "A", 1: "B", 2: "C", 3: "D"}

# =========================================================
# IMAGE HELPERS
# =========================================================
def pil_to_bgr(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def order_points(pts):
    pts = pts.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def warp_perspective(image, quad, out_w=1400, out_h=1900):
    rect = order_points(quad)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))

def find_page_quad(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, k, iterations=2)
    edges = cv2.erode(edges, k, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    H, W = gray.shape[:2]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.30 * (H * W):
            return approx.reshape(4, 2)

    return None

def adaptive_bin_inv(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 8
    )
    return th

# =========================================================
# BUBBLE DETECTION / GRID LOGIC (Tuned for your sheet)
# =========================================================
def bubble_candidates(bin_img, min_area, max_area):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue

        ar = w / float(h)
        if ar < 0.75 or ar > 1.30:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue

        circularity = (4.0 * np.pi * area) / (peri * peri)
        if circularity < 0.55:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        out.append({"c": c, "cx": float(cx), "cy": float(cy), "area": float(area), "w": w, "h": h})
    return out

def cluster_1d(vals, gap):
    vals = sorted(vals)
    groups, cur = [], []
    for v in vals:
        if not cur:
            cur = [v]
        elif abs(v - cur[-1]) <= gap:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    if cur:
        groups.append(cur)
    centers = [float(np.mean(g)) for g in groups]
    return centers

def nearest_idx(v, centers):
    return int(np.argmin([abs(v - c) for c in centers]))

def fill_ratio(bin_img, contour, shrink=0.72):
    mask = np.zeros(bin_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w / 2.0, y + h / 2.0
    rr = int(min(w, h) * 0.5 * shrink)

    inner = np.zeros_like(mask)
    cv2.circle(inner, (int(cx), int(cy)), rr, 255, -1)
    inner = cv2.bitwise_and(inner, mask)

    inside = cv2.bitwise_and(bin_img, bin_img, mask=inner)
    filled = cv2.countNonZero(inside)
    area = cv2.countNonZero(inner)
    if area <= 0:
        return 0.0
    return filled / float(area)

# -----------------------------
# SHEET ROI + 3-COLUMN SPLIT
# Based on original sheet layout analysis:
# Answer zone approx: top~0.23H bottom~0.80H left~0.05W right~0.96W
# -----------------------------
def answer_roi(warped):
    H, W = warped.shape[:2]
    x1 = int(0.05 * W)
    x2 = int(0.96 * W)
    y1 = int(0.23 * H)
    y2 = int(0.80 * H)
    return warped[y1:y2, x1:x2].copy()

def split_three_columns(roi):
    H, W = roi.shape[:2]
    col1 = roi[:, int(0.00 * W):int(0.34 * W)].copy()
    col2 = roi[:, int(0.36 * W):int(0.66 * W)].copy()
    col3 = roi[:, int(0.68 * W):int(1.00 * W)].copy()
    return [col1, col2, col3]

def grade_column(col_bgr, q_start, overlay_offset_x, overlay_offset_y, overlay, params):
    min_area = params["min_area"]
    max_area = params["max_area"]
    y_gap_factor = params["y_gap_factor"]
    x_gap_factor = params["x_gap_factor"]
    blank_thresh = params["blank_thresh"]
    mark_thresh = params["mark_thresh"]
    double_rel = params["double_rel"]

    gray = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2GRAY)
    bin_img = adaptive_bin_inv(gray)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)

    cands = bubble_candidates(bin_img, min_area=min_area, max_area=max_area)
    if len(cands) < 40:
        return [], {"error": f"Too few bubbles detected in column Q{q_start}-{q_start+19}. Found={len(cands)}"}

    scale = float(np.median([(d["w"] + d["h"]) / 2.0 for d in cands]))
    y_gap = max(10.0, scale * y_gap_factor)
    x_gap = max(10.0, scale * x_gap_factor)

    y_centers = sorted(cluster_1d([d["cy"] for d in cands], gap=y_gap))
    x_centers = sorted(cluster_1d([d["cx"] for d in cands], gap=x_gap))

    # reduce x centers to 4 strongest if more
    if len(x_centers) > 4:
        counts = np.zeros(len(x_centers), dtype=int)
        for d in cands:
            counts[nearest_idx(d["cx"], x_centers)] += 1
        top = np.argsort(counts)[::-1][:4]
        x_centers = sorted([x_centers[i] for i in top])

    logs = []
    q = q_start

    rows = [[] for _ in range(len(y_centers))]
    for d in cands:
        ri = nearest_idx(d["cy"], y_centers)
        rows[ri].append(d)

    # sort rows top-to-bottom
    row_order = np.argsort(y_centers)
    rows = [rows[i] for i in row_order]

    for r in rows:
        if q > q_start + 19:
            break

        per_opt = [[] for _ in range(len(x_centers))]
        for d in r:
            ci = nearest_idx(d["cx"], x_centers)
            per_opt[ci].append(d)

        chosen = []
        for ci, bucket in enumerate(per_opt):
            if not bucket:
                chosen.append(None)
            else:
                bucket.sort(key=lambda dd: abs(dd["cx"] - x_centers[ci]))
                chosen.append(bucket[0])

        if len(chosen) != 4 or any(x is None for x in chosen):
            logs.append({"Q": q, "Status": "Missing bubbles", "Picked": None, "Answer": OPTS[ANS_KEY[q]-1],
                         "BestFill": None, "SecondFill": None, "Confidence": 0.0})
            q += 1
            continue

        ratios = [(fill_ratio(bin_img, chosen[j]["c"], shrink=0.72), j) for j in range(4)]
        ratios.sort(key=lambda x: x[0], reverse=True)

        best, best_j = ratios[0]
        second, second_j = ratios[1]
        k_ans = ANS_KEY[q] - 1

        conf = float(np.clip((best - second) / max(best, 1e-6), 0, 1))

        def draw_global(contour, color, thick=3):
            shifted = contour.copy()
            shifted[:, 0, 0] += overlay_offset_x
            shifted[:, 0, 1] += overlay_offset_y
            cv2.drawContours(overlay, [shifted], -1, color, thick)

        if best < blank_thresh:
            status = "Blank"
            picked = None
            draw_global(chosen[k_ans]["c"], (255, 0, 0), 2)  # correct in blue
        else:
            if (best >= mark_thresh) and (second >= mark_thresh) and (second >= best * double_rel):
                status = "Double"
                picked = None
                draw_global(chosen[best_j]["c"], (0, 255, 255), 3)
                draw_global(chosen[second_j]["c"], (0, 255, 255), 3)
                draw_global(chosen[k_ans]["c"], (255, 0, 0), 2)
            else:
                picked = best_j
                if picked == k_ans:
                    status = "Correct"
                    draw_global(chosen[picked]["c"], (0, 255, 0), 3)
                else:
                    status = "Incorrect"
                    draw_global(chosen[picked]["c"], (0, 0, 255), 3)
                    draw_global(chosen[k_ans]["c"], (255, 0, 0), 2)

        logs.append({
            "Q": q,
            "Status": status,
            "Picked": OPTS[picked] if picked is not None else None,
            "Answer": OPTS[k_ans],
            "BestFill": round(best, 3),
            "SecondFill": round(second, 3),
            "Confidence": round(conf, 3),
        })
        q += 1

    return logs, {"rows": len(y_centers), "x_centers": x_centers, "bubble_scale": round(scale, 2), "bubbles_found": len(cands)}

def grade_sheet(image_bgr, params):
    quad = find_page_quad(image_bgr)
    if quad is None:
        warped = cv2.resize(image_bgr, (1400, 1900))
        doc_mode = "fallback_resize"
    else:
        warped = warp_perspective(image_bgr, quad, 1400, 1900)
        doc_mode = "warp_from_quad"

    roi = answer_roi(warped)
    overlay = roi.copy()

    cols = split_three_columns(roi)
    H, W = roi.shape[:2]
    offsets = [(0, 0), (int(0.36 * W), 0), (int(0.68 * W), 0)]
    starts = [1, 21, 41]

    all_logs = []
    debug = {"doc_mode": doc_mode}

    for i in range(3):
        logs, info = grade_column(
            cols[i],
            q_start=starts[i],
            overlay_offset_x=offsets[i][0],
            overlay_offset_y=offsets[i][1],
            overlay=overlay,
            params=params,
        )
        debug[f"col_{i+1}"] = info
        all_logs.extend(logs)

    df = pd.DataFrame(all_logs).sort_values("Q")

    stats = {"correct": 0, "wrong": 0, "blank": 0, "double": 0, "missing": 0}
    for s in df["Status"].tolist():
        if s == "Correct":
            stats["correct"] += 1
        elif s == "Incorrect":
            stats["wrong"] += 1
        elif s == "Blank":
            stats["blank"] += 1
        elif s == "Double":
            stats["double"] += 1
        elif s == "Missing bubbles":
            stats["missing"] += 1

    return stats, overlay, roi, debug, df

# =========================================================
# ULTRA PRO UI
# =========================================================

# ---- top bar (minimal, pro) ----
st.markdown(
    """
    <div style="padding: 14px 16px; border-radius: 14px; background: linear-gradient(90deg, #0B1220, #111C33); border:1px solid rgba(255,255,255,0.08);">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="font-size:18px; font-weight:700; color:#EAF2FF; line-height:1.2;">
            Yuva Gyan Mahotsav • OMR Grading Tool
          </div>
          <div style="font-size:12px; color:rgba(234,242,255,0.72); margin-top:4px;">
            Fast • Accurate • Audit-friendly (Blank/Double/Missing detection)
          </div>
        </div>
        <div style="font-size:12px; color:rgba(234,242,255,0.72); text-align:right;">
          Layout tuned to official sheet<br/>
          <a href="https://www.genspark.ai/api/files/s/6eWzcOhD" target="_blank" style="color:#9AD0FF; text-decoration:none;">View reference OMR</a>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---- main action row ----
left, right = st.columns([1.35, 1])

with left:
    uploaded = st.file_uploader("Upload filled OMR image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with right:
    st.markdown(
        """
        <div style="padding:12px 14px; border-radius:14px; border:1px solid rgba(0,0,0,0.08); background:#F7F9FC;">
            <div style="font-weight:700; font-size:13px;">Recommended upload</div>
            <div style="font-size:12px; color:#334155; margin-top:6px;">
                • Flat scan / top-down photo<br/>
                • Good lighting (avoid shadows)<br/>
                • Full sheet visible (borders included)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- advanced (hidden) ----
with st.expander("Advanced (Calibration)", expanded=False):
    st.caption("Use only if detection is weak on real photos. Defaults are tuned for your sheet.")
    params = {
        "min_area": st.slider("Min bubble area", 50, 2000, 220, 10),
        "max_area": st.slider("Max bubble area", 1000, 20000, 9000, 100),
        "y_gap_factor": st.slider("Row clustering gap factor", 0.8, 3.0, 1.45, 0.05),
        "x_gap_factor": st.slider("Option clustering gap factor", 0.8, 3.0, 1.45, 0.05),
        "blank_thresh": st.slider("Blank threshold (fill ratio)", 0.02, 0.30, 0.10, 0.01),
        "mark_thresh": st.slider("Marked threshold (fill ratio)", 0.05, 0.50, 0.18, 0.01),
        "double_rel": st.slider("Double similarity (2nd/best)", 0.60, 0.98, 0.85, 0.01),
    }
if "params" not in locals():
    # clean default (no clutter)
    params = {
        "min_area": 220,
        "max_area": 9000,
        "y_gap_factor": 1.45,
        "x_gap_factor": 1.45,
        "blank_thresh": 0.10,
        "mark_thresh": 0.18,
        "double_rel": 0.85,
    }

# ---- run grading ----
if uploaded:
    img = pil_to_bgr(uploaded)

    with st.spinner("Grading OMR…"):
        stats, overlay, roi, debug, df = grade_sheet(img, params)

    # KPI strip (compact pro)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Correct", stats["correct"])
    k2.metric("Incorrect", stats["wrong"])
    k3.metric("Blank", stats["blank"])
    k4.metric("Double", stats["double"])
    k5.metric("Missing", stats["missing"])

    # Tabs (only what’s needed)
    tab1, tab2 = st.tabs(["Checked Sheet", "Results"])
    with tab1:
        st.image(overlay, channels="BGR", use_container_width=True)
        st.caption("Legend: Green=Correct • Red=Wrong • Blue=Correct Answer • Yellow=Double")

        # optional debug (collapsed)
        with st.expander("Diagnostics", expanded=False):
            st.json(debug)

        # Download annotated image
        # Convert BGR -> RGB and encode as PNG
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", overlay_rgb)
        if ok:
            st.download_button(
                "Download Checked Sheet (PNG)",
                data=buf.tobytes(),
                file_name="checked_sheet.png",
                mime="image/png",
            )

    with tab2:
        st.dataframe(df, use_container_width=True, height=520)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results (CSV)", data=csv, file_name="omr_results.csv", mime="text/csv")

else:
    st.info("Upload a filled OMR image to start grading.")
