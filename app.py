import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io, json
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Yuva Gyan Mahotsav – OMR Grader",
    page_icon="📝",
    layout="wide",
)

TOTAL_QUESTIONS  = 60
OPTIONS_PER_Q    = 4       # A B C D
QUESTIONS_PER_COL = 20     # 3 columns × 20 = 60

OPT_LABELS = ["A", "B", "C", "D"]
OPT_REV    = {v: i for i, v in enumerate(OPT_LABELS)}

# ──────────────────────────────────────────────
# IMAGE UTILITIES
# ──────────────────────────────────────────────

def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def order_pts(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(img, pts):
    rect = order_pts(pts)
    tl, tr, br, bl = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h))

def deskew_sheet(bgr):
    """Try to find and warp the sheet. Returns gray of warped (or original)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 100)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (bgr.shape[0]*bgr.shape[1]*0.2):
            warped = four_point_transform(bgr, approx)
            return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray   # fallback – no sheet boundary found

def standardize(gray, target_w=900):
    """Resize to standard width for consistent processing."""
    h, w = gray.shape
    scale = target_w / w
    return cv2.resize(gray, (target_w, int(h * scale)))

# ──────────────────────────────────────────────
# BUBBLE DETECTION
# ──────────────────────────────────────────────

def find_bubbles(gray):
    """Return list of (x,y,r) for all detected bubbles."""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Hough circles for robust bubble detection
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=12,
        param1=60,
        param2=25,
        minRadius=6,
        maxRadius=20,
    )
    if circles is not None:
        return np.round(circles[0, :]).astype(int).tolist()
    return []

def find_bubbles_contour(gray):
    """Fallback: contour-based bubble detection."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h) if h > 0 else 0
        area = cv2.contourArea(c)
        if 0.65 <= ar <= 1.45 and 50 <= area <= 2000 and w >= 8:
            cx, cy = x + w//2, y + h//2
            r = (w + h) // 4
            bubbles.append([cx, cy, r])
    return bubbles

def is_filled(gray, cx, cy, r, threshold=0.45):
    """Return True if ≥ threshold fraction of the circle area is dark."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (cx, cy), max(r-2, 3), 255, -1)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    _, dark = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(cv2.bitwise_and(dark, dark, mask=mask))
    total_pixels = cv2.countNonZero(mask)
    return (dark_pixels / total_pixels) >= threshold if total_pixels > 0 else False

# ──────────────────────────────────────────────
# GROUP BUBBLES → QUESTIONS
# ──────────────────────────────────────────────

def group_into_rows(bubbles, y_tol=14):
    """Cluster bubbles by y-coordinate into rows."""
    sorted_b = sorted(bubbles, key=lambda b: b[1])
    rows = []
    cur = [sorted_b[0]]
    for b in sorted_b[1:]:
        if abs(b[1] - cur[-1][1]) <= y_tol:
            cur.append(b)
        else:
            rows.append(sorted(cur, key=lambda b: b[0]))
            cur = [b]
    rows.append(sorted(cur, key=lambda b: b[0]))
    return rows

def extract_answers(gray, show_debug=False):
    """
    Main pipeline. Returns:
        answers    – list[int] length 60  (0=A,1=B,2=C,3=D,-1=blank,-2=multi)
        debug_img  – BGR image with overlays
        n_found    – how many bubbles were detected
    """
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1. Detect bubbles
    bubbles = find_bubbles(gray)
    if len(bubbles) < TOTAL_QUESTIONS * OPTIONS_PER_Q * 0.5:
        bubbles = find_bubbles_contour(gray)

    n_found = len(bubbles)

    if n_found < TOTAL_QUESTIONS:
        # Not enough bubbles – return blanks
        return [-1] * TOTAL_QUESTIONS, debug_img, n_found

    # 2. Group into rows
    rows = group_into_rows(bubbles)

    # 3. Keep only rows that have exactly OPTIONS_PER_Q bubbles
    q_rows = [r for r in rows if len(r) == OPTIONS_PER_Q]

    # 4. Determine fill status
    answers = []
    for row in q_rows[:TOTAL_QUESTIONS]:
        filled = []
        for (cx, cy, r) in row:
            filled_flag = is_filled(gray, cx, cy, r)
            color = (0, 0, 220) if filled_flag else (0, 180, 0)
            if show_debug:
                cv2.circle(debug_img, (cx, cy), r, color, 2)
                cv2.circle(debug_img, (cx, cy), 2, color, -1)
            if filled_flag:
                filled.append(row.index((cx, cy, r)))

        if len(filled) == 1:
            answers.append(filled[0])
        elif len(filled) == 0:
            answers.append(-1)
        else:
            answers.append(-2)

    # Pad
    while len(answers) < TOTAL_QUESTIONS:
        answers.append(-1)

    return answers[:TOTAL_QUESTIONS], debug_img, n_found

# ──────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────

def score(answers, key, m_correct=1.0, m_wrong=-0.25):
    correct = wrong = unattempted = multi = 0
    rows = []
    for i, (a, k) in enumerate(zip(answers, key)):
        a_lbl = OPT_LABELS[a] if 0 <= a <= 3 else ("—" if a == -1 else "Multi")
        k_lbl = OPT_LABELS[k]
        if a == -1:
            unattempted += 1; s = 0; status = "Unattempted"
        elif a == -2:
            multi += 1; wrong += 1; s = m_wrong; status = "Multi-Mark ⚠️"
        elif a == k:
            correct += 1; s = m_correct; status = "Correct ✅"
        else:
            wrong += 1; s = m_wrong; status = "Wrong ❌"
        rows.append({"Q#": i+1, "Marked": a_lbl, "Correct": k_lbl,
                     "Status": status, "Score": s})
    total = round(correct * m_correct + wrong * m_wrong, 2)
    return dict(correct=correct, wrong=wrong, unattempted=unattempted,
                multi=multi, total=total, details=rows)

def grade_label(pct):
    return ("A+" if pct>=90 else "A" if pct>=80 else "B" if pct>=70
            else "C" if pct>=60 else "D" if pct>=50 else "F")

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Answer Key ──
    st.markdown("### 📋 Answer Key (60 Qs)")
    key_method = st.radio("Input method", ["Manual", "Upload CSV", "Upload JSON"],
                          horizontal=True)

    answer_key = []

    if key_method == "Manual":
        for i in range(0, TOTAL_QUESTIONS, 10):
            cols = st.columns(10)
            for j, col in enumerate(cols):
                qi = i + j
                if qi < TOTAL_QUESTIONS:
                    v = col.selectbox(f"Q{qi+1}", OPT_LABELS,
                                      key=f"k{qi}", label_visibility="collapsed")
                    answer_key.append(OPT_REV[v])

    elif key_method == "Upload CSV":
        f = st.file_uploader("CSV: one column of answers", type="csv")
        if f:
            try:
                df = pd.read_csv(f)
                col = df.columns[-1]
                answer_key = [OPT_REV[str(v).strip().upper()] for v in df[col][:TOTAL_QUESTIONS]]
                st.success(f"Key loaded – {len(answer_key)} questions")
            except Exception as e:
                st.error(f"Parse error: {e}")
        if not answer_key:
            answer_key = [0] * TOTAL_QUESTIONS

    else:  # JSON
        f = st.file_uploader("JSON: list or dict", type="json")
        if f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    answer_key = [OPT_REV[x.upper()] for x in data[:TOTAL_QUESTIONS]]
                else:
                    answer_key = [OPT_REV[data[str(i+1)].upper()] for i in range(TOTAL_QUESTIONS)]
                st.success(f"Key loaded – {len(answer_key)} questions")
            except Exception as e:
                st.error(f"Parse error: {e}")
        if not answer_key:
            answer_key = [0] * TOTAL_QUESTIONS

    if not answer_key:
        answer_key = [0] * TOTAL_QUESTIONS
    while len(answer_key) < TOTAL_QUESTIONS:
        answer_key.append(0)

    # ── Marking Scheme ──
    st.markdown("### 🎯 Marking Scheme")
    m_correct = st.number_input("Correct  (+)", value=1.0, step=0.25, min_value=0.0)
    m_wrong   = st.number_input("Wrong    (−)", value=-0.25, step=0.25, max_value=0.0)

    # ── Debug ──
    st.markdown("### 🔬 Debug")
    show_debug = st.toggle("Show bubble overlay", value=False)
    sensitivity = st.slider("Fill sensitivity (lower = stricter)", 0.2, 0.7, 0.45, 0.05,
                            help="Fraction of circle that must be dark to count as filled")

# ──────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────

st.title("📝 Yuva Gyan Mahotsav 2026 – OMR Auto Grader")
st.caption("Tiranga Yuva Samiti · Upload scanned OMR sheets to auto-detect and grade answers.")

tab1, tab2, tab3 = st.tabs(["📤 Upload & Grade", "📊 Batch Results", "ℹ️ Help"])

# ─── TAB 1 ───────────────────────────────────
with tab1:
    files = st.file_uploader(
        "Upload OMR sheet images (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("👆 Upload one or more scanned OMR sheet images to begin.")
        st.image(
            "https://via.placeholder.com/700x350/FFF5E6/FF9933?text=Upload+your+OMR+sheets+above",
            use_column_width=True,
        )
    else:
        batch = []

        for idx, uf in enumerate(files):
            st.divider()
            st.subheader(f"Sheet {idx+1} · `{uf.name}`")

            try:
                pil_img = Image.open(uf)
            except Exception as e:
                st.error(f"Cannot open image: {e}")
                continue

            with st.spinner("Detecting bubbles…"):
                bgr = pil_to_cv(pil_img)
                gray = deskew_sheet(bgr)
                gray = standardize(gray)
                answers, dbg_img, n_found = extract_answers(gray, show_debug)
                result = score(answers, answer_key, m_correct, m_wrong)

            # ── Layout ──
            col_l, col_r = st.columns([1, 1])

            with col_l:
                if show_debug:
                    st.image(cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB),
                             caption=f"Bubble overlay · {n_found} bubbles found",
                             use_column_width=True)
                else:
                    st.image(pil_img, caption="Uploaded sheet", use_column_width=True)
                st.caption(f"ℹ️ {n_found} bubbles detected "
                           f"(need {TOTAL_QUESTIONS * OPTIONS_PER_Q} for 60 Qs × 4 opts)")
                if n_found < TOTAL_QUESTIONS * OPTIONS_PER_Q * 0.7:
                    st.warning("⚠️ Low bubble count – try a clearer / higher-res scan, "
                               "or adjust sensitivity in sidebar.")

            with col_r:
                pct = (result["correct"] / TOTAL_QUESTIONS) * 100
                g = grade_label(pct)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("✅ Correct",     result["correct"])
                m2.metric("❌ Wrong",       result["wrong"])
                m3.metric("⬜ Unattempted", result["unattempted"])
                m4.metric("🏆 Score",       result["total"])

                st.progress(pct / 100)
                grade_color = {"A+":"🟢","A":"🟢","B":"🔵","C":"🟡","D":"🟠","F":"🔴"}
                st.markdown(
                    f"**Grade: {grade_color.get(g,'⚪')} {g}** &nbsp; ({pct:.1f}%)"
                )

                if result["multi"]:
                    st.warning(f"⚠️ {result['multi']} question(s) have multiple bubbles filled.")

            with st.expander("📋 Question-wise breakdown"):
                df_d = pd.DataFrame(result["details"])

                def highlight(row):
                    if "Correct" in row["Status"]:   return ["background-color:#d4edda"]*len(row)
                    if "Wrong"   in row["Status"]:   return ["background-color:#f8d7da"]*len(row)
                    if "Multi"   in row["Status"]:   return ["background-color:#fff3cd"]*len(row)
                    return [""]*len(row)

                st.dataframe(
                    df_d.style.apply(highlight, axis=1),
                    use_container_width=True, height=280,
                )

            uf.seek(0)
            csv_bytes = df_d.to_csv(index=False).encode()
            st.download_button(
                f"⬇️ Download result – {uf.name}",
                data=csv_bytes,
                file_name=f"result_{uf.name.rsplit('.',1)[0]}.csv",
                mime="text/csv",
                key=f"dl_{idx}",
            )

            batch.append({
                "File": uf.name,
                "Bubbles Found": n_found,
                "Correct": result["correct"],
                "Wrong": result["wrong"],
                "Unattempted": result["unattempted"],
                "Multi-Mark": result["multi"],
                "Total Score": result["total"],
                "Percentage": round(pct, 2),
                "Grade": g,
                "Answers": ",".join([
                    OPT_LABELS[a] if 0<=a<=3 else ("—" if a==-1 else "M")
                    for a in answers
                ]),
            })

        if batch:
            st.session_state["batch"] = batch

# ─── TAB 2 ───────────────────────────────────
with tab2:
    br = st.session_state.get("batch", [])
    if not br:
        st.info("Grade sheets in Tab 1 first.")
    else:
        df_b = pd.DataFrame(br)
        st.subheader(f"📊 Batch Summary – {len(df_b)} sheet(s)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Score",  f"{df_b['Total Score'].mean():.2f}")
        c2.metric("Highest Score",  f"{df_b['Total Score'].max():.2f}")
        c3.metric("Lowest Score",   f"{df_b['Total Score'].min():.2f}")
        c4.metric("Average %",      f"{df_b['Percentage'].mean():.1f}%")

        st.dataframe(df_b.drop(columns=["Answers"]), use_container_width=True)

        st.subheader("Grade Distribution")
        gc = df_b["Grade"].value_counts().reindex(
            ["A+","A","B","C","D","F"], fill_value=0
        )
        st.bar_chart(gc)

        col_a, col_b = st.columns(2)
        col_a.download_button(
            "⬇️ Download CSV",
            data=df_b.to_csv(index=False).encode(),
            file_name=f"omr_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        col_b.download_button(
            "⬇️ Download JSON",
            data=json.dumps(br, indent=2).encode(),
            file_name=f"omr_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ─── TAB 3 ───────────────────────────────────
with tab3:
    st.markdown("""
## How to Use

### 1 · Set the Answer Key
Choose **Manual**, **Upload CSV**, or **Upload JSON** in the sidebar.

**CSV format:**
```
Question,Answer
1,A
2,C
3,B
```

**JSON format (dict or list):**
```json
{"1":"A","2":"C","3":"B"}
```
or `["A","C","B","D",...]`

---

### 2 · Adjust Marking Scheme
Default: **+1** correct, **−0.25** wrong. Change in sidebar.

---

### 3 · Upload OMR Sheets
- Go to **Upload & Grade** tab
- Upload JPG / PNG images (one or many)

---

### 4 · If Bubbles Aren't Detected Correctly
- Enable **Show bubble overlay** in sidebar to see what's being detected
- Adjust **Fill sensitivity** slider (lower = stricter; raise it if blanks are marked filled)
- Rescan at higher resolution / better lighting

---

## 📸 Scanning Tips

| ✅ Do | ❌ Avoid |
|---|---|
| 300 DPI scan or clear photo | Blurry phone shots |
| Even, bright lighting | Shadows / glare |
| Sheet flat & fully in frame | Edges cut off |
| Dark, fully filled bubbles | Partial / light fills |

---
*Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti*
    """)
