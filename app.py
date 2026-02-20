import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import json
import zipfile
from datetime import datetime
import imutils
from imutils import contours as cont_utils

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_QUESTIONS = 60
OPTIONS_PER_Q   = 4          # A B C D
COLS_ON_SHEET   = 3          # 3 columns × 20 rows = 60 questions
ROWS_PER_COL    = 20

st.set_page_config(
    page_title="Yuva Gyan Mahotsav – OMR Grader",
    page_icon="📝",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER – image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return gray, thresh


def find_document_contour(thresh):
    """Find the largest quadrilateral – the OMR sheet boundary."""
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    doc_cnt = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
    return doc_cnt


def four_point_transform(image, pts):
    """Perspective-correct the sheet to a flat rectangle."""
    rect = order_points(pts.reshape(4, 2).astype("float32"))
    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ─────────────────────────────────────────────────────────────────────────────
# CORE OMR DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_answers(warped_gray, debug=False):
    """
    Detect filled bubbles on the warped (perspective-corrected) OMR image.

    Returns:
        answers  – list of 60 ints (0=A,1=B,2=C,3=D, -1=unanswered, -2=multi)
        debug_img – annotated image for review
    """
    h, w = warped_gray.shape

    # Threshold
    _, thresh = cv2.threshold(warped_gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find all circular contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    bubble_cnts = []
    for c in cnts:
        (x, y, cw, ch) = cv2.boundingRect(c)
        ar = cw / float(ch)
        area = cv2.contourArea(c)
        if 0.7 <= ar <= 1.3 and 80 <= area <= 1800 and cw >= 8:
            bubble_cnts.append(c)

    if len(bubble_cnts) < TOTAL_QUESTIONS * OPTIONS_PER_Q:
        # Fallback: relax constraints
        bubble_cnts = []
        for c in cnts:
            (x, y, cw, ch) = cv2.boundingRect(c)
            ar = cw / float(ch)
            area = cv2.contourArea(c)
            if 0.6 <= ar <= 1.5 and 40 <= area <= 2500 and cw >= 6:
                bubble_cnts.append(c)

    # Sort top-to-bottom then left-to-right
    if not bubble_cnts:
        return [-1] * TOTAL_QUESTIONS, warped_gray

    bubble_cnts = sorted(bubble_cnts, key=lambda c: cv2.boundingRect(c)[1])

    # Group into rows (questions)
    rows = []
    current_row = [bubble_cnts[0]]
    for c in bubble_cnts[1:]:
        cy = cv2.boundingRect(c)[1]
        prev_cy = cv2.boundingRect(current_row[-1])[1]
        if abs(cy - prev_cy) < 12:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)

    # Each row should have OPTIONS_PER_Q bubbles
    valid_rows = [r for r in rows if len(r) == OPTIONS_PER_Q]

    # Re-sort each row left-to-right
    valid_rows = [sorted(r, key=lambda c: cv2.boundingRect(c)[0]) for r in valid_rows]

    answers = []
    debug_img = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    for row in valid_rows[:TOTAL_QUESTIONS]:
        filled = []
        pixel_vals = []
        for bubble in row:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            pixel_vals.append(total)

        max_val = max(pixel_vals)
        threshold_val = max_val * 0.6  # 60 % of max = filled

        for idx, (bubble, val) in enumerate(zip(row, pixel_vals)):
            color = (0, 200, 0)
            if val >= threshold_val:
                filled.append(idx)
                color = (0, 0, 255)
            if debug:
                cv2.drawContours(debug_img, [bubble], -1, color, 2)

        if len(filled) == 1:
            answers.append(filled[0])
        elif len(filled) == 0:
            answers.append(-1)   # unanswered
        else:
            answers.append(-2)   # multiple marks

    # Pad if fewer rows detected
    while len(answers) < TOTAL_QUESTIONS:
        answers.append(-1)

    return answers[:TOTAL_QUESTIONS], debug_img


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_sheet(answers, answer_key, marks_correct=1.0, marks_wrong=-0.25):
    correct = wrong = unattempted = multi = 0
    details = []
    for i, (ans, key) in enumerate(zip(answers, answer_key)):
        opt_map = {0: "A", 1: "B", 2: "C", 3: "D", -1: "—", -2: "Multi"}
        marked = opt_map.get(ans, "?")
        correct_opt = opt_map.get(key, "?")
        if ans == -1:
            unattempted += 1
            status = "Unattempted"
            score = 0
        elif ans == -2:
            multi += 1
            status = "Multi-Mark"
            score = marks_wrong
            wrong += 1
        elif ans == key:
            correct += 1
            status = "Correct ✅"
            score = marks_correct
        else:
            wrong += 1
            status = "Wrong ❌"
            score = marks_wrong
        details.append({
            "Q#": i + 1,
            "Marked": marked,
            "Correct": correct_opt,
            "Status": status,
            "Score": score,
        })
    total = correct * marks_correct + wrong * marks_wrong
    return {
        "correct": correct,
        "wrong": wrong,
        "unattempted": unattempted,
        "multi": multi,
        "total": round(total, 2),
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def process_image(uploaded_file, answer_key, marks_correct, marks_wrong, debug):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None, None, "Could not decode image."

    gray, thresh = preprocess(img_bgr)
    doc_cnt = find_document_contour(thresh)

    if doc_cnt is not None:
        warped = four_point_transform(gray, doc_cnt)
    else:
        warped = gray  # Use as-is if no boundary found

    # Resize to standard height for consistent processing
    target_h = 1400
    scale = target_h / warped.shape[0]
    warped_resized = cv2.resize(warped, (int(warped.shape[1] * scale), target_h))

    answers, debug_img = detect_answers(warped_resized, debug=debug)
    result = score_sheet(answers, answer_key, marks_correct, marks_wrong)
    return answers, result, debug_img, None


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar: Answer Key + Settings ───────────────────────────────────────
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/FF9933/FFFFFF?text=Yuva+Gyan+Mahotsav",
                 use_column_width=True)
        st.title("⚙️ Settings")

        st.subheader("📋 Answer Key")
        key_method = st.radio("Enter key via:", ["Manual Entry", "Upload CSV/JSON"])

        answer_key = []
        opt_map_rev = {"A": 0, "B": 1, "C": 2, "D": 3}

        if key_method == "Manual Entry":
            st.caption("Enter A / B / C / D for each question.")
            cols_k = st.columns(3)
            for i in range(TOTAL_QUESTIONS):
                col = cols_k[i % 3]
                val = col.selectbox(f"Q{i+1}", ["A","B","C","D"],
                                    key=f"key_{i}", label_visibility="collapsed")
                answer_key.append(opt_map_rev[val])
            # Show compact preview
            key_str = " ".join([["A","B","C","D"][k] for k in answer_key])
            st.caption(f"Key: `{key_str}`")
        else:
            key_file = st.file_uploader("Upload key (CSV or JSON)", type=["csv","json"])
            if key_file:
                try:
                    if key_file.name.endswith(".json"):
                        data = json.load(key_file)
                        # Accept {"1":"A","2":"B",...} or ["A","B",...]
                        if isinstance(data, list):
                            answer_key = [opt_map_rev[x.upper()] for x in data[:TOTAL_QUESTIONS]]
                        else:
                            answer_key = [opt_map_rev[data[str(i+1)].upper()] for i in range(TOTAL_QUESTIONS)]
                    else:
                        df_key = pd.read_csv(key_file)
                        col_name = df_key.columns[-1]
                        answer_key = [opt_map_rev[str(v).strip().upper()] for v in df_key[col_name][:TOTAL_QUESTIONS]]
                    st.success(f"✅ Key loaded ({len(answer_key)} questions)")
                except Exception as e:
                    st.error(f"Key parse error: {e}")
                    answer_key = [0] * TOTAL_QUESTIONS
            else:
                answer_key = [0] * TOTAL_QUESTIONS
                st.info("No key uploaded – defaulting all to A.")

        st.subheader("🎯 Marking Scheme")
        marks_correct = st.number_input("Marks for Correct", value=1.0, step=0.25)
        marks_wrong   = st.number_input("Marks for Wrong (negative)",
                                        value=-0.25, max_value=0.0, step=0.25)

        st.subheader("🔬 Debug")
        show_debug = st.checkbox("Show bubble detection overlay", value=False)

    # ── Main Area ─────────────────────────────────────────────────────────────
    st.title("📝 Yuva Gyan Mahotsav 2026 – OMR Auto Grader")
    st.markdown("Upload scanned OMR sheets to auto-detect and grade answers.")

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Grade", "📊 Batch Results", "ℹ️ How to Use"])

    # ── Tab 1: Upload & Grade ─────────────────────────────────────────────────
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload OMR sheet images (JPG/PNG/PDF*)",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if len(answer_key) < TOTAL_QUESTIONS:
                st.warning("⚠️ Answer key is incomplete. Please set it in the sidebar.")

            all_results = []
            for idx, uf in enumerate(uploaded_files):
                st.divider()
                st.subheader(f"Sheet {idx+1}: `{uf.name}`")

                with st.spinner("Processing..."):
                    uf.seek(0)
                    answers, result, debug_img, err = process_image(
                        uf, answer_key, marks_correct, marks_wrong, show_debug
                    )

                if err:
                    st.error(f"❌ Error: {err}")
                    continue

                # Layout
                col_img, col_score = st.columns([1, 1])

                with col_img:
                    if show_debug and debug_img is not None:
                        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                                 caption="Bubble Detection Overlay", use_column_width=True)
                    else:
                        uf.seek(0)
                        st.image(uf, caption="Uploaded Sheet", use_column_width=True)

                with col_score:
                    # Score summary
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("✅ Correct", result["correct"])
                    m2.metric("❌ Wrong", result["wrong"])
                    m3.metric("⬜ Unattempted", result["unattempted"])
                    m4.metric("🏆 Total Score", result["total"])

                    pct = (result["correct"] / TOTAL_QUESTIONS) * 100
                    grade = ("A+" if pct>=90 else "A" if pct>=80 else "B" if pct>=70
                             else "C" if pct>=60 else "D" if pct>=50 else "F")
                    st.progress(pct / 100)
                    st.markdown(f"**Grade: {grade}** &nbsp;&nbsp; ({pct:.1f}%)")

                    if result["multi"] > 0:
                        st.warning(f"⚠️ {result['multi']} questions have multiple marks.")

                # Detailed table
                with st.expander("📋 Question-wise breakdown"):
                    df_detail = pd.DataFrame(result["details"])
                    def color_status(val):
                        if "Correct" in str(val): return "background-color: #d4edda"
                        if "Wrong" in str(val) or "Multi" in str(val): return "background-color: #f8d7da"
                        return ""
                    st.dataframe(
                        df_detail.style.applymap(color_status, subset=["Status"]),
                        use_container_width=True, height=300
                    )

                # Download single result
                csv_single = df_detail.to_csv(index=False).encode()
                st.download_button(
                    f"⬇️ Download result for {uf.name}",
                    data=csv_single,
                    file_name=f"result_{uf.name.rsplit('.',1)[0]}.csv",
                    mime="text/csv",
                    key=f"dl_{idx}"
                )

                all_results.append({
                    "File": uf.name,
                    "Correct": result["correct"],
                    "Wrong": result["wrong"],
                    "Unattempted": result["unattempted"],
                    "Multi-Mark": result["multi"],
                    "Total Score": result["total"],
                    "Percentage": round(pct, 2),
                    "Grade": grade,
                    "Answers": ",".join([["A","B","C","D","—","Multi"][min(a,5)] for a in answers]),
                })

            # Store for Batch tab
            if all_results:
                st.session_state["batch_results"] = all_results

    # ── Tab 2: Batch Results ──────────────────────────────────────────────────
    with tab2:
        if "batch_results" not in st.session_state or not st.session_state["batch_results"]:
            st.info("Upload and grade sheets in Tab 1 to see batch results here.")
        else:
            br = st.session_state["batch_results"]
            df_batch = pd.DataFrame(br)

            st.subheader(f"📊 Summary – {len(df_batch)} sheet(s) graded")

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Score", f"{df_batch['Total Score'].mean():.2f}")
            c2.metric("Highest", f"{df_batch['Total Score'].max():.2f}")
            c3.metric("Lowest", f"{df_batch['Total Score'].min():.2f}")

            st.dataframe(df_batch.drop(columns=["Answers"]), use_container_width=True)

            # Grade distribution
            st.subheader("Grade Distribution")
            grade_counts = df_batch["Grade"].value_counts().reset_index()
            grade_counts.columns = ["Grade","Count"]
            st.bar_chart(grade_counts.set_index("Grade"))

            # Download all
            csv_all = df_batch.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download All Results (CSV)",
                data=csv_all,
                file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            # JSON export
            json_all = json.dumps(br, indent=2).encode()
            st.download_button(
                "⬇️ Download All Results (JSON)",
                data=json_all,
                file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # ── Tab 3: How to Use ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
## 📖 How to Use the OMR Grader

### Step 1 – Set the Answer Key
- In the **sidebar**, enter the correct answer (A/B/C/D) for all 60 questions.
- Or upload a **CSV** (one column of answers) or **JSON** file.

### Step 2 – Configure Marking Scheme
- Default: **+1** for correct, **−0.25** for wrong.
- Adjust in the sidebar as needed.

### Step 3 – Upload OMR Sheets
- Go to **Upload & Grade** tab.
- Upload one or more scanned OMR images (JPG/PNG).
- Best scan quality: **300 DPI**, flat, well-lit, no shadows.

### Step 4 – Review Results
- Each sheet shows a score summary and question-wise breakdown.
- Switch to **Batch Results** to compare all sheets and download reports.

---
### 📸 Scanning Tips
| Do ✅ | Avoid ❌ |
|---|---|
| Scan at 300 DPI | Low-res phone photos |
| Keep sheet flat | Wrinkled / folded sheets |
| Good even lighting | Strong shadows / glare |
| Full sheet visible | Cropped edges |
| Dark filled bubbles | Lightly filled bubbles |

### 📁 Answer Key Format (CSV)
```
Question,Answer
1,A
2,C
3,B
...
```

### 📁 Answer Key Format (JSON)
```json
{"1":"A","2":"C","3":"B",...}
```
or as a list:
```json
["A","C","B","D",...]
```

---
*Powered by OpenCV + Streamlit · Yuva Gyan Mahotsav 2026 · Tiranga Yuva Samiti*
        """)


if __name__ == "__main__":
    main()
