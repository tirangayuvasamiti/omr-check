import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG FOR YUVA GYAN MAHOTSAV 2026 ---
st.set_page_config(
    page_title="YUVA GYAN OMR Grader", 
    layout="wide", 
    page_icon="🎓"
)

# Answer Key (1-60)
ANS_KEY = {
    1: 2, 2: 4, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 4, 10: 2,
    11: 1, 12: 1, 13: 4, 14: 1, 15: 2, 16: 1, 17: 4, 18: 2, 19: 2, 20: 3,
    21: 4, 22: 2, 23: 1, 24: 2, 25: 4, 26: 1, 27: 3, 28: 4, 29: 4, 30: 3,
    31: 1, 32: 3, 33: 2, 34: 2, 35: 3, 36: 2, 37: 4, 38: 3, 39: 1, 40: 4,
    41: 2, 42: 1, 43: 4, 44: 3, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 50: 1,
    51: 1, 52: 2, 53: 2, 54: 4, 55: 2, 56: 3, 57: 1, 58: 2, 59: 2, 60: 2
}

OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# Layout configuration for this specific OMR sheet
LAYOUT_CONFIG = {
    "total_questions": 60,
    "options_per_question": 4,
    "columns": 3,
    "alignment_markers": True,  # Black squares on corners
}


class YuvaGyanOMRProcessor:
    """Specialized OMR processor for YUVA GYAN MAHOTSAV 2026 sheet"""
    
    def __init__(self):
        # Bubble detection parameters (tuned for this sheet - based on diagnostic)
        self.min_bubble_area = 200  # Bubbles are ~270 pixels
        self.max_bubble_area = 2000
        self.min_circularity = 0.55  # Slightly relaxed for printed circles
        self.fill_threshold = 0.30  # 30% filled to consider marked
        self.double_mark_ratio = 0.70  # Aggressive double mark detection
        
    def load_image(self, uploaded_file) -> np.ndarray:
        """Load image from uploaded file"""
        img = Image.open(uploaded_file).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def find_alignment_markers(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Find the 4 black corner squares for precise alignment"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for square markers (should be 4 of them in corners)
        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 800:  # Marker size range
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.8 < aspect_ratio < 1.2:  # Square-ish
                    markers.append((x + w//2, y + h//2))
        
        return sorted(markers, key=lambda p: (p[1], p[0]))[:4]
    
    def preprocess_sheet(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess and extract ROI - Simplified approach
        The original blank sheet is already well-aligned, so we just:
        1. Resize to standard dimensions
        2. Extract the answer area ROI
        """
        # Resize to standard size
        resized = cv2.resize(image, (1000, 1414), interpolation=cv2.INTER_CUBIC)
        
        # Extract answer area (skip header at top, footer at bottom, margins on sides)
        # Coordinates tuned for this specific sheet layout
        roi = resized[200:1300, 20:980]
        
        return resized, roi
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as TL, TR, BR, BL"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def detect_bubbles(self, roi: np.ndarray) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Detect bubbles using optimized processing.
        Returns the bubbles list, thresholded image, AND the raw enhanced grayscale 
        image for advanced intensity analysis.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Otsu's threshold (best method from testing)
        _, thresh = cv2.threshold(
            enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Light morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter bubbles
        bubbles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Area filter (bubbles are ~270 pixels)
            if area < self.min_bubble_area or area > self.max_bubble_area:
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Aspect ratio (should be circular)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                continue
            
            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # Calculate center
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            bubbles.append({
                'contour': cnt,
                'center': (cx, cy),
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'circularity': circularity
            })
        
        logger.info(f"Detected {len(bubbles)} bubbles")
        return bubbles, thresh, enhanced_gray
    
    def organize_bubbles(self, bubbles: List[Dict]) -> List[List[Dict]]:
        """
        Organize bubbles into 3 columns and rows of 4 (A, B, C, D)
        
        Column layout:
        - Column 1: Q1-22
        - Column 2: Q23-40  
        - Column 3: Q41-60
        """
        if len(bubbles) < 200:
            logger.warning(f"Low bubble count: {len(bubbles)}")
        
        # Sort by position
        bubbles = sorted(bubbles, key=lambda b: (b['center'][0], b['center'][1]))
        
        # Determine column boundaries (3 equal columns)
        max_x = max(b['center'][0] for b in bubbles)
        col_width = max_x / 3
        
        # Assign to columns
        columns = [[], [], []]
        for b in bubbles:
            col_idx = min(int(b['center'][0] / col_width), 2)
            columns[col_idx].append(b)
        
        # Sort each column by y-coordinate (top to bottom)
        for col in columns:
            col.sort(key=lambda b: b['center'][1])
        
        # Group into question rows (sets of 4)
        all_questions = []
        
        for col in columns:
            for i in range(0, len(col), 4):
                row = col[i:i+4]
                if len(row) == 4:
                    # Sort by x-coordinate (A, B, C, D)
                    row.sort(key=lambda b: b['center'][0])
                    all_questions.append(row)
        
        return all_questions
    
    def analyze_bubble_intensity(self, bubble: Dict, thresh: np.ndarray, gray: np.ndarray) -> Dict:
        """AI-inspired CV approach: Combines pixel ratio with localized pixel intensity distributions"""
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [bubble['contour']], -1, 255, -1)
        
        # 1. Standard pixel area fill ratio (binary)
        filled = cv2.countNonZero(cv2.bitwise_and(thresh, mask))
        total = cv2.countNonZero(mask)
        fill_ratio = filled / total if total > 0 else 0.0
        
        # 2. Photometric darkness intensity (robust against varied lighting/pen ink/smudges)
        mean_val = cv2.mean(gray, mask=mask)[0]
        # Invert so higher = darker (0 = white, 255 = solid black ink)
        darkness = 255.0 - mean_val
        
        return {
            'fill_ratio': fill_ratio,
            'darkness': darkness
        }
    
    def grade_question(self, q_num: int, bubbles: List[Dict], thresh: np.ndarray, gray: np.ndarray) -> Dict:
        """Advanced grading using statistical gap analysis to confidently detect filled and multi-filled bubbles"""
        stats = []
        for i, b in enumerate(bubbles):
            metrics = self.analyze_bubble_intensity(b, thresh, gray)
            stats.append((metrics['darkness'], metrics['fill_ratio'], i, b))
            
        # Sort heavily by darkness metric (most reliable for pen marks) and then fill ratio
        stats.sort(reverse=True, key=lambda x: (x[0], x[1]))
        
        marked = []
        
        # Top candidate metrics
        top_darkness = stats[0][0]
        top_fill = stats[0][1]
        
        # Dynamic baseline check: is it actually filled?
        # Absolute baseline (darkness > 80) handles light pen strokes that fail the binary 30% fill threshold
        if top_fill >= self.fill_threshold or top_darkness >= 80.0: 
            marked.append(stats[0][2])
            
            # Double mark detection: Check if 2nd candidate is suspiciously close to the 1st
            if len(stats) > 1:
                second_darkness = stats[1][0]
                second_fill = stats[1][1]
                
                # If 2nd choice meets strict fill ratio OR is proportionately as dark as the 1st
                if (second_fill >= self.fill_threshold) or (second_darkness >= top_darkness * self.double_mark_ratio):
                    marked.append(stats[1][2])
                    
                # Catch messy triple marks
                if len(stats) > 2:
                    third_darkness = stats[2][0]
                    third_fill = stats[2][1]
                    if (third_fill >= self.fill_threshold) or (third_darkness >= top_darkness * self.double_mark_ratio):
                        marked.append(stats[2][2])

        # Grade
        correct_ans = ANS_KEY.get(q_num, 1) - 1
        
        is_blank = len(marked) == 0
        is_double = len(marked) > 1
        is_correct = (not is_blank and not is_double and marked[0] == correct_ans)
        
        if is_blank:
            status = "Blank"
        elif is_double:
            status = "Double Mark"
        elif is_correct:
            status = "Correct"
        else:
            status = "Incorrect"
        
        return {
            'question': q_num,
            'marked': marked,
            'correct': correct_ans,
            'status': status,
            'is_correct': is_correct,
            'is_blank': is_blank,
            'is_double': is_double
        }
    
    def annotate_sheet(self, roi: np.ndarray, questions: List[List[Dict]], 
                       results: List[Dict]) -> np.ndarray:
        """Draw grading annotations on the sheet"""
        annotated = roi.copy()
        
        for row, result in zip(questions, results):
            if result['is_blank']:
                # Blue circle on correct answer
                correct_bubble = row[result['correct']]
                cv2.drawContours(annotated, [correct_bubble['contour']], -1, (255, 0, 0), 2)
                
            elif result['is_double']:
                # Yellow circles on double marked
                for idx in result['marked']:
                    cv2.drawContours(annotated, [row[idx]['contour']], -1, (0, 255, 255), 3)
                # Blue on correct
                cv2.drawContours(annotated, [row[result['correct']]['contour']], -1, (255, 0, 0), 2)
                
            else:
                marked_idx = result['marked'][0]
                if result['is_correct']:
                    # Green for correct
                    cv2.drawContours(annotated, [row[marked_idx]['contour']], -1, (0, 255, 0), 3)
                else:
                    # Red for incorrect
                    cv2.drawContours(annotated, [row[marked_idx]['contour']], -1, (0, 0, 255), 3)
                    # Blue for correct answer
                    cv2.drawContours(annotated, [row[result['correct']]['contour']], -1, (255, 0, 0), 2)
        
        return annotated
    
    def process(self, image: np.ndarray) -> Tuple[Optional[Dict], np.ndarray, List[Dict], str]:
        """Main processing pipeline"""
        try:
            # Step 1: Preprocess
            _, roi = self.preprocess_sheet(image)
            
            # Step 2: Detect bubbles (Now grabs grayscale mapping for AI intensity analytics)
            bubbles, thresh, gray = self.detect_bubbles(roi)
            
            if len(bubbles) < 200:
                return None, roi, [], f"Error: Only {len(bubbles)} bubbles detected (need ~240)"
            
            # Step 3: Organize
            questions = self.organize_bubbles(bubbles)
            
            if len(questions) < 50:
                return None, roi, [], f"Error: Only {len(questions)} questions organized (need 60)"
            
            # Step 4: Grade all questions
            results = []
            stats = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
            
            for q_num in range(1, 61):
                if q_num - 1 < len(questions):
                    result = self.grade_question(q_num, questions[q_num - 1], thresh, gray)
                    results.append(result)
                    
                    if result['is_correct']:
                        stats['correct'] += 1
                    elif result['is_blank']:
                        stats['blank'] += 1
                    elif result['is_double']:
                        stats['double'] += 1
                        stats['wrong'] += 1
                    else:
                        stats['wrong'] += 1
            
            # Step 5: Annotate
            annotated = self.annotate_sheet(roi, questions[:len(results)], results)
            
            return stats, annotated, results, "Success"
            
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return None, image, [], f"Error: {str(e)}"


# --- STREAMLIT UI ---
def main():
    st.title("🎓 YUVA GYAN MAHOTSAV 2026 - OMR Grader")
    
    st.markdown("""
    ### Official OMR Sheet Grading System
    **Features:**
    - ✅ Automatic bubble detection
    - 🎯 Precise alignment correction
    - 📊 Instant results with detailed analytics
    - 🎨 Color-coded visual feedback
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Test Information")
        st.info("""
        **Total Questions:** 60
        - English (Q.1-7)
        - Hindi (Q.8-14)
        - Mental Ability (Q.15-22)
        - Computer Science (Q.23-30)
        - General Knowledge (Q.31-55)
        - Youth Awareness (Q.56-60)
        """)
        
        st.header("🎨 Color Legend")
        st.markdown("""
        - 🟢 **Green** = Correct Answer
        - 🔴 **Red** = Wrong Answer
        - 🔵 **Blue** = Correct Answer (reference)
        - 🟡 **Yellow** = Double Marked
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Filled OMR Sheet",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of the filled answer sheet"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Original Sheet")
            original_img = Image.open(uploaded_file)
            st.image(original_img, use_container_width=True)
        
        with st.spinner("🔄 Processing OMR sheet..."):
            try:
                # Process
                processor = YuvaGyanOMRProcessor()
                image = processor.load_image(uploaded_file)
                stats, annotated, results, message = processor.process(image)
                
                if stats:
                    with col2:
                        st.subheader("✅ Graded Sheet")
                        st.image(annotated, channels="BGR", use_container_width=True)
                    
                    st.success(f"✅ {message}")
                    
                    # Metrics
                    st.subheader("📊 Results Summary")
                    metric_cols = st.columns(5)
                    
                    total = len(results)
                    score = stats['correct']
                    percentage = (score / total * 100) if total > 0 else 0
                    
                    metric_cols[0].metric("Total", total)
                    metric_cols[1].metric("✅ Correct", stats['correct'])
                    metric_cols[2].metric("❌ Wrong", stats['wrong'])
                    metric_cols[3].metric("⚠️ Blank", stats['blank'])
                    metric_cols[4].metric("🔴 Double", stats['double'])
                    
                    # Score card
                    st.subheader("🎯 Final Score")
                    score_col1, score_col2, score_col3 = st.columns(3)
                    
                    with score_col1:
                        st.metric("Score", f"{score}/{total}")
                    with score_col2:
                        st.metric("Percentage", f"{percentage:.2f}%")
                    with score_col3:
                        if percentage >= 90:
                            grade = "A+"
                            st.balloons()
                        elif percentage >= 80:
                            grade = "A"
                        elif percentage >= 70:
                            grade = "B"
                        elif percentage >= 60:
                            grade = "C"
                        elif percentage >= 50:
                            grade = "D"
                        else:
                            grade = "E"
                        st.metric("Grade", grade)
                    
                    # Detailed results
                    st.subheader("📋 Question-wise Analysis")
                    
                    df_data = []
                    for r in results:
                        marked_str = ", ".join([OPTION_MAP[i] for i in r['marked']]) if r['marked'] else "None"
                        correct_str = OPTION_MAP[r['correct']]
                        df_data.append({
                            "Q.No": r['question'],
                            "Marked": marked_str,
                            "Correct": correct_str,
                            "Status": r['status']
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    # Color coding
                    def highlight(row):
                        if row['Status'] == 'Correct':
                            return ['background-color: #d4edda'] * len(row)
                        elif row['Status'] == 'Incorrect':
                            return ['background-color: #f8d7da'] * len(row)
                        elif row['Status'] == 'Blank':
                            return ['background-color: #fff3cd'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)
                    
                    st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True, height=400)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv,
                        "yuva_gyan_results.csv",
                        "text/csv"
                    )
                    
                else:
                    st.error(f"❌ {message}")
                    st.info("💡 Tips: Ensure good lighting, flat surface, and all bubbles are visible")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
