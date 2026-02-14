import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
st.set_page_config(
    page_title="Yuva Gyan AI Vision Grader Pro", 
    layout="wide", 
    page_icon="🎯"
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
TOTAL_QUESTIONS = 60
OPTIONS_PER_QUESTION = 4
EXPECTED_BUBBLES = TOTAL_QUESTIONS * OPTIONS_PER_QUESTION  # 240

@dataclass
class Bubble:
    """Represents a single bubble/circle on the OMR sheet"""
    contour: np.ndarray
    center: Tuple[int, int]
    x: int
    y: int
    w: int
    h: int
    area: float
    circularity: float
    
@dataclass
class QuestionResult:
    """Represents the result for a single question"""
    question_num: int
    marked_options: List[int]  # 0=A, 1=B, 2=C, 3=D
    correct_answer: int
    is_correct: bool
    is_blank: bool
    is_double_marked: bool
    status: str


class OMRProcessor:
    """Main OMR processing engine with advanced computer vision techniques"""
    
    def __init__(self):
        # Bubble detection parameters (auto-tuned)
        self.min_bubble_area = 400
        self.max_bubble_area = 2500
        self.min_circularity = 0.65
        self.fill_threshold = 0.25  # 25% filled to consider marked
        self.double_mark_threshold = 0.80  # If 2nd bubble is 80%+ of 1st, it's double marked
        
    def load_image(self, uploaded_file) -> np.ndarray:
        """Load and convert uploaded image to OpenCV format"""
        img = Image.open(uploaded_file).convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def find_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the OMR sheet boundary using edge detection and contour approximation.
        Returns the 4-corner polygon of the sheet.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Multi-stage edge detection for better results
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area and find the largest quadrilateral
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:10]:  # Check top 10 largest
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4 and cv2.contourArea(approx) > 50000:
                return approx.reshape(4, 2).astype(np.float32)
        
        return None
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in [top-left, top-right, bottom-right, bottom-left] order"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
        
        return rect
    
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get bird's eye view"""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Calculate width and height
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)
        
        # Get perspective transform matrix and warp
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the OMR sheet:
        1. Find document boundary
        2. Apply perspective transform
        3. Resize to standard dimensions
        4. Crop to ROI (remove headers/footers)
        """
        # Find and transform document
        doc_contour = self.find_document_contour(image)
        
        if doc_contour is not None:
            warped = self.four_point_transform(image, doc_contour)
        else:
            logger.warning("Could not find document boundary, using full image")
            warped = image.copy()
        
        # Resize to standard dimensions for consistency
        warped = cv2.resize(warped, (1200, 1600), interpolation=cv2.INTER_CUBIC)
        
        # Crop to ROI (remove header and footer)
        # These values may need adjustment based on your specific form layout
        roi = warped[300:1500, 50:1150]
        
        return warped, roi
    
    def calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate circularity metric: 4π * area / perimeter²
        Perfect circle = 1.0, other shapes < 1.0
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity
    
    def detect_bubbles(self, roi: np.ndarray) -> List[Bubble]:
        """
        Detect all bubbles in the ROI using advanced filtering:
        1. Adaptive thresholding
        2. Morphological operations
        3. Contour detection
        4. Shape filtering (circularity, size, aspect ratio)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Multiple thresholding approaches for robustness
        # Method 1: Adaptive threshold
        thresh1 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Method 2: Otsu's threshold
        _, thresh2 = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Combine both methods
        thresh = cv2.bitwise_or(thresh1, thresh2)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_bubble_area or area > self.max_bubble_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (bubbles should be roughly square)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
            
            # Calculate circularity
            circularity = self.calculate_circularity(contour)
            
            # Filter by circularity (must be reasonably circular)
            if circularity < self.min_circularity:
                continue
            
            # Calculate center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            bubble = Bubble(
                contour=contour,
                center=(cx, cy),
                x=x, y=y, w=w, h=h,
                area=area,
                circularity=circularity
            )
            bubbles.append(bubble)
        
        logger.info(f"Detected {len(bubbles)} potential bubbles")
        return bubbles
    
    def cluster_bubbles_into_grid(self, bubbles: List[Bubble]) -> List[List[Bubble]]:
        """
        Organize bubbles into a structured grid (questions and options).
        Assumes 3-column layout with 20 questions per column.
        """
        if len(bubbles) < EXPECTED_BUBBLES * 0.8:  # Allow 20% tolerance
            logger.warning(f"Expected ~{EXPECTED_BUBBLES} bubbles, found {len(bubbles)}")
        
        # Sort bubbles by position (left to right, top to bottom)
        bubbles_sorted = sorted(bubbles, key=lambda b: (b.center[0] // 300, b.center[1]))
        
        # Split into 3 columns based on x-coordinate
        col_width = max(b.center[0] for b in bubbles_sorted) // 3
        
        columns = [[], [], []]
        for bubble in bubbles_sorted:
            col_idx = min(bubble.center[0] // col_width, 2)
            columns[col_idx].append(bubble)
        
        # Sort each column by y-coordinate (top to bottom)
        for col in columns:
            col.sort(key=lambda b: b.center[1])
        
        # Group into rows of 4 (A, B, C, D)
        all_question_rows = []
        
        for col in columns:
            for i in range(0, len(col), OPTIONS_PER_QUESTION):
                row = col[i:i + OPTIONS_PER_QUESTION]
                if len(row) == OPTIONS_PER_QUESTION:
                    # Sort by x-coordinate to ensure A, B, C, D order
                    row.sort(key=lambda b: b.center[0])
                    all_question_rows.append(row)
        
        return all_question_rows
    
    def calculate_bubble_fill(self, bubble: Bubble, thresh_image: np.ndarray) -> float:
        """
        Calculate how filled a bubble is (0.0 to 1.0).
        Uses pixel counting within the bubble contour.
        """
        mask = np.zeros(thresh_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [bubble.contour], -1, 255, -1)
        
        # Count filled pixels
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh_image, mask))
        
        # Total possible pixels in the bubble
        total_pixels = cv2.countNonZero(mask)
        
        if total_pixels == 0:
            return 0.0
        
        fill_ratio = filled_pixels / total_pixels
        return fill_ratio
    
    def grade_question(self, question_num: int, bubbles: List[Bubble], 
                       thresh_image: np.ndarray) -> QuestionResult:
        """
        Grade a single question by analyzing bubble fill levels.
        Handles blank answers and double marks.
        """
        # Calculate fill ratio for each option
        fill_ratios = []
        for i, bubble in enumerate(bubbles):
            fill = self.calculate_bubble_fill(bubble, thresh_image)
            fill_ratios.append((fill, i, bubble))
        
        # Sort by fill ratio (descending)
        fill_ratios.sort(reverse=True, key=lambda x: x[0])
        
        # Determine marked options
        marked_options = []
        
        # Check if most filled bubble exceeds threshold
        if fill_ratios[0][0] >= self.fill_threshold:
            marked_options.append(fill_ratios[0][1])
            
            # Check for double marks (if second bubble is also heavily filled)
            if (len(fill_ratios) > 1 and 
                fill_ratios[1][0] >= self.fill_threshold and
                fill_ratios[1][0] >= fill_ratios[0][0] * self.double_mark_threshold):
                marked_options.append(fill_ratios[1][1])
        
        # Get correct answer
        correct_answer_idx = ANS_KEY.get(question_num, 1) - 1  # Convert to 0-indexed
        
        # Determine result
        is_blank = len(marked_options) == 0
        is_double = len(marked_options) > 1
        is_correct = (not is_blank and not is_double and 
                     marked_options[0] == correct_answer_idx)
        
        # Determine status
        if is_blank:
            status = "Blank"
        elif is_double:
            status = "Double Mark"
        elif is_correct:
            status = "Correct"
        else:
            status = "Incorrect"
        
        return QuestionResult(
            question_num=question_num,
            marked_options=marked_options,
            correct_answer=correct_answer_idx,
            is_correct=is_correct,
            is_blank=is_blank,
            is_double_marked=is_double,
            status=status
        )
    
    def annotate_image(self, roi: np.ndarray, question_rows: List[List[Bubble]], 
                       results: List[QuestionResult]) -> np.ndarray:
        """
        Draw annotations on the image showing grading results:
        - Green: Correct answer
        - Red: Incorrect answer
        - Blue: Correct answer (when student marked wrong or left blank)
        - Yellow: Double marked
        """
        annotated = roi.copy()
        
        for idx, (row, result) in enumerate(zip(question_rows, results)):
            if result.is_blank:
                # Draw blue circle around correct answer
                correct_bubble = row[result.correct_answer]
                cv2.drawContours(annotated, [correct_bubble.contour], -1, (255, 0, 0), 2)
                
            elif result.is_double_marked:
                # Draw yellow circles around double marked bubbles
                for opt_idx in result.marked_options:
                    bubble = row[opt_idx]
                    cv2.drawContours(annotated, [bubble.contour], -1, (0, 255, 255), 3)
                # Show correct answer in blue
                correct_bubble = row[result.correct_answer]
                cv2.drawContours(annotated, [correct_bubble.contour], -1, (255, 0, 0), 2)
                
            else:
                marked_idx = result.marked_options[0]
                marked_bubble = row[marked_idx]
                
                if result.is_correct:
                    # Green for correct
                    cv2.drawContours(annotated, [marked_bubble.contour], -1, (0, 255, 0), 3)
                else:
                    # Red for incorrect
                    cv2.drawContours(annotated, [marked_bubble.contour], -1, (0, 0, 255), 3)
                    # Blue for correct answer
                    correct_bubble = row[result.correct_answer]
                    cv2.drawContours(annotated, [correct_bubble.contour], -1, (255, 0, 0), 2)
        
        return annotated
    
    def process(self, image: np.ndarray) -> Tuple[Dict, np.ndarray, List[Dict], str]:
        """
        Main processing pipeline:
        1. Preprocess image
        2. Detect bubbles
        3. Organize into grid
        4. Grade each question
        5. Annotate results
        """
        try:
            # Step 1: Preprocess
            _, roi = self.preprocess_image(image)
            
            # Step 2: Detect bubbles
            bubbles = self.detect_bubbles(roi)
            
            if len(bubbles) < 200:  # Need at least 200 bubbles
                return None, roi, [], f"Error: Only detected {len(bubbles)} bubbles (expected ~240). Please check image quality."
            
            # Step 3: Organize into grid
            question_rows = self.cluster_bubbles_into_grid(bubbles)
            
            if len(question_rows) != TOTAL_QUESTIONS:
                logger.warning(f"Expected {TOTAL_QUESTIONS} questions, found {len(question_rows)}")
            
            # Create threshold image for fill detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Step 4: Grade all questions
            results = []
            stats = {"correct": 0, "wrong": 0, "blank": 0, "double": 0}
            
            for q_num, row in enumerate(question_rows, start=1):
                if q_num > TOTAL_QUESTIONS:
                    break
                
                result = self.grade_question(q_num, row, thresh)
                results.append(result)
                
                if result.is_correct:
                    stats["correct"] += 1
                elif result.is_blank:
                    stats["blank"] += 1
                elif result.is_double_marked:
                    stats["double"] += 1
                    stats["wrong"] += 1
                else:
                    stats["wrong"] += 1
            
            # Step 5: Annotate image
            annotated = self.annotate_image(roi, question_rows, results)
            
            # Prepare logs
            logs = []
            for result in results:
                marked_str = ", ".join([OPTION_MAP[i] for i in result.marked_options]) if result.marked_options else "None"
                correct_str = OPTION_MAP[result.correct_answer]
                
                logs.append({
                    "Question": result.question_num,
                    "Marked": marked_str,
                    "Correct": correct_str,
                    "Status": result.status
                })
            
            return stats, annotated, logs, "Success"
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return None, image, [], f"Error: {str(e)}"


# --- STREAMLIT UI ---
def main():
    st.title("🎯 AI Vision OMR Grader Pro")
    st.markdown("""
    **Advanced OMR grading system with:**
    - 🔍 Adaptive bubble detection
    - 📐 Automatic perspective correction
    - 🎨 CLAHE image enhancement
    - 🧠 Intelligent circularity filtering
    - ⚡ High accuracy grading
    """)
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Upload an OMR sheet to begin grading")
        
        with st.expander("📋 Answer Key Preview"):
            st.write("Total Questions: 60")
            st.write("Options per Question: 4 (A, B, C, D)")
            key_df = pd.DataFrame([
                {"Q": k, "Answer": OPTION_MAP[v-1]} 
                for k, v in list(ANS_KEY.items())[:10]
            ])
            st.dataframe(key_df, hide_index=True)
            st.caption("Showing first 10 questions...")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload OMR Sheet", 
        type=['jpg', 'png', 'jpeg'],
        help="Upload a clear image of the filled OMR sheet"
    )
    
    if uploaded_file:
        with st.spinner("🔄 Processing OMR sheet... Please wait."):
            try:
                # Initialize processor
                processor = OMRProcessor()
                
                # Load image
                image = processor.load_image(uploaded_file)
                
                # Process
                stats, annotated_img, logs, message = processor.process(image)
                
                if stats is not None:
                    st.success(f"✅ {message} - Graded {len(logs)} questions")
                    
                    # Display metrics
                    st.subheader("📊 Results Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(logs)
                    score = stats['correct']
                    percentage = (score / total * 100) if total > 0 else 0
                    
                    col1.metric("✅ Correct", stats['correct'], 
                               delta=f"{percentage:.1f}%")
                    col2.metric("❌ Wrong", stats['wrong'])
                    col3.metric("⚠️ Blank", stats['blank'])
                    col4.metric("🔴 Double Mark", stats['double'])
                    
                    # Score card
                    if percentage >= 90:
                        st.balloons()
                        st.success(f"🎉 Outstanding! Score: {score}/{total} ({percentage:.1f}%)")
                    elif percentage >= 75:
                        st.success(f"👍 Great job! Score: {score}/{total} ({percentage:.1f}%)")
                    elif percentage >= 60:
                        st.info(f"📝 Good effort! Score: {score}/{total} ({percentage:.1f}%)")
                    else:
                        st.warning(f"📚 Keep practicing! Score: {score}/{total} ({percentage:.1f}%)")
                    
                    # Tabs for different views
                    tab1, tab2, tab3 = st.tabs(["🖼️ Annotated Sheet", "📋 Detailed Results", "📈 Analysis"])
                    
                    with tab1:
                        st.subheader("Graded OMR Sheet")
                        st.image(annotated_img, channels="BGR", use_container_width=True)
                        st.caption("""
                        **Color Legend:**
                        🟢 Green = Correct | 🔴 Red = Incorrect | 🔵 Blue = Correct Answer | 🟡 Yellow = Double Mark
                        """)
                    
                    with tab2:
                        st.subheader("Question-wise Results")
                        df = pd.DataFrame(logs)
                        
                        # Add color coding
                        def highlight_status(row):
                            if row['Status'] == 'Correct':
                                return ['background-color: #d4edda'] * len(row)
                            elif row['Status'] == 'Incorrect':
                                return ['background-color: #f8d7da'] * len(row)
                            elif row['Status'] == 'Blank':
                                return ['background-color: #fff3cd'] * len(row)
                            else:  # Double Mark
                                return ['background-color: #f8d7da'] * len(row)
                        
                        styled_df = df.style.apply(highlight_status, axis=1)
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="omr_results.csv",
                            mime="text/csv"
                        )
                    
                    with tab3:
                        st.subheader("Performance Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Questions", total)
                            st.metric("Attempted", total - stats['blank'])
                            st.metric("Accuracy (of attempted)", 
                                     f"{(score / (total - stats['blank']) * 100):.1f}%" if (total - stats['blank']) > 0 else "N/A")
                        
                        with col2:
                            st.metric("Overall Score", f"{percentage:.1f}%")
                            st.metric("Questions Needing Review", stats['wrong'] + stats['blank'])
                            
                            # Grade calculation
                            if percentage >= 90:
                                grade = "A+"
                            elif percentage >= 80:
                                grade = "A"
                            elif percentage >= 70:
                                grade = "B"
                            elif percentage >= 60:
                                grade = "C"
                            else:
                                grade = "D"
                            st.metric("Grade", grade)
                        
                        # Wrong answers list
                        if stats['wrong'] > 0:
                            st.subheader("❌ Questions to Review")
                            wrong_df = df[df['Status'].isin(['Incorrect', 'Double Mark'])]
                            st.dataframe(wrong_df, use_container_width=True, hide_index=True)
                
                else:
                    st.error(f"❌ {message}")
                    st.image(annotated_img, channels="BGR", use_container_width=True)
                    st.info("💡 Tips: Ensure the image is clear, well-lit, and the OMR sheet is fully visible.")
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.exception(e)
                st.info("Please try uploading a different image or contact support.")


if __name__ == "__main__":
    main()
