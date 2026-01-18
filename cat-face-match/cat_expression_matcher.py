import cv2
import numpy as np
import os
import urllib.request

# Try to use MediaPipe - handle both old and new API
USE_NEW_API = False
USE_OLD_API = False

try:
    import mediapipe as mp
    # Try new API first
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        USE_NEW_API = True
    except:
        USE_NEW_API = False
        # Try old API
        try:
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
            USE_OLD_API = True
        except:
            USE_OLD_API = False
            print("Error: Could not import MediaPipe. Please install: pip install mediapipe")
            exit(1)
except ImportError:
    print("Error: MediaPipe not installed. Please install: pip install mediapipe")
    exit(1)

# Download model if using new API
if USE_NEW_API:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download manually from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            exit(1)
    
    # Initialize new API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    face_mesh = None
elif USE_OLD_API:
    # Initialize old API
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_landmarker = None

# Load cat images
CAT_IMAGES = {
    'tongue_out': 'toung out.jpeg',
    'shocked': 'shocked.jpeg',
    'staring': 'staring.jpeg',
    'side_look': 'giving side look.jpeg'
}

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
cat_images_loaded = {}

# Load all cat images
for expression, filename in CAT_IMAGES.items():
    img_path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(img_path):
        cat_images_loaded[expression] = cv2.imread(img_path)
        print(f"Loaded: {filename}")
    else:
        print(f"Warning: {filename} not found!")

# Face mesh landmark indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
NOSE_TIP = 4

def calculate_mouth_openness(landmarks, frame_width, frame_height):
    """Calculate how open the mouth is"""
    # Get mouth corner landmarks
    mouth_left = landmarks[MOUTH_LEFT]
    mouth_right = landmarks[MOUTH_RIGHT]
    mouth_top = landmarks[MOUTH_TOP]
    mouth_bottom = landmarks[MOUTH_BOTTOM]
    
    # Calculate distances
    mouth_width = np.sqrt(
        (mouth_left.x - mouth_right.x)**2 * frame_width**2 +
        (mouth_left.y - mouth_right.y)**2 * frame_height**2
    )
    mouth_height = np.sqrt(
        (mouth_top.x - mouth_bottom.x)**2 * frame_width**2 +
        (mouth_top.y - mouth_bottom.y)**2 * frame_height**2
    )
    
    return mouth_height / mouth_width if mouth_width > 0 else 0

def calculate_eye_openness(landmarks, eye_top_idx, eye_bottom_idx, frame_height):
    """Calculate how open the eyes are"""
    eye_top = landmarks[eye_top_idx]
    eye_bottom = landmarks[eye_bottom_idx]
    eye_height = abs(eye_top.y - eye_bottom.y) * frame_height
    return eye_height

def detect_head_rotation(landmarks, frame_width, frame_height):
    """Detect if head is turned to the side"""
    # Use nose tip and face center
    nose_tip = landmarks[NOSE_TIP]
    face_center_x = frame_width / 2
    
    # Calculate offset from center
    nose_offset = (nose_tip.x * frame_width) - face_center_x
    offset_ratio = abs(nose_offset) / (frame_width / 2)
    
    return offset_ratio > 0.3  # Head turned if nose is significantly off-center

def detect_expression(landmarks, frame_width, frame_height):
    """Detect facial expression based on landmarks"""
    # Calculate mouth openness
    mouth_openness = calculate_mouth_openness(landmarks, frame_width, frame_height)
    
    # Calculate eye openness
    left_eye_open = calculate_eye_openness(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, frame_height)
    right_eye_open = calculate_eye_openness(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, frame_height)
    avg_eye_open = (left_eye_open + right_eye_open) / 2
    
    # Detect head rotation
    head_turned = detect_head_rotation(landmarks, frame_width, frame_height)
    
    # Expression detection logic
    # Tongue out: mouth very open (check first before shocked)
    if mouth_openness > 0.15:
        return 'tongue_out'
    
    # Shocked: eyes wide open (big eyes) - no mouth requirement
    elif avg_eye_open > 18:
        return 'shocked'
    
    # Side look: head turned to side
    elif head_turned:
        return 'side_look'
    
    # Staring: default (eyes focused, mouth closed)
    else:
        return 'staring'

# Start video capture
cap = cv2.VideoCapture(0)

# Set camera resolution for better detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

current_expression = None
last_expression = None
expression_frames = 0
FRAMES_THRESHOLD = 5  # Require 5 frames of same expression before switching
frame_timestamp_ms = 0

print("Starting camera...")
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with appropriate API
        if USE_NEW_API:
            # New API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 1
            
            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                detected_expression = detect_expression(face_landmarks, w, h)
            else:
                detected_expression = None
                current_expression = None
                expression_frames = 0
        elif USE_OLD_API:
            # Old API
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                detected_expression = detect_expression(face_landmarks, w, h)
            else:
                detected_expression = None
                current_expression = None
                expression_frames = 0
        
        # Update expression with frame consistency check
        if detected_expression:
            if detected_expression == last_expression:
                expression_frames += 1
                if expression_frames >= FRAMES_THRESHOLD:
                    current_expression = detected_expression
            else:
                expression_frames = 1
                last_expression = detected_expression
        else:
            expression_frames = 0
        
        # Display the matching cat image
        if current_expression and current_expression in cat_images_loaded:
            cat_img = cat_images_loaded[current_expression].copy()
            
            # Resize cat image to fit on screen (maintain aspect ratio)
            display_height = 300
            aspect_ratio = cat_img.shape[1] / cat_img.shape[0]
            display_width = int(display_height * aspect_ratio)
            cat_img_resized = cv2.resize(cat_img, (display_width, display_height))
            
            # Overlay cat image on top-right corner
            overlay_y = 10
            overlay_x = w - display_width - 10
            
            # Create a region of interest
            if overlay_y + display_height <= h and overlay_x >= 0:
                # Add semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (overlay_x - 5, overlay_y - 25), 
                            (overlay_x + display_width + 5, overlay_y + display_height + 5), 
                            (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Place cat image
                frame[overlay_y:overlay_y + display_height, 
                      overlay_x:overlay_x + display_width] = cat_img_resized
                
                # Add label
                expression_name = current_expression.replace('_', ' ').title()
                cv2.putText(frame, f"Matched: {expression_name}", 
                          (overlay_x, overlay_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw face landmarks (optional, for debugging)
        if USE_OLD_API and 'results' in locals() and hasattr(results, 'multi_face_landmarks') and results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                None,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
        elif USE_NEW_API and 'detection_result' in locals() and hasattr(detection_result, 'face_landmarks') and detection_result.face_landmarks:
            # Draw landmarks for new API
            for landmark in detection_result.face_landmarks[0]:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Add instruction text
        cv2.putText(frame, "Make expressions to match cat images!", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Cat Expression Matcher', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if face_landmarker:
        face_landmarker.close()
    if face_mesh:
        face_mesh.close()
    print("Camera released. Goodbye!")
