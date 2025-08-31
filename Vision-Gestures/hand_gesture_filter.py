import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

# List of filter functions
def filter_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def filter_invert(frame):
    return cv2.bitwise_not(frame)

def filter_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

def filter_depth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    return colored

filters = [filter_bw, filter_invert, filter_thermal, filter_depth]
filter_names = ['Black & White', 'Invert', 'Thermal', 'Depth']
current_filter = 0
last_pinch_time = 0
pinch_threshold = 30  # pixels

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        left_hand_points = []
        right_hand_points = []
        pinch_detected = False
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
                # Pinch detection: distance between thumb and index
                pinch_dist = np.hypot(x_thumb - x_index, y_thumb - y_index)
                if pinch_dist < pinch_threshold:
                    pinch_detected = True
                if label == 'Left':
                    left_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
                elif label == 'Right':
                    right_hand_points = [(x_thumb, y_thumb), (x_index, y_index)]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if left_hand_points and right_hand_points:
                # Order: left thumb, left index, right thumb, right index
                roi_points = left_hand_points + right_hand_points
                left_thumb, left_index, right_thumb, right_index = roi_points
                rect_points = [left_index, right_index, right_thumb, left_thumb]
                pts = np.array(rect_points, np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                # Apply current filter
                filtered = filters[current_filter](frame)
                mask3 = cv2.merge([mask, mask, mask]) // 255
                output = filtered * mask3 + frame * (1 - mask3)
                output = output.astype(np.uint8)
                # Show filter name
                cv2.putText(output, filter_names[current_filter], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Hand Gesture Filter', output)
                # Change filter if pinch detected
                if pinch_detected:
                    current_filter = (current_filter + 1) % len(filters)
                    # Debounce: wait a short moment
                    cv2.waitKey(300)
            else:
                cv2.imshow('Hand Gesture Filter', frame)
        else:
            cv2.imshow('Hand Gesture Filter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
