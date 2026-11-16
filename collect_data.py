import cv2
import mediapipe as mp
import csv
import os

# --- CONFIGURATION ---
GESTURE_LABEL = "F"  # Change this for each new sign you record!
CSV_FILE = "dataset.csv"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Create CSV and headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label']
        for i in range(21):
            headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(headers)

cap = cv2.VideoCapture(0)
print(f"Ready. Press 's' to save a frame for '{GESTURE_LABEL}', 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow('Data Collector', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Get the wrist coordinates to act as the origin
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z
                
                row = [GESTURE_LABEL]
                
                # 2. Normalize all points relative to the wrist
                for landmark in hand_landmarks.landmark:
                    norm_x = landmark.x - base_x
                    norm_y = landmark.y - base_y
                    norm_z = landmark.z - base_z
                    row.extend([norm_x, norm_y, norm_z])
                
                # 3. Save to CSV
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"Saved normalized frame for '{GESTURE_LABEL}'")
        else:
            print("No hand detected!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()