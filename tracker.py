import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 2. Open the default webcam
cap = cv2.VideoCapture(0)

print("Starting tracking loop. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Process the frame
    results = hands.process(rgb_frame)

    # 4. Draw landmarks if a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4), # Joints
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), # Bones
            )
        
    # Show the live feed
    cv2.imshow('Hand Tracker', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Clean up
cap.release()
cv2.destroyAllWindows()