import cv2
import mediapipe as mp
import pickle
import warnings
from collections import Counter, deque

# Suppress the scikit-learn feature name warning
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading model...")
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- STABILIZATION BUFFER SETUP ---
# Stores the last 15 predictions
prediction_buffer = deque(maxlen=15) 
# The stabilized sign we actually show on screen
current_sign = "-" 

cap = cv2.VideoCapture(0)
print("Starting stabilized live translation. Press 'q' to quit.")

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
            
            # Extract and normalize coordinates
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            row = []
            for landmark in hand_landmarks.landmark:
                norm_x = landmark.x - base_x
                norm_y = landmark.y - base_y
                norm_z = landmark.z - base_z
                row.extend([norm_x, norm_y, norm_z])
            
            # 1. Get raw prediction for this single frame
            raw_prediction = model.predict([row])[0]
            
            # 2. Add it to our rolling buffer
            prediction_buffer.append(raw_prediction)
            
            # 3. --- STABILIZATION LOGIC ---
            # Only update the displayed sign if the buffer is full
            if len(prediction_buffer) == prediction_buffer.maxlen:
                # Find the most common prediction in the last 15 frames
                most_common = Counter(prediction_buffer).most_common(1)[0]
                most_common_sign = most_common[0]
                most_common_count = most_common[1]
                
                # If at least 10 out of 15 frames agree, lock it in
                if most_common_count >= 10:
                    current_sign = most_common_sign

    # Draw UI using the stabilized 'current_sign' instead of the raw prediction
    cv2.rectangle(frame, (20, 20), (250, 85), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"Sign: {current_sign}", (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            
    cv2.imshow('Live Translator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()