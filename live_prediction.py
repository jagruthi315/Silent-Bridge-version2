import cv2
import mediapipe as mp
import time

from predict_gesture import predict

# -----------------------------
# Settings
# -----------------------------

GESTURE_HOLD_TIME = 0.8  # seconds
MIN_CONFIDENCE = 59  # %

# -----------------------------
# Sentence Builder Variables
# -----------------------------

sentence = []
current_gesture = ""
gesture_start_time = 0
last_added_gesture = ""

# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

# -----------------------------
# MediaPipe Hands
# -----------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
     max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Drawing Utility
# -----------------------------

mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Main Loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 126 features
           
            row = []

            for landmark in hand_landmarks.landmark:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            # Pad to 126 features
            while len(row) < 126:
                row.append(0)

            # Predict gesture
            gesture, confidence, probabilities, classes = predict(row)

            current_time = time.time()

            # Sentence Builder Logic
            if confidence >= MIN_CONFIDENCE:

                if gesture != current_gesture:
                    current_gesture = gesture
                    gesture_start_time = current_time

                else:
                    if current_time - gesture_start_time >= GESTURE_HOLD_TIME:
                        if gesture != last_added_gesture:
                            sentence.append(gesture)
                            print("Added:", gesture)
                            last_added_gesture = gesture

            # Display current gesture
            cv2.putText(
                frame,
                f"{gesture} ({confidence}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Display built sentence
    lines = []
    current_line = ""

    for word in sentence:
        if word == "\n":
            lines.append(current_line)
            current_line = ""
        else:
            current_line += word + " "

    lines.append(current_line)

    y = 100

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
        y += 35

    # Show frame
    cv2.imshow("SilentBridge", frame)

    key = cv2.waitKey(1)

    # Clear sentence
    if key == ord('c'):
        sentence.clear()
        current_gesture = ""
        last_added_gesture = ""
        print("Sentence cleared")

    # New line
    if key == 13:  # Enter key
        sentence.append("\n")
        print("New line")

    # Quit
    if key == ord('q') or key == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()