import cv2
import mediapipe as mp
import time
import pickle

# -----------------------------
# Load Model
# -----------------------------
with open("dynamic_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Settings
# -----------------------------
GESTURE_HOLD_TIME = 0.8
MIN_CONFIDENCE = 60

SEQ_LEN = 10  # IMPORTANT: matches training structure

# -----------------------------
# Sentence system
# -----------------------------
sentence = []
current_gesture = ""
gesture_start_time = 0
last_added_gesture = ""

sequence = []  # stores frames

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    frame_landmarks = []

    # -----------------------------
    # If hands detected
    # -----------------------------
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                frame_landmarks.append(lm.x)
                frame_landmarks.append(lm.y)
                frame_landmarks.append(lm.z)

    # -----------------------------
    # ALWAYS FIX SIZE = 126
    # -----------------------------
    while len(frame_landmarks) < 126:
        frame_landmarks.append(0)

    # add to sequence
    sequence.append(frame_landmarks)

    if len(sequence) > SEQ_LEN:
        sequence.pop(0)

    gesture = ""
    confidence = 0

    # -----------------------------
    # Predict only when sequence ready
    # -----------------------------
    if len(sequence) == SEQ_LEN:

        flat_input = []

        for f in sequence:
            flat_input.extend(f)

        # MUST be 1260
        if len(flat_input) != 1260:
            continue

        probs = model.predict_proba([flat_input])[0]
        pred = model.predict([flat_input])[0]

        gesture = pred
        confidence = max(probs) * 100

        current_time = time.time()

        # -----------------------------
        # Sentence logic
        # -----------------------------
        if confidence >= MIN_CONFIDENCE:

            if gesture != current_gesture:
                current_gesture = gesture
                gesture_start_time = current_time

            else:
                if current_time - gesture_start_time >= GESTURE_HOLD_TIME:

                    if gesture != last_added_gesture:
                        sentence.append(gesture)
                        last_added_gesture = gesture
                        print("Added:", gesture)

    # -----------------------------
    # UI TEXT (NO OVERLAP FIX)
    # -----------------------------
    cv2.putText(
        frame,
        f"{gesture} ({int(confidence)}%)",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    text = " ".join(sentence)

    cv2.putText(
        frame,
        text,
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("SilentBridge Dynamic", frame)

    key = cv2.waitKey(1)

    # Clear sentence
    if key == ord("c"):
        sentence.clear()
        current_gesture = ""
        last_added_gesture = ""
        print("Sentence cleared")

    # New line
    if key == 13:  # Enter key
        sentence.append("\n")
        print("New line")

    # Quit
    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()