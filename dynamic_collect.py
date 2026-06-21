import cv2
import mediapipe as mp
import csv

# -----------------------------
# Gesture Name
# -----------------------------

gesture_name = input("Enter dynamic gesture name: ")

# -----------------------------
# CSV File
# -----------------------------

file = open("dynamic_dataset.csv", "a", newline="")
writer = csv.writer(file)

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

mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Variables
# -----------------------------

sample_count = 0

sequence = []

SEQUENCE_LENGTH = 10

# -----------------------------
# Main Loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        all_landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for landmark in hand_landmarks.landmark:

                all_landmarks.append(landmark.x)
                all_landmarks.append(landmark.y)
                all_landmarks.append(landmark.z)

        # Always 126 features
        while len(all_landmarks) < 126:
            all_landmarks.append(0)

        sequence.extend(all_landmarks)

        if len(sequence) == (126 * SEQUENCE_LENGTH):

            row = sequence.copy()

            row.append(gesture_name)

            writer.writerow(row)

            sample_count += 1

            print(f"Samples Collected: {sample_count}")

            sequence = []

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dynamic Data Collection", frame)

    key = cv2.waitKey(1)

    if sample_count >= 118:
        print("118 samples collected")
        break

    if key == ord('q') or key == 27:
        break

file.close()
cap.release()
cv2.destroyAllWindows()