import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure the Hands model.
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85)

# Open a connection to the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame color from BGR to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame and detect hands.
    results = hands.process(rgb_frame)

    # If hands are detected, loop through each hand's landmarks.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the original frame.
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
    
    # Display the annotated frame.
    cv2.imshow("MediaPipe Hands", frame)
    
    # Break loop when 'q' is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
hands.close()
