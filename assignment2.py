import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils  

def is_finger_extended(tip, pip, dip):
    return tip[1] < dip[1] and dip[1] < pip[1]  

def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    index_tip, index_dip, index_pip = (lm[8].x, lm[8].y), (lm[7].x, lm[7].y), (lm[6].x, lm[6].y)
    middle_tip, middle_dip, middle_pip = (lm[12].x, lm[12].y), (lm[11].x, lm[11].y), (lm[10].x, lm[10].y)
    ring_tip, ring_dip, ring_pip = (lm[16].x, lm[16].y), (lm[15].x, lm[15].y), (lm[14].x, lm[14].y)
    pinky_tip, pinky_dip, pinky_pip = (lm[20].x, lm[20].y), (lm[19].x, lm[19].y), (lm[18].x, lm[18].y)
    thumb_tip, thumb_ip, thumb_mcp = (lm[4].x, lm[4].y), (lm[3].x, lm[3].y), (lm[2].x, lm[2].y)

    index_extended = is_finger_extended(index_tip, index_pip, index_dip)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_dip)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_dip)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_dip)

    if index_extended and middle_extended and ring_extended and pinky_extended:
        return "Paper"
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Scissors"
    elif index_extended and not middle_extended and not ring_extended and pinky_extended:
        return "Rock"
    elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Fist"
    elif thumb_tip[1] < index_tip[1] and thumb_tip[1] < middle_tip[1]: 
        return "Thumbs Up"
    elif thumb_tip[1] > index_tip[1] and thumb_tip[1] > middle_tip[1]: 
        return "Thumbs Down"
    return "Unknown"

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) 

print("Press 'q' to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame. Exiting...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = hands.process(frame_rgb)  

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = classify_gesture(hand_landmarks)

            cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
