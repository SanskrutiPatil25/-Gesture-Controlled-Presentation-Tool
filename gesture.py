import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Variables
gesture_cooldown = 1.0  # seconds
last_gesture_time = 0
prev_palm_x = None
move_threshold = 80  # pixels for detecting slide
screen_width, screen_height = pyautogui.size()

# Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = img.shape
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            # Draw landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Get palm center (average of some key palm points)
            palm_points = [0, 1, 5, 9, 13, 17]  # Wrist + base of fingers
            palm_x = int(sum(lm_list[i][0] for i in palm_points) / len(palm_points))
            palm_y = int(sum(lm_list[i][1] for i in palm_points) / len(palm_points))

            # Draw palm center
            cv2.circle(img, (palm_x, palm_y), 10, (0, 0, 255), -1)

            # -------------------- 1. Palm Swipe for Slides --------------------
            if prev_palm_x is not None and time.time() - last_gesture_time > gesture_cooldown:
                if palm_x - prev_palm_x > move_threshold:
                    pyautogui.press('right')  # Next slide
                    last_gesture_time = time.time()
                elif prev_palm_x - palm_x > move_threshold:
                    pyautogui.press('left')  # Previous slide
                    last_gesture_time = time.time()

            prev_palm_x = palm_x

            # -------------------- 2. Pointer Mode (Index Finger) --------------------
            index_finger = lm_list[8]  # Index fingertip
            mouse_x = int(index_finger[0] / w * screen_width)
            mouse_y = int(index_finger[1] / h * screen_height)
            pyautogui.moveTo(mouse_x, mouse_y)

            # -------------------- 3. Thumb Pinch for Zoom --------------------
            thumb = lm_list[4]  # Thumb tip
            dist = get_distance(index_finger, thumb)

            if dist < 40:  # Pinch close -> Zoom in
                pyautogui.hotkey('ctrl', '+')
                time.sleep(0.2)
            elif dist > 150:  # Fingers far apart -> Zoom out
                pyautogui.hotkey('ctrl', '-')
                time.sleep(0.2)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
