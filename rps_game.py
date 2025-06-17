import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)
choices = ['rock', 'paper', 'scissors']
user_move = None
ai_move = None
result = ""
last_round_time = time.time()
round_duration = 3
def detect_move(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    if (index_tip.y < hand_landmarks.landmark[6].y and
        middle_tip.y < hand_landmarks.landmark[10].y and
        ring_tip.y > hand_landmarks.landmark[14].y and
        pinky_tip.y > hand_landmarks.landmark[18].y):
        return "scissors"

    elif (index_tip.y < hand_landmarks.landmark[6].y and
          middle_tip.y < hand_landmarks.landmark[10].y and
          ring_tip.y < hand_landmarks.landmark[14].y and
          pinky_tip.y < hand_landmarks.landmark[18].y):
        return "paper"

    else:
        return "rock"

def decide_winner(user, ai):
    if user == ai:
        return "Draw"
    elif (user == "rock" and ai == "scissors") or \
         (user == "scissors" and ai == "paper") or \
         (user == "paper" and ai == "rock"):
        return "You Win!"
    else:
        return "AI Wins!"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)

    current_time = time.time()
    countdown = round_duration - int(current_time - last_round_time)

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            user_move = detect_move(hand_landmarks)

    if current_time - last_round_time > round_duration:
        if user_move:
            ai_move = random.choice(choices)
            result = decide_winner(user_move, ai_move)
        else:
            result = "No hand detected"
            ai_move = random.choice(choices)

        last_round_time = current_time  

    cv2.putText(frame, f"Your move: {user_move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"AI move: {ai_move}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.putText(frame, f"{result}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(frame, f"Next round in: {countdown}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2)

    cv2.imshow("Rock Paper Scissors", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
