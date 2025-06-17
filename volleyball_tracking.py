import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture("volleyball_match.mp4")
H, W = None, None
ball_trajectory = []
court_top = 50
court_bottom = 500
court_left = 50
court_right = 900

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    if H is None or W is None:
        H, W = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                box = det[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                if court_left < centerX < court_right and court_top < centerY < court_bottom:
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(conf))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    team1 = 0
    team2 = 0
    detected_players = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if label == "person":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = x + w // 2, y + h // 2
                if cy < H // 2:
                    team1 += 1
                else:
                    team2 += 1
                detected_players.append((x, y, w, h))

            if label == "sports ball":
                cx, cy = x + w // 2, y + h // 2
                ball_trajectory.append((cx, cy))
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

    for i in range(1, len(ball_trajectory)):
        cv2.line(frame, ball_trajectory[i - 1], ball_trajectory[i], (255, 0, 0), 2)

    cv2.putText(frame, f"Team 1 (Top): {team1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Team 2 (Bottom): {team2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Volleyball Tracking", frame)
    if cv2.waitKey(15) == 27:  
        break

cap.release()
cv2.destroyAllWindows()