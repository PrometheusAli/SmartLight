from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
lamp_brightness = 0.0  # 0–1 aralığında

def finger_open(lm, tip, pip):
    """Return True if finger is open (y coordinate smaller = higher)"""
    return lm[tip].y < lm[pip].y

def gen_frames():
    global lamp_brightness
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                fingers = [
                    finger_open(lm, 8, 6),
                    finger_open(lm, 12, 10),
                    finger_open(lm, 16, 14),
                    finger_open(lm, 20, 18)
                ]
                open_count = sum(fingers)

                # Parlaqlığı 0–1 aralığında hesabla
                lamp_brightness = min(open_count / 4.0, 1.0)

                cv2.putText(display, f"Brightness: {int(lamp_brightness*100)}%", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                color = (0, int(255*lamp_brightness), 0)
                cv2.rectangle(display, (0, 0), (w, h), color, 15)
        else:
            lamp_brightness = 0.0
            cv2.putText(display, "Brightness: 0%", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', display)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lamp_state')
def lamp_state():
    return {"brightness": lamp_brightness}

if __name__ == '__main__':
    app.run(debug=True)
