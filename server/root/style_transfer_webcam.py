import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load pre-trained model
model = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')

# Define video capture function
def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        # Perform style transfer
        blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
        model.setInput(blob)
        output = model.forward()
        output = output.reshape((3, output.shape[2], output.shape[3]))
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.680
        output = output.transpose(1, 2, 0)
        # Encode image to JPEG
        _, jpeg = cv2.imencode('.jpg', output)
        # Yield JPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Define Flask route
@app.route('/')
def index():
    # Render index.html template
    return render_template('index.html')

# Define Flask route for video feed
@app.route('/video_feed')
def video_feed():
    # Return response with video stream generator
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
