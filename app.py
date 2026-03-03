from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
model = YOLO("model/CrowdHuman.pt")


state = {
    "count": 0,
    "frame": 0,          
    "max_people": 0,
    "total_count_sum": 0,
    "avg_people": 0,
    "peak_frame": 0      
}

def generate_frames():
    global state
 
    cap = cv2.VideoCapture("video/MumbaiMetro_DemoCCTVClip_1.mp4")
    frame_id = 0
    
    while True:
        success, frame = cap.read()
        if not success: 
            break
        
       
        results = model(frame)
        count = len(results[0].boxes)
       
        frame_id += 1
        state["count"] = count
        state["frame"] = frame_id  
        
       
        if count > state["max_people"]:
            state["max_people"] = count
            state["peak_frame"] = frame_id 
            
        state["total_count_sum"] += count
        state["avg_people"] = round(state["total_count_sum"] / frame_id, 2)
        
     
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_data')
def live_data():
    return jsonify(state)

@app.route("/")
def dashboard():
   
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)