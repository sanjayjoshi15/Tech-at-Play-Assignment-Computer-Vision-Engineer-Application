# Tech-at-Play-Assignment-Computer-Vision-Engineer-Application
tech-at-play-assignment/
│
├── Task1_Detection_Summary_Engine.ipynb
├── Task2_Video_Summary_Reel.ipynb
├── Task3_Docker_Deployment/
│   ├── app/ 
│   │   ├── main.py
│   │   └── requirements.txt
│   └── Dockerfile
├── data/
│   └── sample_video.mp4
├── outputs/
│   ├── detection_summary.csv
│   ├── detection_frames/
│   ├── event_frames/
│   └── event_reel.mp4
└── README.md

✅ Task 1: Detection Summary Engine
# Task1_Detection_Summary_Engine.ipynb

import cv2
from ultralytics import YOLO
import pandas as pd, os

model = YOLO('yolov5s.pt')
cap = cv2.VideoCapture('data/sample_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
os.makedirs('outputs/detection_frames', exist_ok=True)

summary = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % 5 == 0:
        res = model.predict(frame)[0]
        for box in res.boxes:
            cls = int(box.cls.cpu())
            conf = float(box.conf.cpu())
            summary.append({
                'frame': frame_id,
                'timestamp_s': round(frame_id/fps,2),
                'label': model.model.names[cls],
                'confidence': round(conf,3)
            })
        cv2.imwrite(f"outputs/detection_frames/frame_{frame_id}.jpg", frame)
    frame_id += 1

cap.release()
pd.DataFrame(summary).to_csv('outputs/detection_summary.csv', index=False)

✅ Task 2: Video Summary Reel
# Task2_Video_Summary_Reel.ipynb

import cv2, os
from ultralytics import YOLO

model = YOLO('yolov5s.pt')
cap = cv2.VideoCapture('data/sample_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs('outputs/event_frames', exist_ok=True)
evt_frames = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % 5 == 0:
        res = model.predict(frame)[0]
        names = [model.model.names[int(b.cls.cpu())] for b in res.boxes]
        if any(c in ['person','car','bus'] for c in names):
            path = f"outputs/event_frames/event_{frame_id}.jpg"
            cv2.imwrite(path, frame)
            evt_frames.append(path)
    frame_id += 1

cap.release()

h, w = cv2.imread(evt_frames[0]).shape[:2]
out = cv2.VideoWriter('outputs/event_reel.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
for p in evt_frames:
    out.write(cv2.imread(p))
out.release()

🐳 Task 3: Docker Deployment
Task3_Docker_Deployment/app/main.py

# main.py
from ultralytics import YOLO
import uvicorn, os, cv2

app = FastAPI()

model = YOLO('yolov5s.pt')
UPLOAD = 'uploads'; os.makedirs(UPLOAD, exist_ok=True)

@app.post('/detect/')
async def detect(file: UploadFile = File(...)):
    path = f"{UPLOAD}/{file.filename}"
    with open(path, 'wb') as f: f.write(await file.read())
    cap = cv2.VideoCapture(path); ret, frame = cap.read(); cap.release()
    res = model.predict(frame)[0]
    out = [{'label': model.model.names[int(b.cls.cpu())], 'conf': float(b.conf.cpu())}
           for b in res.boxes]
    return {'results': out}
    
if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
