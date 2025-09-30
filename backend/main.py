from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, base64

app = FastAPI(title="Pothole Detection API")
model = YOLO("models/best.pt")  # your trained model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_b64(img: Image.Image):
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model.predict(source=img, imgsz=640, conf=0.25, iou=0.45)
    r = results[0]

    detections = []
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1,y1,x2,y2 = map(float, box)
        detections.append({
            "bbox": [x1,y1,x2,y2],
            "class": int(cls),
            "confidence": float(conf)
        })

    annotated = r.plot()  # numpy array
    annotated_img = Image.fromarray(annotated)

    return {
        "detections": detections,
        "image_b64": pil_to_b64(annotated_img)
    }
