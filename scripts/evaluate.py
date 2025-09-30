from ultralytics import YOLO

# Simple evaluation script
if __name__ == "__main__":
    model = YOLO("backend/models/best.pt")
    metrics = model.val()
    print("Evaluation metrics:", metrics)
