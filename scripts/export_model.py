from ultralytics import YOLO

# Export model to ONNX and TFLite
if __name__ == "__main__":
    model = YOLO("backend/models/best.pt")
    model.export(format="onnx")
    model.export(format="tflite")
    print("Models exported to ONNX and TFLite")
