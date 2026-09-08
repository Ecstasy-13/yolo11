from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n.pt")

    model.train(
        data=r"data3.yaml",
        epochs=10,
        imgsz=640,
        batch=2,
        degrees=2,
        cache=False,
        workers=1,
    )
