from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source=r"ultralytics\assets",
    save=True,
    show=False,
    # line_width=8,
    visualize=True,
)
