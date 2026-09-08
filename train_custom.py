from ultralytics import YOLO


def main():
    # 1. 加载你自定义的 yaml 文件，构建一个全新的随机初始化的模型
    model = YOLO("ultralytics/cfg/models/11/yolo11-LA-P2-WIoU.yaml")

    # 2. 开始训练
    model.train(
        data="sheep.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="",
        workers=4,  # 如果在 Windows 上报错，可以改为 0
    )


if __name__ == "__main__":
    main()
