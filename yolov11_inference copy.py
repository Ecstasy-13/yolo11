
import os
import random
from pathlib import Path

from ultralytics import YOLO


# =========================
# 1. 配置
# =========================

# 你的模型权重
MODEL_PATH = "G:/deeplearning/yolo_res/train19/weights/best.pt"

# 指定检测图片
IMAGE_PATH = "G:/deeplearning/ultralytics-8.3.163/ultralytics-8.3.163/datasets/aerial-sheep-object-detection/images/train/DJI_0005_0081_jpg.rf.c27000a5d01adcc7596cf3ecf4897819.jpg"

# 检测结果保存路径
SAVE_DIR = "G:/deeplearning/yolo_res/test_img"

# 输入尺寸
IMG_SIZE = 640

# 置信度阈值
CONF = 0.25


# =========================
# 2. 检查文件
# =========================

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"模型不存在: {MODEL_PATH}"
    )


if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(
        f"图片不存在: {IMAGE_PATH}"
    )


# =========================
# 3. 加载模型
# =========================

print("正在加载模型...")

model = YOLO(MODEL_PATH)

print("模型加载完成")


# =========================
# 4. 图片检测
# =========================

print("开始检测图片:")
print(IMAGE_PATH)


results = model.predict(
    source=IMAGE_PATH,
    imgsz=IMG_SIZE,
    conf=CONF,

    # 保存检测图片
    save=True,

    # 保存目录
    project=SAVE_DIR,
    name="result",

    exist_ok=True
)


# =========================
# 5. 输出检测结果
# =========================

result = results[0]


print("\n========== 检测结果 ==========")


if len(result.boxes) == 0:

    print("没有检测到目标")

else:

    for i, box in enumerate(result.boxes):

        # 类别编号
        cls_id = int(box.cls[0])

        # 类别名称
        cls_name = model.names[cls_id]

        # 置信度
        conf = float(box.conf[0])

        # 坐标
        xyxy = box.xyxy[0].tolist()

        x1, y1, x2, y2 = xyxy


        print(
            f"目标{i+1}: "
            f"{cls_name}, "
            f"置信度:{conf:.3f}, "
            f"位置:"
            f"({x1:.1f},{y1:.1f},"
            f"{x2:.1f},{y2:.1f})"
        )


# =========================
# 6. 统计数量
# =========================

count = {}

for cls in result.boxes.cls:

    cls_name = model.names[int(cls)]

    count[cls_name] = count.get(cls_name, 0) + 1


print("\n========== 数量统计 ==========")

if len(count)==0:

    print("无目标")

else:

    for k,v in count.items():

        print(
            f"{k}: {v}"
        )


# =========================
# 7. 输出路径
# =========================

save_path = os.path.join(
    SAVE_DIR,
    "result",
    Path(IMAGE_PATH).name
)


print("\n检测完成")
print("结果图片保存:")
print(save_path)