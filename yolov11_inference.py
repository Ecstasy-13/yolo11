# from ultralytics import YOLO
# model = YOLO(r'G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train34\weights\best.pt')
# results = model(r'G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\datasets\coco128\images\val2017\img_3.png')  # 替换为你的测试图片路径
# results[0].show()  # 显示检测结果

import os
from glob import glob  # 用于筛选图片文件

from ultralytics import YOLO

# 1. 定义图片存放的文件夹路径（替换为你的实际路径）
image_dir = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\datasets\coco128\images\val2017"

# 2. 获取该路径下所有常见格式的图片文件（支持jpg、jpeg、png等）
# 筛选规则：扩展名为.jpg/.jpeg/.png/.bmp，不区分大小写
image_paths = (
    glob(os.path.join(image_dir, "*.jpg"))
    + glob(os.path.join(image_dir, "*.jpeg"))
    + glob(os.path.join(image_dir, "*.png"))
    + glob(os.path.join(image_dir, "*.bmp"))
)

# 3. 加载训练好的模型（替换为你的模型路径）
model = YOLO(r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train34\weights\best.pt")

# 4. 批量推理所有图片
results = model(image_paths)

# 5. 依次显示每张图片的检测结果
for i, result in enumerate(results):
    # 显示图片名称和检测结果（方便对应原图）
    print(f"正在显示第{i + 1}张图：{os.path.basename(image_paths[i])}")
    result.show()  # 弹出窗口显示检测结果
