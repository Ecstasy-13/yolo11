import os

import numpy as np  # 新增：用于处理数组（计算平均值、遍历类别）

from ultralytics import YOLO

# --------------------------
# 配置参数（保持你的实际路径）
# --------------------------
MODEL_PATH = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train41\weights\best.pt"
DATA_YAML = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\ultralytics\cfg\datasets\data3.yaml"
VAL_SAVE_DIR = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train41\validation_results"

if __name__ == "__main__":
    # 加载最优模型
    model = YOLO(MODEL_PATH)
    print(f"📌 已加载模型：{MODEL_PATH}")
    print(f"🔧 当前设备：{model.device.type}（CPU/GPU）")  # 显示当前使用的设备

    # 执行验证（修正设备判断逻辑：用model实例的device，而非YOLO类）
    val_results = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        batch=16,
        device="0" if model.device.type == "cuda" else "cpu",  # 修正：model.device.type才是实际设备
        save=True,
        save_json=False,
        conf=0.25,
        iou=0.5,
        project=VAL_SAVE_DIR,
        name="val_run",
        plots=True,
    )

    # --------------------------
    # 打印关键评估指标（核心修正部分）
    # --------------------------
    # 1. 获取类别名称（如 ['cattle', 'sheep']，与训练时的data.yaml对应）
    class_names = model.names
    # 2. 提取精确率和召回率数组（每个元素对应一个类别的指标）
    class_precisions = val_results.box.p  # 精确率数组（长度=类别数）
    class_recalls = val_results.box.r  # 召回率数组（长度=类别数）

    print("\n📊 验证集核心指标：")
    print(f"  - 平均精度（mAP50）：{val_results.box.map50:.3f}（越高越好，目标>0.8）")
    print(f"  - 平均精度（mAP50-95）：{val_results.box.map:.3f}（越高越好）")

    # 3. 打印每个类别的精确率（解决TypeError的关键）
    print("\n  - 各类别精确率（Precision）：")
    for cls_id, precision in enumerate(class_precisions):
        cls_name = class_names[cls_id]  # 获取类别名称（如“cattle”“sheep”）
        print(f"    类别 {cls_id}（{cls_name}）：{precision:.3f}")
    # 打印所有类别的平均精确率（与验证日志中的“all”类P值一致）
    mean_precision = np.mean(class_precisions)
    print(f"    所有类别平均精确率：{mean_precision:.3f}")

    # 4. 打印每个类别的召回率（同理处理数组）
    print("\n  - 各类别召回率（Recall）：")
    for cls_id, recall in enumerate(class_recalls):
        cls_name = class_names[cls_id]
        print(f"    类别 {cls_id}（{cls_name}）：{recall:.3f}")
    # 打印所有类别的平均召回率（与验证日志中的“all”类R值一致）
    mean_recall = np.mean(class_recalls)
    print(f"    所有类别平均召回率：{mean_recall:.3f}")

    # 提示结果保存位置
    output_dir = os.path.join(VAL_SAVE_DIR, "val_run")
    print(f"\n💾 验证结果已保存至：{output_dir}")
    print("  - 检测图像：在 {output_dir}/val 目录中（带边界框的验证集图片）")
    print("  - 评估图表：confusion_matrix.png（混淆矩阵）、PR_curve.png（精确率-召回率曲线）")
