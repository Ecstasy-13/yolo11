from ultralytics import YOLO
import os
import numpy as np  # 新增：用于数组计算

# --------------------------
# 配置参数（需修改为你的实际路径）
# --------------------------
MODEL_PATH = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train39\weights\best.pt"
DATA_YAML = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\ultralytics\cfg\datasets\data2.yaml"
VAL_SAVE_DIR = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train39\validation_results"

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    print(f"📌 已加载模型：{MODEL_PATH}")

    # 执行验证
    val_results = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=600,
        batch=16,
        device="0" if model.device.type == "cuda" else "cpu",  # 修正设备判断逻辑
        save=True,
        save_json=False,
        conf=0.25,
        iou=0.5,
        project=VAL_SAVE_DIR,
        name="val_run",
        plots=True
    )

    # --------------------------
    # 打印关键评估指标（修复部分）
    # --------------------------
    # 获取类别名称（如['cattle', 'sheep']）
    class_names = model.names
    # 精确率数组（每个类别）
    precisions = val_results.box.p
    # 召回率数组（每个类别）
    recalls = val_results.box.r

    print("\n📊 验证集核心指标：")
    print(f"  - 平均精度（mAP50）：{val_results.box.map50:.3f}（越高越好，目标>0.8）")
    print(f"  - 平均精度（mAP50-95）：{val_results.box.map:.3f}（越高越好）")

    # 打印每个类别的精确率
    print("\n  - 精确率（Precision）：")
    for cls_id, p in enumerate(precisions):
        print(f"    类别 {cls_id}（{class_names[cls_id]}）：{p:.3f}")
    # 打印平均精确率
    print(f"    平均精确率：{np.mean(precisions):.3f}")

    # 打印每个类别的召回率
    print("\n  - 召回率（Recall）：")
    for cls_id, r in enumerate(recalls):
        print(f"    类别 {cls_id}（{class_names[cls_id]}）：{r:.3f}")
    # 打印平均召回率
    print(f"    平均召回率：{np.mean(recalls):.3f}")

    print(f"\n💾 验证结果已保存至：{os.path.join(VAL_SAVE_DIR, 'val_run')}")
    print("  - 检测图像：在'val_run'文件夹的'val'子目录中")
    print("  - 评估图表：混淆矩阵（confusion_matrix.png）、PR曲线（PR_curve.png）等")
