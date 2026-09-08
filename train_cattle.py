from ultralytics import YOLO
import os

# --------------------------
# 配置参数（需修改为你的实际路径）
# --------------------------
# 1. 训练好的最优模型路径（重点！用best.pt而非last.pt）
MODEL_PATH = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train39\weights\best.pt"
# 2. 数据集配置文件（与训练时相同的data.yaml）
DATA_YAML = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\ultralytics\cfg\datasets\data2.yaml"
# 3. 验证结果保存路径（会自动生成可视化文件）
VAL_SAVE_DIR = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train39\validation_results"

# --------------------------
# 执行验证
# --------------------------
if __name__ == "__main__":
    # 加载最优模型
    model = YOLO(MODEL_PATH)
    print(f"📌 已加载模型：{MODEL_PATH}")

    # 执行验证（参数说明见下方）
    val_results = model.val(
        data=DATA_YAML,  # 数据集配置文件（指定验证集路径）
        split="val",  # 验证集名称（对应data.yaml中的val字段）
        imgsz=600,  # 输入图像尺寸（与训练时一致）
        batch=16,  # 批次大小（显存不足可减小）
        device="0" if YOLO.device == "cuda" else "cpu",  # 自动选择设备
        save=True,  # 保存验证集的检测结果图像
        save_json=False,  # 如需COCO格式评估结果可设为True
        conf=0.25,  # 置信度阈值（过滤低置信度预测）
        iou=0.5,  # IOU阈值（用于NMS和评估）
        project=VAL_SAVE_DIR,  # 验证结果保存文件夹
        name="val_run",  # 本次验证名称
        plots=True  # 生成评估图表（PR曲线、混淆矩阵等）
    )

    # --------------------------
    # 打印关键评估指标
    # --------------------------
    print("\n📊 验证集核心指标：")
    print(f"  - 平均精度（mAP50）：{val_results.box.map50:.3f}（越高越好，目标>0.8）")
    print(f"  - 平均精度（mAP50-95）：{val_results.box.map:.3f}（越高越好）")
    print(f"  - 精确率（Precision）：{val_results.box.p:.3f}（预测为正例的准确率）")
    print(f"  - 召回率（Recall）：{val_results.box.r:.3f}（实际正例被预测出的比例）")

    # 提示结果保存位置
    print(f"\n💾 验证结果已保存至：{os.path.join(VAL_SAVE_DIR, 'val_run')}")
    print("  - 检测图像：在'val_run'文件夹的'val'子目录中")
    print("  - 评估图表：混淆矩阵（confusion_matrix.png）、PR曲线（PR_curve.png）等")
