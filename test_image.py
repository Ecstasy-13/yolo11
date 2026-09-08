from ultralytics import YOLO
import os

# --------------------------
# 配置参数（需修改为你的实际路径）
# --------------------------
# 1. 训练好的最优模型路径（必须用best.pt）
MODEL_PATH = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train37\weights\best.pt"
# 2. 测试图片路径（替换为你的测试图片）
TEST_IMAGE_PATH = r"G:\dataset\Cattle_and_sheep_dataset\test_img\u=2397778498,2076219080&fm=253&fmt=auto&app=138&f=JPEG.webp"  # 例如：一张无人机拍摄的牛羊照片
# 3. 检测结果保存路径（会自动生成）
OUTPUT_DIR = r"G:\deeplearning\ultralytics-8.3.163\ultralytics-8.3.163\runs\detect\train37\test_results"

# --------------------------
# 执行单张图片检测
# --------------------------
if __name__ == "__main__":
    # 加载模型
    model = YOLO(MODEL_PATH)
    print(f"✅ 已加载模型：{MODEL_PATH}")
    print(f"📷 测试图片：{TEST_IMAGE_PATH}")

    # 执行预测（参数说明）
    # conf=0.3：只保留置信度>30%的检测结果（可调整，如0.2更宽松，0.5更严格）
    # iou=0.5：NMS阈值，过滤重叠框
    # save=True：自动保存带检测框的图片到OUTPUT_DIR
    results = model.predict(
        source=TEST_IMAGE_PATH,
        conf=0.3,
        iou=0.5,
        save=True,
        project=OUTPUT_DIR,
        name="single_image_test",
        show=False  # 若需要实时显示图片，设为True（需有GUI环境）
    )

    # --------------------------
    # 打印检测结果详情
    # --------------------------
    print("\n📊 检测结果：")
    # 遍历检测到的目标
    for result in results:
        for box in result.boxes:
            # 获取类别ID和名称
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            # 获取置信度（百分比）
            confidence = float(box.conf[0]) * 100
            # 获取边界框坐标（xyxy格式：左上角x, y，右下角x, y）
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  - 目标：{cls_name}，置信度：{confidence:.1f}%，位置：({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")

    # 提示结果保存位置
    output_image_path = os.path.join(OUTPUT_DIR, "single_image_test", os.path.basename(TEST_IMAGE_PATH))
    print(f"\n💾 带检测框的图片已保存至：{output_image_path}")
    print("  - 打开该图片即可查看模型识别效果（边界框+类别+置信度）")
