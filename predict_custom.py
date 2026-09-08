from ultralytics import YOLO

def main():
    # 1. 加载你训练好的最优模型权重
    # ⚠️ 请根据你的实际训练日志，将 train47 替换为你最新/最好的那次训练文件夹
    # 权重文件通常保存在 runs/detect/trainXX/weights/best.pt
    weight_path = "G:/deeplearning/ultralytics-8.3.163/result/train6/weights/best.pt" 
    model = YOLO(weight_path)

    # 2. 指定你要测试的图片路径
    # 可以是一张包含伪装目标的测试集图片，也可以是你自己从网上下载的任何照片
    test_image = "G:/deeplearning/ultralytics-8.3.163/ultralytics-8.3.163/datasets/aerial-sheep-object-detection/images/train/img_2695_jpg.rf.28b116cdb2bf3f3af781936ce84a8072.jpg"

    # 3. 运行推理预测
    print(f"开始对图片 {test_image} 进行伪装目标检测...")
    results = model.predict(
        source=test_image, 
        conf=0.25,     # 置信度阈值：低于 25% 概率的框会被过滤掉
        iou=0.45,      # NMS 阈值：控制重叠框的合并程度
        save=True,     # 🌟 关键参数：将画好框的预测结果保存为新图片
        show=False,    # ⚠️ 服务器上通常没有显示器，设为 False 防止报错
        line_width=2   # 画框的线条粗细
    )

    # 4. 打印保存位置
    # YOLO 默认会将预测后的图片保存在 runs/detect/predict 文件夹下（每次预测会自动递增如 predict2, predict3）
    if results:
        print(f"✅ 测试完成！请前往 {results[0].save_dir} 文件夹查看画好框的图片。")

if __name__ == '__main__':
    main()