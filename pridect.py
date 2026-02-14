from ultralytics import YOLO

if __name__ == '__main__':
    # 加载普通目标检测模型
    model = YOLO(r"D:\YOLO\runs\detect\train2\weights\best.pt")  # 假设您已经训练了检测模型
    
    # 目标检测预测
    results = model.predict(
        source=r"dataset\test\images",
        save=True,
        show=False,  # 是否显示图像
        conf=0.5,   # 置信度阈值
        iou=0.3,     # IOU阈值
        imgsz=320    # 推理尺寸
    )