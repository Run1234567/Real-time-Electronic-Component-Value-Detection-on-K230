from ultralytics import YOLO

if __name__ == "__main__":
    # 使用YOLO12检测模型（普通目标检测）
    model = YOLO("yolo12m.pt")    # 小版本
    # model = YOLO("yolo12n.pt")  # 纳米版本
    # model = YOLO("yolo12m.pt")  # 中版本
    # model = YOLO("yolo12l.pt")  # 大版本
    # model = YOLO("yolo12x.pt")  # 超大版本
    
    model.train(
        data="data.yaml",
        imgsz=320,
        epochs=1000,
        batch=8,
        workers=4
    )