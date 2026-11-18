from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

# 使用当前最空闲的一块 GPU 进行训练
results = model.train(data="config/dataset_fall.yaml", epochs=100, imgsz=640, device=-1)