import os
from ultralytics import YOLO


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("🚀 开始训练 YOLO11-cls 织物分类模型...")

    # 1. 加载官方预训练分类模型
    model = YOLO("yolo11n-cls.pt")

    # 2. 训练
    results = model.train(
        data="./date-gray",              # 分类数据集根目录，需包含 train/val 子目录
        epochs=50,
        imgsz=128,                       # 输入尺寸统一到 128x128
        batch=8,
        device=0,
        workers=2,

        project="runs/classify",
        name="textile_yolo11_fair",
        exist_ok=True,

        pretrained=True,
        optimizer="AdamW",
        lr0=1e-4,
        dropout=0.0,

        # 为了和你的灰度纹理任务更公平，先关闭颜色增强
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # 分类任务里更适合收紧几何增强
        scale=0.0,
        fliplr=0.0,
        flipud=0.0,

        # 分类任务默认还会有 auto_augment 和 erasing，先关掉做公平对比
        auto_augment=None,
        erasing=0.0,

        # 常规训练选项
        val=True,
        plots=True,
        save=True,
        seed=0,
        deterministic=True,
        verbose=True
    )

    print("\n🏁 YOLO11-cls 训练完成！")
    print("最佳权重一般保存在：./runs/classify/textile_yolo11_fair/weights/best.pt")