import os
from ultralytics import YOLO

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    best_weight_path = "./runs/classify/runs/classify/textile_yolo11_fair/weights/best.pt"
    data_root = "./date-gray"

    print("==================================================")
    print("🚀 开始进行 YOLO11-cls test 集评估")
    print("==================================================")
    print("当前工作目录：", os.getcwd())
    print("权重路径：", os.path.abspath(best_weight_path))
    print("数据集路径：", os.path.abspath(data_root))
    print("==================================================")

    if not os.path.exists(best_weight_path):
        print(f"❌ 未找到最佳权重文件：{best_weight_path}")
        raise SystemExit(1)

    test_dir = os.path.join(data_root, "test")
    if not os.path.exists(test_dir):
        print(f"❌ 未找到 test 目录：{test_dir}")
        raise SystemExit(1)

    model = YOLO(best_weight_path)

    metrics = model.val(
        data=data_root,
        split="test",
        imgsz=128,
        batch=1,
        device=0,
        plots=True,
        verbose=True
    )

    print("\n==================================================")
    print("📊 YOLO11 在 test 数据集上的结果")
    print("==================================================")
    print(f"🥇 Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
    print(f"🏅 Top-5 Accuracy: {metrics.top5 * 100:.2f}%")
    print("==================================================")