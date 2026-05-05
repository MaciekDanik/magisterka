from ultralytics import YOLO, RTDETR
import shutil
import os

def download_model(model_name="rtdetr-resnet34"):
    # 🔥 RTDETR wymusza download poprawnie
    model = RTDETR(f"{model_name}.pt")

    src_path = model.ckpt_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dst_path = os.path.join(current_dir, f"{model_name}.pt")

    if os.path.exists(dst_path):
        print(f"[INFO] Model already exists: {dst_path}")
        return dst_path

    shutil.copy(src_path, dst_path)

    print(f"[OK] Model copied to: {dst_path}")
    return dst_path


if __name__ == "__main__":
    download_model("rtdetr-resnet50")