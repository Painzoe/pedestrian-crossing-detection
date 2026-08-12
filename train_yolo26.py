"""
train_yolo26.py
------------------
train.py'nin YOLO26 versiyonu - AYNI mantik, AYNI veri (data.yaml,
part_1), tek fark baslangic modeli: yolov8x.pt yerine yolo26x.pt
(Ultralytics'in daha yeni nesil modeli, ayni "x" - en buyuk/dogru -
boyutu, adil kiyaslama icin).

train.py'ye (YOLOv8x) KESINLIKLE DOKUNULMADI - bu tamamen AYRI,
PARALEL bir dosya, mevcut yolov8x sonuclarini etkilemez.

KULLANIM:
  python train_yolo26.py [epoch_sayisi]
  ornek: python train_yolo26.py 25
  (belirtilmezse varsayilan 25 epoch)
"""

from ultralytics import YOLO
import torch
import os
import sys

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
BASE_MODEL = "yolo26x.pt"

# Komut satirindan epoch sayisi verilebilir - once 25, sonra 50 ile
# denemek icin script'i degistirmeden tekrar tekrar kullanabiliyoruz.
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25

# NOT: YOLOv8x'te oldugu gibi (train.py) imgsz=640 kullaniyoruz -
# 1280'de GPU bellegimiz (8GB) yetmiyordu, 640 YOLO'nun standart
# egitim cozunurlugu.
IMG_SIZE = 640

# batch=-1 -> Ultralytics'in "AutoBatch" ozelligi, GPU bellegine gore
# guvenli batch boyutunu kendisi hesaplar.
BATCH_SIZE = -1

RUN_NAME = f"part1_pilot_yolo26x_{EPOCHS}ep"
PROJECT_DIR = os.path.join(BASE_DIR, "outputs", "train_runs")


def main():
    # GPU KONTROLU - train.py'deki ayni mantik, sessizce CPU'da
    # calisip saatlerce beklemek istemiyoruz.
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (GPU) kullanilabilir mi: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("[UYARI] GPU bulunamadi, egitim CPU'da yapilacak - COK YAVAS olur.")
        device = "cpu"

    print(f"\nBaslangic modeli: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Egitim basliyor: {EPOCHS} epoch, imgsz={IMG_SIZE}, batch={BATCH_SIZE} (auto)\n")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
    )

    print("\nEgitim tamamlandi.")
    print(f"En iyi agirliklar: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
