"""
train.py
---------
yolov8x.pt (COCO uzerinde egitilmis, "person" dahil 80 sinif) baslangic
noktasi alinarak, data.yaml'da tanimlanan kendi veri setimiz (part_1,
Label Studio'da elle etiketlenmis 294/300 kare) uzerinde fine-tune
(ince ayar) yapar.

Bu bir PILOT denemedir: sadece ~300 goruntu var, amac "bu yontem ise
yarar mi, veri seti buyudukce devam etmeli miyiz" sorusuna hizli bir
cevap almak - production-kalite bir model beklemiyoruz.

CIKTI: outputs/train_runs/<run_adi>/weights/best.pt - validation
kaybinin en dusuk oldugu epoch'un agirliklari. compare_models.py bunu
orijinal (fine-tune edilmemis) yolov8x.pt ile karsilastiracak.
"""

from ultralytics import YOLO
import torch
import os

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
BASE_MODEL = "yolov8x.pt"

EPOCHS = 50

# NOT: ilk denemede IMG_SIZE=1280 (detector.py/roi.py ile ayni) ile
# CUDA out of memory hatasi aldik - yolov8x (68M parametre) + 1280px,
# RTX 4070 Laptop'un 8GB bellegine (ozellikle epoch sonu validation
# adiminda) sigmiyordu. 640, YOLOv8'in STANDART/varsayilan egitim
# cozunurlugu - bellek ihtiyacini ~4'te 1'e indirir, bu kartta rahat
# calisir. Egitim cozunurlugunun inference (detector.py/roi.py, 1280)
# ile birebir ayni olmasi sart degil, model yine de iyi ogrenir.
IMG_SIZE = 640

# batch=-1 -> Ultralytics'in "AutoBatch" ozelligi: GPU bellegine (RTX 4070
# Laptop, 8GB) gore guvenli en buyuk batch boyutunu KENDISI hesaplar.
BATCH_SIZE = -1

RUN_NAME = "part1_pilot_yolov8x"
PROJECT_DIR = os.path.join(BASE_DIR, "outputs", "train_runs")


def main():
    # ============================================================
    # GPU KONTROLU - egitim GPU'da mi CPU'da mi calisacak, once dogrula.
    # (Onceki bir venv sorunundan sonra bunu her egitim oncesi acikca
    # yazdiriyoruz - "sessizce CPU'da calisiyor" durumunu fark etmeden
    # saatlerce beklemek istemiyoruz.)
    # ============================================================
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
