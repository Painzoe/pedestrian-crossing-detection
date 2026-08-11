"""
train_rfdetr.py
------------------
data/coco_dataset/ (prepare_dataset_rfdetr.py'nin urettigi COCO
formatindaki veri, YOLO icin kullandigimiz AYNI 240 train / 60
validation goruntuleri) uzerinde RF-DETR modelini fine-tune eder.

Mevcut YOLO pipeline'ina (train.py, detector.py, tracker.py) HIC
DOKUNMUYORUZ - bu tamamen PARALEL, ayri bir deneme. Amac: ayni veriyle
iki farkli model mimarisini (YOLOv8x - CNN tabanli vs RF-DETR -
transformer tabanli) kiyaslamak.

MODEL SECIMI (RFDETRMedium): rfdetr paketinde "RFDETRBase" ismi
ARTIK KULLANILMIYOR (deprecated, v2.0.0'da tamamen kaldirilacak) -
guncel/onerilen siniflar Nano/Small/Medium/Large. Orta seviye
(Medium) secildi - Nano/Small'dan daha guclu, Large'dan daha az GPU
bellegi ister (varsayilan cozunurluk 576px - YOLOv8x'in 1280px'te
GPU bellegimize (8GB) sigmadigini gordugumuz icin bu daha guvenli).

KULLANIM:
  python train_rfdetr.py
"""

from rfdetr import RFDETRMedium
import os

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

DATASET_DIR = os.path.join(BASE_DIR, "data", "coco_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models", "rfdetr_part1_finetune", "training")

# YOLO ile ADIL kiyaslama icin AYNI epoch sayisi (train.py'de EPOCHS = 25).
EPOCHS = 25

# batch_size="auto" -> RF-DETR'nin kendi "AutoBatch" ozelligi: GPU
# bellegine (8GB, RTX 4070 Laptop) gore guvenli batch boyutunu KENDISI
# hesaplar. YOLO egitiminde (train.py) sabit bir batch/cozunurlukle
# CUDA out of memory almistik - bunu tekrar yasamamak icin auto kullaniyoruz.
BATCH_SIZE = "auto"
GRAD_ACCUM_STEPS = 4  # RF-DETR'nin kendi varsayilani - batch_size ile
                       # carpilarak "etkin" batch boyutunu olusturur
LR = 1e-4              # RF-DETR'nin kendi varsayilan ogrenme orani


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("RFDETRMedium modeli yukleniyor (COCO on-egitimli baslangic agirliklari indirilecek)...")
    model = RFDETRMedium()

    print(f"\nEgitim basliyor: {EPOCHS} epoch, batch_size={BATCH_SIZE}, "
          f"grad_accum_steps={GRAD_ACCUM_STEPS}, lr={LR}\n")
    model.train(
        dataset_dir=DATASET_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        output_dir=OUTPUT_DIR,
    )

    print(f"\nEgitim tamamlandi. Ciktilar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
