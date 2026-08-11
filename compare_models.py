"""
compare_models.py
-------------------
Fine-tune edilmis modelin (best.pt) orijinal (COCO uzerinde egitilmis,
hic fine-tune edilmemis) yolov8x.pt'ye gore GERCEKTEN daha iyi olup
olmadigini gosterir:

  Birkac validation goruntusunde (egitim sirasinda modelin HIC gormedigi
  60 goruntuden ornekler) her iki modelin tespitlerini ayri ayri cizip
  yan yana (solda orijinal, sagda fine-tune) tek bir resimde birlestirir,
  ayrica her ikisinin bulduğu kisi sayisini da yazdirir.

NOT: Orijinal yolov8x.pt COCO'nun 80 sinifiyla egitilmis, bizim veri
setimiz ise tek sinifli (person). Bu yuzden ikisini ayni "mAP" sayisiyla
resmi olarak kiyaslamak (val() ile) class-sayisi uyusmazligi yuzunden
kirilgan/yanlis yorumlanabilir sonuclar verebilir. Bunun yerine daha
basit ve guvenilir bir kiyas yapiyoruz: AYNI goruntude, AYNI conf/imgsz
ile, kac kisi tespit ettiler - goz ile ve sayiyla karsilastir.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import random

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

ORIGINAL_MODEL_PATH = "yolov8x.pt"
# NOT: train.py calistirildiginda "part1_pilot_yolov8x" klasoru zaten
# vardi (ilk, OOM ile cokmus denemeden kalma bos klasor) - Ultralytics
# bu yuzden otomatik olarak "-2" ekleyip YENI bir klasore kaydetti.
FINETUNED_MODEL_PATH = os.path.join(
    BASE_DIR, "outputs", "train_runs", "part1_pilot_yolov8x-2", "weights", "best.pt"
)

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset", "val", "images")

# Projede her yerde tutarli kullandigimiz inference ayarlari.
CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 1280
PERSON_CLASS_ID = 0

NUM_SAMPLE_IMAGES = 6   # yan yana gorsel karsilastirma icin kac goruntu
RANDOM_SEED = 7          # ayni ornekleri tekrar secebilmek icin sabit tohum

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model_comparison")


def draw_detections(model, image, label_text, label_color):
    """Bir goruntude tespit yapar, kutulari cizer, sol-ust kosede
    hangi model oldugunu ve kac kisi buldugunu yazar. (kutulu goruntu,
    kisi_sayisi) doner."""
    results = model(image, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
                     imgsz=IMG_SIZE, verbose=False)
    annotated = results[0].plot()
    count = len(results[0].boxes)

    cv2.putText(annotated, f"{label_text} - {count} kisi",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
    return annotated, count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FINETUNED_MODEL_PATH):
        raise FileNotFoundError(f"Fine-tune agirliklari bulunamadi: {FINETUNED_MODEL_PATH}")

    print(f"Orijinal model yukleniyor: {ORIGINAL_MODEL_PATH}")
    original_model = YOLO(ORIGINAL_MODEL_PATH)

    print(f"Fine-tune model yukleniyor: {FINETUNED_MODEL_PATH}")
    finetuned_model = YOLO(FINETUNED_MODEL_PATH)

    val_images = sorted([
        f for f in os.listdir(VAL_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if len(val_images) == 0:
        raise FileNotFoundError(f"'{VAL_IMAGES_DIR}' icinde goruntu bulunamadi.")

    rng = random.Random(RANDOM_SEED)
    sample_images = rng.sample(val_images, min(NUM_SAMPLE_IMAGES, len(val_images)))

    print(f"\n{len(sample_images)} validation goruntusunde karsilastirma yapiliyor "
          f"(bu goruntuler egitimde kullanilmadi)...\n")

    total_original = 0
    total_finetuned = 0

    for image_name in sample_images:
        image_path = os.path.join(VAL_IMAGES_DIR, image_name)
        image = cv2.imread(image_path)

        left, left_count = draw_detections(original_model, image, "ORIJINAL yolov8x", (0, 0, 255))
        right, right_count = draw_detections(finetuned_model, image, "FINE-TUNE", (0, 200, 0))

        combined = np.hstack([left, right])
        output_path = os.path.join(OUTPUT_DIR, f"compare_{image_name}")
        cv2.imwrite(output_path, combined)

        total_original += left_count
        total_finetuned += right_count

        fark = right_count - left_count
        fark_yazi = f"+{fark}" if fark > 0 else str(fark)
        print(f"  {image_name}: orijinal={left_count}, fine-tune={right_count} ({fark_yazi})")

    print(f"\nToplam (tum ornek goruntulerde): orijinal={total_original}, "
          f"fine-tune={total_finetuned}")
    print(f"Yan yana karsilastirma gorselleri: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
