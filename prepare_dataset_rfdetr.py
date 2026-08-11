"""
prepare_dataset_rfdetr.py
---------------------------
data/yolo_dataset/{train,val} (YOLO formati, prepare_dataset.py'nin
urettigi - class_id x_center y_center width height satirlari) klasorlerini
RF-DETR'nin bekledigi COCO formatina (goruntu basina degil, TUM split
icin TEK bir _annotations.coco.json dosyasi) cevirir.

RF-DETR train/valid/test olmak uzere UC ayri klasor bekliyor. Bizim
veri setimiz kucuk (300 gorsel) oldugu icin AYRI bir 3'lu bolme
YAPMIYORUZ - YOLO icin kullandigimiz AYNI 240 train / 60 validation
ayrimini koruyoruz (birebir ayni goruntuler), "test" klasorunu de
validation ile AYNI 60 goruntu olacak sekilde uretiyoruz. Boylece
YOLO ve RF-DETR modellerini TAM OLARAK ayni goruntuler uzerinde
kiyaslayabiliyoruz - adil karsilastirma icin bu onemli (rastgele farkli
goruntulerde degerlendirilselerdi hangi modelin gercekten daha iyi
oldugunu soylemek zorlasirdi).

Donusumu kendimiz elle (COCO JSON formatini sifirdan yazarak) YAPMIYORUZ -
zaten kurulu olan supervision kutuphanesinin from_yolo() / as_coco()
fonksiyonlarini kullaniyoruz, bunlar test edilmis/guvenilir kutuphane
fonksiyonlari, hataya daha kapali.
"""

import os
import supervision as sv

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

YOLO_DATASET_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

COCO_DATASET_DIR = os.path.join(BASE_DIR, "data", "coco_dataset")


def convert_split(images_dir, labels_dir, output_images_dir, output_annotations_path):
    """Bir YOLO split'ini (goruntuler + .txt etiketler) supervision ile
    okuyup COCO formatinda (goruntuler + tek json dosyasi) disari yazar."""
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=images_dir,
        annotations_directory_path=labels_dir,
        data_yaml_path=DATA_YAML_PATH,
    )
    print(f"  {len(dataset)} goruntu okundu ({images_dir})")

    os.makedirs(output_images_dir, exist_ok=True)
    dataset.as_coco(
        images_directory_path=output_images_dir,
        annotations_path=output_annotations_path,
    )
    print(f"  COCO formatinda yazildi: {output_annotations_path}")


def main():
    print("Train split donusturuluyor...")
    convert_split(
        images_dir=os.path.join(YOLO_DATASET_DIR, "train", "images"),
        labels_dir=os.path.join(YOLO_DATASET_DIR, "train", "labels"),
        output_images_dir=os.path.join(COCO_DATASET_DIR, "train"),
        output_annotations_path=os.path.join(COCO_DATASET_DIR, "train", "_annotations.coco.json"),
    )

    print("\nValidation split donusturuluyor (RF-DETR'nin 'valid' klasoru)...")
    convert_split(
        images_dir=os.path.join(YOLO_DATASET_DIR, "val", "images"),
        labels_dir=os.path.join(YOLO_DATASET_DIR, "val", "labels"),
        output_images_dir=os.path.join(COCO_DATASET_DIR, "valid"),
        output_annotations_path=os.path.join(COCO_DATASET_DIR, "valid", "_annotations.coco.json"),
    )

    print("\n'test' split olusturuluyor (YOLO ile ADIL kiyaslama icin validation ile AYNI 60 goruntu)...")
    convert_split(
        images_dir=os.path.join(YOLO_DATASET_DIR, "val", "images"),
        labels_dir=os.path.join(YOLO_DATASET_DIR, "val", "labels"),
        output_images_dir=os.path.join(COCO_DATASET_DIR, "test"),
        output_annotations_path=os.path.join(COCO_DATASET_DIR, "test", "_annotations.coco.json"),
    )

    print(f"\nCOCO veri seti hazir: {COCO_DATASET_DIR}")


if __name__ == "__main__":
    main()
