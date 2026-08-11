"""
prepare_dataset.py
--------------------
Label Studio'dan export edilen YOLO-format etiketleri (zip icinde
labels/*.txt, dosya adlari UUID onekli, ornek: "00177658-frame_0011.txt")
gercek goruntulerle (data/dataset_frames/<video>/images/) eslestirip
egitim (train) / dogrulama (validation) klasorlerine ayirir.

CIKTI: data/yolo_dataset/{train,val}/{images,labels}/ - ultralytics'in
YOLO egitimi icin bekledigi standart klasor yapisi (goruntu yolundaki
"images" kelimesini "labels" ile degistirerek etiket dosyasini otomatik
buluyor, bu yuzden klasor adlandirmasi onemli).

VERI SETI BUYUDUKCE (part_2, part_3... Label Studio'da etiketlendikce):
asagidaki LABELED_SOURCES listesine yeni bir {"zip":..., "images_dir":...}
girdisi eklemen yeterli - script hepsini birlestirip TEK bir train/val
ayrimina sokar (boylece her parca kendi icinde degil, butun veri seti
uzerinden karistirilip bolunmus olur).
"""

import os
import re
import random
import shutil
import zipfile

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

# Etiketlenmis (Label Studio export'u tamamlanmis) veri kaynaklari.
# Her biri: bir Label Studio YOLO export zip'i + o zip'teki etiketlerin
# hangi goruntu klasorune ait oldugu.
LABELED_SOURCES = [
    {
        "zip": os.path.join(BASE_DIR, "data", "labeled_export", "project-3-at-2026-08-10-13-00-e5cadc40.zip"),
        "images_dir": os.path.join(BASE_DIR, "data", "dataset_frames", "part_1", "images"),
    },
]

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")
VAL_RATIO = 0.20          # %80 train / %20 validation
RANDOM_SEED = 42          # ayni ayrimi tekrar uretebilmek icin sabit tohum

# UUID onekini ("00177658-frame_0011.txt" -> "frame_0011.txt") ayiklamak icin.
UUID_PREFIX_PATTERN = re.compile(r"^[0-9a-fA-F]+-")


def extract_labels_from_zip(zip_path):
    """Zip icindeki labels/*.txt dosyalarinin icerigini okur, UUID
    onekini kaldirip {frame_adi: dosya_icerigi} sozlugu olarak doner.
    Diske extract etmiyoruz - metin dosyalari kucuk oldugu icin
    dogrudan bellekte tutmak yeterli ve daha basit."""
    labels_by_frame = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("labels/") or not name.endswith(".txt"):
                continue
            raw_filename = os.path.basename(name)
            frame_basename = UUID_PREFIX_PATTERN.sub("", raw_filename)
            frame_basename = os.path.splitext(frame_basename)[0]
            with zf.open(name) as f:
                labels_by_frame[frame_basename] = f.read().decode("utf-8")
    return labels_by_frame


def collect_pairs(source):
    """Bir LABELED_SOURCES girdisi icin (goruntu_yolu, etiket_icerigi)
    ciftlerinin listesini olusturur. Zip'te etiketi olmayan goruntuler
    icin BOS etiket icerigi kullanilir (o karede kimse yok demektir -
    YOLO bunu 'negatif ornek' olarak ogrenir, es gecmekten daha dogru)."""
    labels_by_frame = extract_labels_from_zip(source["zip"])

    image_files = sorted([
        f for f in os.listdir(source["images_dir"])
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    pairs = []
    missing_label_count = 0
    for image_name in image_files:
        frame_basename = os.path.splitext(image_name)[0]
        label_content = labels_by_frame.get(frame_basename)
        if label_content is None:
            label_content = ""
            missing_label_count += 1
        pairs.append({
            "image_path": os.path.join(source["images_dir"], image_name),
            "image_name": image_name,
            "label_content": label_content,
        })

    print(f"  '{source['images_dir']}': {len(pairs)} goruntu, "
          f"{len(pairs) - missing_label_count} etiketli, "
          f"{missing_label_count} bos (kisi yok) etiket.")
    return pairs


def main():
    print("Etiket kaynaklari isleniyor...")
    all_pairs = []
    for source in LABELED_SOURCES:
        all_pairs.extend(collect_pairs(source))

    if len(all_pairs) == 0:
        raise RuntimeError("Hic goruntu/etiket cifti bulunamadi.")

    print(f"\nToplam {len(all_pairs)} goruntu/etiket cifti bulundu.")

    # Ayni tohumla her calistirmada AYNI ayrimi uretmek icin random.Random
    # kullaniyoruz (global random durumunu etkilemesin diye ayri instance).
    rng = random.Random(RANDOM_SEED)
    shuffled = all_pairs[:]
    rng.shuffle(shuffled)

    val_count = round(len(shuffled) * VAL_RATIO)
    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]

    print(f"Train: {len(train_pairs)} goruntu, Validation: {len(val_pairs)} goruntu "
          f"(hedef oran: %{int((1 - VAL_RATIO) * 100)}/%{int(VAL_RATIO * 100)})")

    # data/yolo_dataset TURETILMIS (kaynak degil) bir klasor - onceki
    # calistirmadan kalma varsa temizleyip yeniden olustururuz, boylece
    # eski/yeni ayrim karismaz. Orijinal goruntuler/etiketler (dataset_frames,
    # labeled_export) BURADA silinmiyor, sadece bu turetilmis kopya.
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        images_out = os.path.join(OUTPUT_DIR, split_name, "images")
        labels_out = os.path.join(OUTPUT_DIR, split_name, "labels")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        for pair in split_pairs:
            shutil.copy2(pair["image_path"], os.path.join(images_out, pair["image_name"]))
            label_filename = os.path.splitext(pair["image_name"])[0] + ".txt"
            with open(os.path.join(labels_out, label_filename), "w") as f:
                f.write(pair["label_content"])

    print(f"\nVeri seti hazir: {OUTPUT_DIR}")
    print(f"  {OUTPUT_DIR}/train/images, {OUTPUT_DIR}/train/labels")
    print(f"  {OUTPUT_DIR}/val/images,   {OUTPUT_DIR}/val/labels")


if __name__ == "__main__":
    main()
