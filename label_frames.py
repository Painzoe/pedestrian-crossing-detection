"""
label_frames.py
-----------------
frames/ klasorundeki TUM karelerde YOLO ile kisi tespiti yapip kutulu
goruntuleri diske kaydeder. ROI/sayim KARSILASTIRMASI yok - sadece
"bu model bu kamerada neyi kisi olarak isaretliyor" sorusuna gorsel
cevap.

Ayni script'i IKI FARKLI model icin (asagidaki MODELS listesi)
calistiriyoruz, her biri KENDI klasorune kaydediliyor:
  1) orijinal yolov8x.pt        -> outputs/frames_labeled_original/
  2) fine-tune edilmis best.pt  -> outputs/frames_labeled_finetuned/

Boylece fine-tune modelin, egitimde HIC GORMEDIGI bu kamerada (frames/,
roi.py'nin kullandigi dusuk cozunurluklu/tepeden acili kamera) nasil
davrandigini, orijinal modelle klasor klasor karsilastirip goz ile
gorebiliyoruz.
"""

from ultralytics import YOLO
import cv2
import os

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

# detector.py/roi.py ile TUTARLI tespit ayarlari.
CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 1280
PERSON_CLASS_ID = 0

# Etiketleyecegimiz modeller ve her birinin ciktisinin kaydedilecegi klasor.
MODELS = [
    {
        "name": "orijinal",
        "weights": "yolov8x.pt",
        "output_dir": os.path.join(BASE_DIR, "outputs", "frames_labeled_original"),
    },
    {
        "name": "fine-tune",
        "weights": os.path.join(
            BASE_DIR, "outputs", "train_runs", "part1_pilot_yolov8x-2", "weights", "best.pt"
        ),
        "output_dir": os.path.join(BASE_DIR, "outputs", "frames_labeled_finetuned"),
    },
]


def label_frames_with_model(model_info, frame_files):
    print(f"\nModel yukleniyor ({model_info['name']}): {model_info['weights']}")
    model = YOLO(model_info["weights"])

    os.makedirs(model_info["output_dir"], exist_ok=True)

    for frame_name in frame_files:
        image = cv2.imread(os.path.join(FRAMES_DIR, frame_name))
        if image is None:
            print(f"  [UYARI] '{frame_name}' okunamadi, atlaniyor.")
            continue

        results = model(image, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
                         imgsz=IMG_SIZE, verbose=False)
        # result.plot() -> YOLO'nun kendi cizdigi kutu + etiket + guven
        # skoru, detector.py'de kullandigimiz ayni yontem.
        annotated = results[0].plot()

        output_path = os.path.join(model_info["output_dir"], frame_name)
        cv2.imwrite(output_path, annotated)

    print(f"  -> {len(frame_files)} kare etiketlendi: {model_info['output_dir']}")


def main():
    frame_files = sorted([
        f for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if len(frame_files) == 0:
        raise FileNotFoundError(f"'{FRAMES_DIR}' icinde goruntu bulunamadi.")

    print(f"Toplam {len(frame_files)} frame bulundu: {FRAMES_DIR}")

    for model_info in MODELS:
        label_frames_with_model(model_info, frame_files)

    print("\nTamamlandi. Karsilastirmak icin:")
    for model_info in MODELS:
        print(f"  {model_info['name']}: {model_info['output_dir']}")


if __name__ == "__main__":
    main()
