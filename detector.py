"""
detector.py
------------
Bir VIDEO dosyasi uzerinde kare kare YOLO ile kisi tespiti yapar, her
karede tespit edilen kisilerin etrafina kutu cizip, TUM kareleri
OpenCV VideoWriter ile birlestirerek kutulu/izlenebilir bir MP4 video
dosyasi uretir.

ONCEKI VERSIYONDAN FARKI: Eskiden frames/ klasorundeki ayri ayri resim
dosyalarini okuyup, ayri ayri kutulu resimler + bir CSV ozet
uretiyordu. ARTIK dogrudan bir video dosyasi okuyup, dogrudan bir
video dosyasi yaziyoruz - resimlere bolme islemi SADECE Label Studio'ya
etiketleme hazirligi icin extract_frames.py'de kaliyor, tespit/tracking
adimlarinda artik kullanilmiyor.

Video dosya yolu SABIT KODLANMIS DEGIL - komut satirindan parametre
olarak veriliyor, boylece ayni script'i farkli videolarla
calistirabiliyoruz (orn. farklı Label Studio partlari, ham videolar).

KULLANIM:
  python detector.py <video_yolu> [--model MODEL_YOLU] [--output CIKTI_YOLU]
                      [--conf ESIK] [--imgsz BOYUT]

ORNEK (fine-tune edilmis modelle, part_1'in kaynak videosunda test):
  python detector.py data/raw_videos/part_1.mp4 \\
      --model outputs/train_runs/part1_pilot_yolov8x-2/weights/best.pt
"""

from ultralytics import YOLO
import cv2
import os
import argparse

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

# --- Komut satirinda belirtilmezse kullanilacak varsayilan degerler ---
DEFAULT_MODEL = "yolov8x.pt"
PERSON_CLASS_ID = 0
# NOT: 0.15, bu tarz dusuk cozunurluklu/tepeden acili/golgeli kameralar
# icin test edilerek dogrulanmis deger (roi.py'de de ayni deger kullaniliyor).
CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 1280


def parse_args():
    """Komut satirindan video yolunu ve istege bagli ayarlari okur.
    argparse kullaniyoruz ki script'i HER SEFERINDE FARKLI bir video/
    model ile calistirabilelim - kodun icini degistirmeden."""
    parser = argparse.ArgumentParser(
        description="Bir video uzerinde YOLO ile kisi tespiti yapip kutulu video uretir."
    )
    parser.add_argument("video_path", help="Girdi video dosyasinin yolu (orn. data/raw_videos/part_1.mp4)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"Kullanilacak YOLO agirlik dosyasi (varsayilan: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=None,
                         help="Cikti video yolu (belirtilmezse outputs/detected_videos/ altina otomatik uretilir)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                         help=f"Guven esigi (varsayilan: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE,
                         help=f"Tespit icin goruntu boyutu (varsayilan: {IMG_SIZE})")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video bulunamadi: {args.video_path}")

    # Cikti yolu belirtilmediyse: dosya adindan otomatik uret.
    if args.output:
        output_path = args.output
    else:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(BASE_DIR, "outputs", "detected_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_basename}_detected.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Model yukleniyor: {args.model}")
    model = YOLO(args.model)

    # ============================================================
    # GIRDI VIDEOYU AC, OZELLIKLERINI (fps, cozunurluk) OKU
    # ============================================================
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Bazi bozuk/eksik videolarda fps bilgisi okunamayabilir.
        fps = 30.0
        print("[UYARI] Video fps bilgisi okunamadi, 30 varsayiliyor.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Girdi video: {width}x{height}, {fps:.2f} fps, ~{total_frames} kare")

    # ============================================================
    # CIKTI VIDEO YAZICIYI HAZIRLA - GIRDI ILE AYNI fps/cozunurluk,
    # boylece cikti video DOGAL HIZINDA oynar (hizlandirilmis/
    # yavaslatilmis gorunmez).
    # ============================================================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # yaygin/uyumlu bir MP4 codec kodu
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ============================================================
    # KARE KARE TESPIT YAP, KUTULARI CIZ, CIKTI VIDEOYA YAZ
    # ============================================================
    frame_index = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:  # video bitti
            break

        results = model(frame, classes=[PERSON_CLASS_ID], conf=args.conf,
                         imgsz=args.imgsz, verbose=False)
        # result.plot() -> YOLO'nun kendi cizdigi kutu + etiket + guven skoru
        annotated_frame = results[0].plot()
        person_count = len(results[0].boxes)
        total_detections += person_count

        writer.write(annotated_frame)

        frame_index += 1
        # Her karede degil, 30 karede bir (ve son karede) ilerleme yazdir -
        # binlerce kareli bir videoda terminali bogmamak icin.
        if frame_index % 30 == 0 or frame_index == total_frames:
            print(f"  {frame_index}/{total_frames} kare islendi (bu karede {person_count} kisi)")

    cap.release()
    writer.release()

    print(f"\nTamamlandi: {frame_index} kare islendi.")
    print(f"Ortalama kisi/kare: {total_detections / max(frame_index, 1):.2f}")
    print(f"Cikti video: {output_path}")


if __name__ == "__main__":
    main()
