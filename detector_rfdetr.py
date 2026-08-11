"""
detector_rfdetr.py
--------------------
detector.py'nin RF-DETR versiyonu - AYNI mantik (video oku, kare kare
tespit yap, kutulari ciz, video yaz), ama YOLO/ultralytics yerine
RF-DETR (rfdetr paketi) kullanir.

detector.py'ye (YOLO versiyonu) KESINLIKLE DOKUNULMADI - bu tamamen
AYRI, PARALEL bir dosya. Amac: ayni videoda iki farkli model mimarisini
(YOLOv8x vs RF-DETR) gozle kiyaslayabilmek.

ONEMLI FARKLAR (YOLO'ya gore):
- RF-DETR goruntuleri RGB sirasinda bekliyor, ama OpenCV goruntuleri
  BGR sirasinda okur - bu yuzden her karede once BGR->RGB cevrimi
  yapiyoruz (yoksa renkler ters olur, model dogru calismaz).
- RF-DETR sonucu dogrudan bir supervision Detections nesnesi doner
  (YOLO'daki gibi kendi .plot() metodu yok) - kutulari cizmek icin
  supervision'in BoxAnnotator/LabelAnnotator'larini kullaniyoruz
  (tracker.py'de zaten kullandigimiz ayni araclar).

KULLANIM:
  python detector_rfdetr.py <video_yolu> [--model MODEL_YOLU] [--output CIKTI_YOLU]
                             [--conf ESIK]

ORNEK:
  python detector_rfdetr.py data/raw_videos/part_1.mp4 \\
      --model outputs/models/rfdetr_part1_finetune/weights/best.pth
"""

from rfdetr import RFDETRMedium
import supervision as sv
import cv2
import os
import argparse

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

DEFAULT_MODEL = os.path.join(BASE_DIR, "outputs", "models", "rfdetr_part1_finetune", "weights", "best.pth")
# NOT: YOLO'da 0.15 kullaniyorduk (o kamera icin test edilerek
# dogrulanmisti) ama RF-DETR'nin guven skoru YOLO ile AYNI olcekte
# DEGIL - bunu 0.15 ile test edince Video1.mp4'te bircok karede
# insanla ilgisi olmayan (direk, tabela, golge) yanlis kutular
# cikti; 0.5'e cikarinca kayboldular. rfdetr kutuphanesinin kendi
# predict() fonksiyonunun VARSAYILANI da zaten 0.5 - YOLO'dan
# kopyalamak yerine bunu kullanmaliydik. 0.4-0.6 araligi test edildi,
# sonuc kararli (ayni kutular), 0.5'te sabitledik.
CONFIDENCE_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bir video uzerinde RF-DETR ile kisi tespiti yapip kutulu video uretir."
    )
    parser.add_argument("video_path", help="Girdi video dosyasinin yolu")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"RF-DETR agirlik dosyasi (.pth) (varsayilan: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=None,
                         help="Cikti video yolu (belirtilmezse outputs/detected_videos_rfdetr/ altina otomatik uretilir)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                         help=f"Guven esigi (varsayilan: {CONFIDENCE_THRESHOLD})")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video bulunamadi: {args.video_path}")

    if args.output:
        output_path = args.output
    else:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(BASE_DIR, "outputs", "detected_videos_rfdetr")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_basename}_detected_rfdetr.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Model yukleniyor: {args.model}")
    # trust_checkpoint=True: kendi egittigimiz (Roboflow'un resmi
    # dagitmadigi) bir checkpoint oldugu icin gerekli.
    model = RFDETRMedium(pretrain_weights=args.model, trust_checkpoint=True)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # ============================================================
    # GIRDI VIDEOYU AC, OZELLIKLERINI OKU
    # ============================================================
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print("[UYARI] Video fps bilgisi okunamadi, 30 varsayiliyor.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Girdi video: {width}x{height}, {fps:.2f} fps, ~{total_frames} kare")

    # ============================================================
    # CIKTI VIDEO YAZICIYI HAZIRLA - GIRDI ILE AYNI fps/cozunurluk
    # ============================================================
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ============================================================
    # KARE KARE TESPIT YAP, KUTULARI CIZ, CIKTI VIDEOYA YAZ
    # ============================================================
    frame_index = 0
    total_detections = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # RF-DETR RGB bekliyor, OpenCV BGR okuyor - cevirmezsek renkler
        # ters olur ve model yanlis/eksik tespit yapar.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = model.predict(frame_rgb, threshold=args.conf)
        person_count = len(detections)
        total_detections += person_count

        labels = [f"person {conf:.2f}" for conf in detections.confidence]
        annotated_frame = box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        writer.write(annotated_frame)

        frame_index += 1
        if frame_index % 30 == 0 or frame_index == total_frames:
            print(f"  {frame_index}/{total_frames} kare islendi (bu karede {person_count} kisi)")

    cap.release()
    writer.release()

    print(f"\nTamamlandi: {frame_index} kare islendi.")
    print(f"Ortalama kisi/kare: {total_detections / max(frame_index, 1):.2f}")
    print(f"Cikti video: {output_path}")


if __name__ == "__main__":
    main()
