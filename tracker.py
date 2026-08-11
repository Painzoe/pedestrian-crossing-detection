"""
tracker.py
-----------
Bir VIDEO dosyasi uzerinde kare kare YOLO ile kisi tespiti yapar,
ardindan supervision kutuphanesinin ByteTrack algoritmasi ile her
kisiye KALICI bir ID atar (ayni kisi farkli karelerde ayni ID'yi
tasir), kutu+ID etiketlerini cizip TUM kareleri OpenCV VideoWriter ile
birlestirerek kutulu/izlenebilir bir MP4 video dosyasi uretir.

ONCEKI VERSIYONDAN FARKI: Eskiden frames/ klasorundeki SEYREK (saniyede
1 kare) ornekle nmis resimleri okuyup ayri ayri resim uretiyordu. ARTIK
dogrudan video okuyup video yaziyoruz - bunun onemli bir yan faydasi da
var: videonun DOGAL (yuksek) fps'i sayesinde ByteTrack'in "kareler
arasi kucuk hareket" varsayimi artik gercekten geçerli oluyor - eski
"seyrek ornekleme" sorunu (asagidaki 2.5 bolumundeki ozel eslestirme
mantiginin sebebi) ortadan kalkiyor. Yine de o ozel mantigi (asagida)
DEGISTIRMEDEN koruyoruz - sorun cozulmus olsa bile fazladan zarari yok.

KULLANIM:
  python tracker.py <video_yolu> [--model MODEL_YOLU] [--output CIKTI_YOLU]
                     [--conf ESIK] [--imgsz BOYUT]
"""

from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import os
import argparse

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

DEFAULT_MODEL = "yolov8x.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 1280

# Bir ID'nin "gercek" sayilmasi icin en az kac farkli karede
# gorulmus olmasi gerekiyor. 1 = hicbir filtre yok.
MIN_FRAMES_TO_KEEP = 2

IOU_MATCH_THRESHOLD = 0.3  # bir tespiti bir track'e baglamak icin gereken min ortusme


def compute_iou(box_a, box_b):
    """
    Iki kutunun (x1, y1, x2, y2) ne kadar ortustugunu 0-1 arasi bir
    sayiyla olcer. 1 = tam ust uste, 0 = hic kesismiyor.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def assign_ids_to_all_tracks(tracker, detections):
    """
    supervision'in tracker.update_with_detections() fonksiyonu YERINE
    kullaniyoruz. Ayni ic takip guncellemesini yapar, ama "en az 2
    ardisik kare" sartini uygulamadan, o an takip edilen HERKESE ID
    atar - gercek/gurultu ayrimini asagidaki MIN_FRAMES_TO_KEEP
    filtresine birakiyoruz (detayli aciklama icin bkz. dosyanin ust
    kismindaki NOT).
    """
    tensors = np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))
    tracker.update_with_tensors(tensors=tensors)
    all_tracks = tracker.tracked_tracks

    tracker_ids = np.full(len(detections), -1, dtype=int)
    used_track_indices = set()

    # Bu karedeki her tespiti, ByteTrack'in guncelledigi track kutularindan
    # en cok ortusenle eslestiriyoruz (basit/acgozlu bir eslestirme).
    for i_det, det_box in enumerate(detections.xyxy):
        best_iou = IOU_MATCH_THRESHOLD
        best_track_idx = -1
        for i_track, track in enumerate(all_tracks):
            if i_track in used_track_indices:
                continue
            iou = compute_iou(det_box, track.tlbr)
            if iou > best_iou:
                best_iou = iou
                best_track_idx = i_track
        if best_track_idx != -1:
            tracker_ids[i_det] = all_tracks[best_track_idx].internal_track_id
            used_track_indices.add(best_track_idx)

    detections.tracker_id = tracker_ids
    return detections[tracker_ids != -1]


def parse_args():
    """Komut satirindan video yolunu ve istege bagli ayarlari okur -
    boylece ayni script'i farkli videolarla calistirabiliyoruz."""
    parser = argparse.ArgumentParser(
        description="Bir video uzerinde YOLO+ByteTrack ile kisi takibi yapip kutulu/ID'li video uretir."
    )
    parser.add_argument("video_path", help="Girdi video dosyasinin yolu")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"Kullanilacak YOLO agirlik dosyasi (varsayilan: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=None,
                         help="Cikti video yolu (belirtilmezse outputs/tracked_videos/ altina otomatik uretilir)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                         help=f"Guven esigi (varsayilan: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE,
                         help=f"Tespit icin goruntu boyutu (varsayilan: {IMG_SIZE})")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video bulunamadi: {args.video_path}")

    if args.output:
        output_path = args.output
    else:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(BASE_DIR, "outputs", "tracked_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_basename}_tracked.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Model yukleniyor: {args.model}")
    model = YOLO(args.model)

    # ByteTrack nesnesini DONGU DISINDA bir kere olusturuyoruz - onceki
    # karelerin "hafizasini" kendi icinde tutar. track_activation_threshold
    # aciklamasi icin bkz. eski versiyondaki detayli not (asagida ozetle):
    # YOLO esigimizden (args.conf) 0.1 dusuk vererek, YOLO'nun gecirdigi
    # HER tespitin ilk goründugu anda track acabilmesini sagliyoruz.
    tracker = sv.ByteTrack(track_activation_threshold=args.conf - 0.1)
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

    # Her ID'nin kac farkli karede gorundugunu sayacagimiz sozluk.
    track_id_frame_counts = {}
    frame_index = 0

    # ============================================================
    # KARE KARE TESPIT + TAKIP CALISTIR, CIKTI VIDEOYA YAZ
    # ============================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[PERSON_CLASS_ID], conf=args.conf,
                         imgsz=args.imgsz, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        tracked_detections = assign_ids_to_all_tracks(tracker, detections)

        if tracked_detections.tracker_id is not None:
            for track_id in tracked_detections.tracker_id:
                track_id_frame_counts[track_id] = track_id_frame_counts.get(track_id, 0) + 1

        labels = [f"ID:{tid}" for tid in tracked_detections.tracker_id] \
            if tracked_detections.tracker_id is not None else []

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

        writer.write(annotated_frame)

        frame_index += 1
        # Her karede degil, 30 karede bir (ve son karede) ilerleme yazdir -
        # binlerce kareli bir videoda terminali bogmamak icin (eski
        # versiyon HER karede yazdiriyordu, resim klasoru icin sorun
        # degildi ama gercek videoda binlerce satir olurdu).
        if frame_index % 30 == 0 or frame_index == total_frames:
            print(f"  {frame_index}/{total_frames} kare islendi "
                  f"({len(tracked_detections)} kisi takip ediliyor)")

    cap.release()
    writer.release()

    # ============================================================
    # OZET: HANGI ID'LER "GUVENILIR", HANGILERI "SUPHELI"
    # ============================================================
    print("\n" + "=" * 50)
    print("TAKIP OZETI")
    print("=" * 50)

    reliable_ids = [tid for tid, c in track_id_frame_counts.items() if c >= MIN_FRAMES_TO_KEEP]
    suspicious_ids = [tid for tid, c in track_id_frame_counts.items() if c < MIN_FRAMES_TO_KEEP]

    print(f"Toplam farkli ID: {len(track_id_frame_counts)}")
    print(f"Guvenilir (>= {MIN_FRAMES_TO_KEEP} karede goruldu): {len(reliable_ids)}")
    print(f"Supheli (< {MIN_FRAMES_TO_KEEP} karede goruldu): {len(suspicious_ids)}")
    print(f"\nCikti video: {output_path}")


if __name__ == "__main__":
    main()
