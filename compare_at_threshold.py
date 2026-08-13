"""
compare_at_threshold.py
--------------------------
Iki farkli model ailesini (YOLO ve RF-DETR) AYNI validation setinde
(part_1'in 60 gorsellik, gercek insan etiketli dogrulama seti) AYNI
guven esiginde calistirip GERCEK dogruluk (Precision/Recall/F1)
hesaplar.

Bu, video testlerindeki "ortalama kisi/kare" gibi kaba bir sayidan
FARKLI - o sayida gercek etiket yoktu, sadece "kac kutu cizildi"
sayiyorduk. Burada her tespit, GERCEK insan etiketiyle (IoU>=0.5)
eslestiriliyor - yani gercekten "dogru mu yanlis mi" olculuyor.

KULLANIM:
  python compare_at_threshold.py <esik>
  ornek: python compare_at_threshold.py 0.3
"""

import sys
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from rfdetr import RFDETRMedium
import cv2

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "data/yolo_dataset/val/images")
VAL_LABELS_DIR = os.path.join(BASE_DIR, "data/yolo_dataset/val/labels")
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")

IOU_THRESHOLD = 0.5  # bir tespitin "dogru" sayilmasi icin gercek kutuyla en az bu kadar ortusmesi lazim
IMG_SIZE_YOLO = 1280  # detector.py'de kullandigimiz AYNI cozunurluk - tutarlilik icin

# Kiyaslanacak modeller - HER MODEL kendi "esik" degeriyle calisir, cunku
# guven skorlari mimariler arasi AYNI OLCEKTE DEGIL (bunu RF-DETR'de
# 0.15 vs 0.5 denerken oğrenmiştik). Komut satirindan tek bir esik
# verilirse (orn. "python compare_at_threshold.py 0.3") TUM modeller
# o esikle calisir - istege bagli, varsayilan olarak her modelin KENDI
# projede kullandigimiz esigi kullanilir.
CLI_THRESHOLD_OVERRIDE = float(sys.argv[1]) if len(sys.argv) > 1 else None

MODELS = {
    "YOLOv8x-25ep": {"type": "yolo", "threshold": 0.15, "weights": os.path.join(BASE_DIR, "outputs/models/yolov8x_part1_finetune/weights/best.pt")},
    "YOLOv8x-50ep": {"type": "yolo", "threshold": 0.15, "weights": os.path.join(BASE_DIR, "outputs/models/yolov8x_part1_finetune_50ep/weights/best.pt")},
    "YOLO26x-25ep": {"type": "yolo", "threshold": 0.15, "weights": os.path.join(BASE_DIR, "outputs/models/yolo26x_part1_finetune/weights/best.pt")},
    "YOLO26x-50ep": {"type": "yolo", "threshold": 0.3, "weights": os.path.join(BASE_DIR, "outputs/models/yolo26x_part1_finetune_50ep/weights/best.pt")},
    "RF-DETR-25ep": {"type": "rfdetr", "threshold": 0.5, "weights": os.path.join(BASE_DIR, "outputs/models/rfdetr_part1_finetune/weights/best.pth")},
    "RF-DETR-50ep": {"type": "rfdetr", "threshold": 0.5, "weights": os.path.join(BASE_DIR, "outputs/models/rfdetr_part1_finetune_50ep/weights/best.pth")},
}


def compute_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    return sv.detection.utils.iou_and_nms.box_iou_batch(boxes_a, boxes_b)


def get_yolo_predictor(weights_path, threshold):
    model = YOLO(weights_path)

    def predict(image_bgr):
        results = model(image_bgr, classes=[0], conf=threshold, imgsz=IMG_SIZE_YOLO, verbose=False)
        boxes = results[0].boxes
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()

    return predict


def get_rfdetr_predictor(weights_path, threshold):
    model = RFDETRMedium(pretrain_weights=weights_path, trust_checkpoint=True)

    def predict(image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(image_rgb, threshold=threshold)
        return det.xyxy, det.confidence

    return predict


def evaluate_model(name, predict_fn, gt_dataset):
    total_tp, total_fp, total_fn = 0, 0, 0

    for image_name, image_bgr, gt in gt_dataset:
        pred_boxes, pred_scores = predict_fn(image_bgr)

        # Yuksek guvenden dusuge dogru sirala - eslestirmede oncelik
        # yuksek guvenli tahminlerde olsun (coklu tahmin ayni gercek
        # kutuya "carpismasin" diye).
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]

        gt_boxes = gt.xyxy
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        ious = compute_iou_matrix(pred_boxes, gt_boxes)

        for i in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                total_fp += 1
                continue
            best_j = int(np.argmax(ious[i])) if ious.shape[1] > 0 else -1
            if best_j >= 0 and ious[i, best_j] >= IOU_THRESHOLD and not matched_gt[best_j]:
                matched_gt[best_j] = True
                total_tp += 1
            else:
                total_fp += 1

        total_fn += int((~matched_gt).sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"{name:<16} TP={total_tp:<4} FP={total_fp:<4} FN={total_fn:<4}  "
          f"Precision={precision*100:5.1f}%  Recall={recall*100:5.1f}%  F1={f1*100:5.1f}%")
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    print(f"IoU esigi: {IOU_THRESHOLD}")
    print(f"Validation seti: {VAL_IMAGES_DIR} (60 gorsel, gercek insan etiketli)\n")

    gt_dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=VAL_IMAGES_DIR,
        annotations_directory_path=VAL_LABELS_DIR,
        data_yaml_path=DATA_YAML,
    )

    results = {}
    for name, info in MODELS.items():
        threshold = CLI_THRESHOLD_OVERRIDE if CLI_THRESHOLD_OVERRIDE is not None else info["threshold"]
        print(f"{name} yukleniyor ve degerlendiriliyor (esik={threshold})...")
        if info["type"] == "yolo":
            predict_fn = get_yolo_predictor(info["weights"], threshold)
        else:
            predict_fn = get_rfdetr_predictor(info["weights"], threshold)
        results[name] = evaluate_model(f"{name} (esik={threshold})", predict_fn, gt_dataset)

    print(f"\n{'='*70}")
    print("OZET - HER MODEL KENDI ESIGINDE (adil kiyaslama)")
    print(f"{'='*70}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {name:<16} Precision={r['precision']*100:5.1f}%  Recall={r['recall']*100:5.1f}%  F1={r['f1']*100:5.1f}%")


if __name__ == "__main__":
    main()
