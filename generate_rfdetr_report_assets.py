"""
generate_rfdetr_report_assets.py
-----------------------------------
YOLO (Ultralytics) egitim bitince OTOMATIK olarak results.png,
BoxP/R/F1/PR egrileri, confusion_matrix.png, labels.jpg, train_batch
ornekleri ve val_batch (gercek-vs-tahmin) gorsellerini kendisi
uretiyordu. RF-DETR bunlarin HICBIRINI otomatik uretmiyor - sadece
ham metrics.csv ve checkpoint dosyalari veriyor. Bu script, RF-DETR
icin AYNI gorsel takimini EL ILE (matplotlib + supervision ile)
uretip outputs/models/rfdetr_part1_finetune/training/ altina, YOLO
ile AYNI dosya isimleriyle kaydeder - boylece iki modelin report.md'si
ayni yapida/kalitede olur.

NOT: Bu script SADECE gorsel/rapor uretir, modeli YENIDEN EGITMEZ -
zaten egitilmis outputs/models/rfdetr_part1_finetune/weights/best.pth
agirligini kullanir.

KULLANIM:
  python generate_rfdetr_report_assets.py [model_klasoru_adi]
  ornek: python generate_rfdetr_report_assets.py rfdetr_part1_finetune_50ep
  (belirtilmezse varsayilan "rfdetr_part1_finetune" kullanilir)
"""

import os
import sys
import json
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ekrana degil dosyaya ciz (sunucu/terminal ortaminda calisir)
import matplotlib.pyplot as plt
import cv2
import supervision as sv
from rfdetr import RFDETRMedium

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"
# Komut satirindan model klasor adi verilebilir - boylece ayni script'i
# hem 25 epoch'luk hem 50 epoch'luk (ya da ileride baska) RF-DETR
# denemeleri icin tekrar tekrar kullanabiliyoruz.
MODEL_FOLDER_NAME = sys.argv[1] if len(sys.argv) > 1 else "rfdetr_part1_finetune"
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models", MODEL_FOLDER_NAME)
TRAINING_DIR = os.path.join(MODEL_DIR, "training")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights", "best.pth")

COCO_DATASET_DIR = os.path.join(BASE_DIR, "data", "coco_dataset")
TRAIN_DIR = os.path.join(COCO_DATASET_DIR, "train")
VALID_DIR = os.path.join(COCO_DATASET_DIR, "valid")

METRICS_CSV = os.path.join(TRAINING_DIR, "metrics.csv")

# roi.py/detector.py'de kullandigimiz "resmi" karar esigi - confusion
# matrix ve val_batch ornekleri bu esikte cizilecek (raporun geri
# kalaniyla tutarli olsun diye).
OPERATING_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

RANDOM_SEED = 11
random.seed(RANDOM_SEED)


# ============================================================
# 1) results.png BENZERI - EGITIM EGRILERI (metrics.csv'den)
# ============================================================

def plot_training_curves():
    df = pd.read_csv(METRICS_CSV)
    # Ayni epoch'a ait birden fazla satir olabiliyor (train/val ayri
    # loglaniyor) - val kolonlari dolu olan SON satiri her epoch icin al.
    val_df = df.dropna(subset=["val/precision", "val/recall", "val/mAP_50", "val/mAP_50_95"])
    val_df = val_df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    # train kayiplari (loss) - val olmayan satirlarda dolu
    train_df = df.dropna(subset=["train/loss_bbox", "train/loss_ce", "train/loss_giou"])
    train_df = train_df.groupby("epoch", as_index=False).last()

    epochs_val = val_df["epoch"] + 1  # RF-DETR epoch'lari 0-indeksli, YOLO gibi 1'den baslat
    epochs_train = train_df["epoch"] + 1

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # --- Ust sira: train loss bilesenleri (YOLO'nun box/cls/dfl_loss'una benzer) ---
    axes[0, 0].plot(epochs_train, train_df["train/loss_bbox"], marker="o")
    axes[0, 0].set_title("train/loss_bbox")
    axes[0, 1].plot(epochs_train, train_df["train/loss_ce"], marker="o")
    axes[0, 1].set_title("train/loss_ce (siniflandirma)")
    axes[0, 2].plot(epochs_train, train_df["train/loss_giou"], marker="o")
    axes[0, 2].set_title("train/loss_giou (kutu ortusme)")
    axes[0, 3].plot(epochs_val, val_df["val/precision"], marker="o", color="tab:green")
    axes[0, 3].set_title("metrics/precision")

    # --- Alt sira: val loss + basari metrikleri ---
    axes[1, 0].plot(epochs_val, val_df["val/loss_bbox"], marker="o", color="tab:orange")
    axes[1, 0].set_title("val/loss_bbox")
    axes[1, 1].plot(epochs_val, val_df["val/recall"], marker="o", color="tab:green")
    axes[1, 1].set_title("metrics/recall")
    axes[1, 2].plot(epochs_val, val_df["val/mAP_50"], marker="o", color="tab:green")
    axes[1, 2].set_title("metrics/mAP50")
    axes[1, 3].plot(epochs_val, val_df["val/mAP_50_95"], marker="o", color="tab:green")
    axes[1, 3].set_title("metrics/mAP50-95")

    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)

    fig.suptitle("RF-DETR-Medium egitim egrileri (part1_finetune)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(TRAINING_DIR, "results.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  results.png kaydedildi: {out_path}")


# ============================================================
# 2) VALIDATION SETINDE HAM TESPITLERI TOPLA (bir kere calistir,
#    hem PR/F1 egrileri hem confusion matrix hem val_batch gorselleri
#    icin TEKRAR KULLAN - modeli her defasinda yeniden calistirmaya
#    gerek yok)
# ============================================================

def collect_validation_predictions(model):
    """Validation setindeki HER goruntu icin, COK DUSUK bir esikle
    (0.01) TUM aday kutulari toplar - boylece sonradan istedigimiz
    HERHANGI bir esikte (0.1, 0.2, ..., 0.9) yeniden tespit
    calistirmadan filtreleyebiliriz (hizli)."""
    gt_dataset = sv.DetectionDataset.from_coco(
        images_directory_path=VALID_DIR,
        annotations_path=os.path.join(VALID_DIR, "_annotations.coco.json"),
    )

    image_names = []
    ground_truths = []
    raw_predictions = []

    for image_name, image_bgr, gt_detections in gt_dataset:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred = model.predict(image_rgb, threshold=0.01)

        image_names.append(image_name)
        ground_truths.append(gt_detections)
        raw_predictions.append(pred)

    print(f"  {len(image_names)} validation goruntusu icin ham tespitler toplandi.")
    return image_names, ground_truths, raw_predictions


# ============================================================
# 3) BoxP/R/F1/PR EGRILERI - guven esigini tarayarak
# ============================================================

def compute_iou_matrix(boxes_a, boxes_b):
    """boxes_a: (N,4), boxes_b: (M,4) -> (N,M) IoU matrisi."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    return sv.detection.utils.iou_and_nms.box_iou_batch(boxes_a, boxes_b)


def evaluate_at_threshold(ground_truths, raw_predictions, conf_threshold):
    """Verilen guven esiginde TUM validation seti icin toplam
    TP/FP/FN sayar (IoU >= IOU_THRESHOLD eslesme kurali ile,
    her ground-truth kutusu EN FAZLA bir tahminle eslesebilir)."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for gt, pred in zip(ground_truths, raw_predictions):
        keep = pred.confidence >= conf_threshold
        pred_boxes = pred.xyxy[keep]
        pred_scores = pred.confidence[keep]
        # yuksek guvenden dusuge dogru sirala - eslestirmede oncelik
        # yuksek guvenli tahminlerde olsun
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]

        gt_boxes = gt.xyxy
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        ious = compute_iou_matrix(pred_boxes, gt_boxes)
        for i in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                total_fp += 1
                continue
            best_j = np.argmax(ious[i]) if ious.shape[1] > 0 else -1
            if best_j >= 0 and ious[i, best_j] >= IOU_THRESHOLD and not matched_gt[best_j]:
                matched_gt[best_j] = True
                total_tp += 1
            else:
                total_fp += 1

        total_fn += int((~matched_gt).sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def plot_pr_f1_curves(ground_truths, raw_predictions):
    thresholds = np.linspace(0.02, 0.98, 49)
    precisions, recalls, f1s = [], [], []

    for th in thresholds:
        p, r, f1 = evaluate_at_threshold(ground_truths, raw_predictions, th)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    precisions, recalls, f1s = map(np.array, (precisions, recalls, f1s))
    best_idx = int(np.argmax(f1s))

    def _save_curve(y, ylabel, filename, color):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thresholds, y, color=color, linewidth=2)
        ax.axvline(thresholds[best_idx], color="gray", linestyle="--", alpha=0.6,
                    label=f"en iyi F1 esigi = {thresholds[best_idx]:.2f}")
        ax.set_xlabel("Guven esigi (confidence)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} - Guven Esigi Egrisi (person)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(TRAINING_DIR, filename)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  {filename} kaydedildi (en iyi deger={y[best_idx]:.3f} @ esik={thresholds[best_idx]:.2f})")

    _save_curve(precisions, "Precision", "BoxP_curve.png", "tab:blue")
    _save_curve(recalls, "Recall", "BoxR_curve.png", "tab:orange")
    _save_curve(f1s, "F1", "BoxF1_curve.png", "tab:green")

    # PR egrisi: Precision'i Recall'a karsi ciz (esik parametrik)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, color="tab:purple", linewidth=2)
    ax.scatter([recalls[best_idx]], [precisions[best_idx]], color="red", zorder=5,
               label=f"en iyi F1 noktasi (esik={thresholds[best_idx]:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Egrisi (person)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(TRAINING_DIR, "BoxPR_curve.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  BoxPR_curve.png kaydedildi")

    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], f1s[best_idx]


# ============================================================
# 4) CONFUSION MATRIX (OPERATING_THRESHOLD'ta, supervision ile)
# ============================================================

def plot_confusion_matrix(ground_truths, raw_predictions):
    # OPERATING_THRESHOLD'a gore filtrelenmis "gercek" tahmin listesi olustur
    filtered_predictions = []
    for pred in raw_predictions:
        keep = pred.confidence >= OPERATING_THRESHOLD
        filtered_predictions.append(pred[keep])

    cm = sv.ConfusionMatrix.from_detections(
        predictions=filtered_predictions,
        targets=ground_truths,
        classes=["person"],
        conf_threshold=OPERATING_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    cm.plot(save_path=os.path.join(TRAINING_DIR, "confusion_matrix.png"),
            title="Confusion Matrix (RF-DETR, esik=0.5)")
    cm.plot(save_path=os.path.join(TRAINING_DIR, "confusion_matrix_normalized.png"),
            title="Confusion Matrix Normalized (RF-DETR, esik=0.5)", normalize=True)
    print("  confusion_matrix.png ve confusion_matrix_normalized.png kaydedildi")


# ============================================================
# 5) labels.jpg - EGITIM VERISININ KUTU DAGILIMI
# ============================================================

def plot_labels_summary():
    with open(os.path.join(TRAIN_DIR, "_annotations.coco.json")) as f:
        coco = json.load(f)

    image_size_by_id = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    x_centers, y_centers, widths, heights = [], [], [], []
    for ann in coco["annotations"]:
        img_w, img_h = image_size_by_id[ann["image_id"]]
        bx, by, bw, bh = ann["bbox"]
        # 0-1 arasina normalize et (YOLO'nun labels.jpg'si de normalize gosterir)
        x_centers.append((bx + bw / 2) / img_w)
        y_centers.append((by + bh / 2) / img_h)
        widths.append(bw / img_w)
        heights.append(bh / img_h)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist2d(x_centers, y_centers, bins=30, cmap="viridis")
    axes[0].set_title("Kutu merkezlerinin goruntudeki dagilimi (x, y)")
    axes[0].set_xlabel("x (normalize)")
    axes[0].set_ylabel("y (normalize)")
    axes[0].invert_yaxis()

    axes[1].hist2d(widths, heights, bins=30, cmap="viridis")
    axes[1].set_title("Kutu genislik/yukseklik dagilimi")
    axes[1].set_xlabel("genislik (normalize)")
    axes[1].set_ylabel("yukseklik (normalize)")

    axes[2].hist(widths, bins=30, alpha=0.6, label="genislik")
    axes[2].hist(heights, bins=30, alpha=0.6, label="yukseklik")
    axes[2].set_title("Kutu boyut histogrami")
    axes[2].legend()

    fig.suptitle(f"Egitim verisi ozeti - {len(coco['annotations'])} kutu, {len(coco['images'])} goruntu", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(TRAINING_DIR, "labels.jpg")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  labels.jpg kaydedildi: {out_path}")


# ============================================================
# 6) train_batch ORNEKLERI - egitim goruntulerinden birkac ornek,
#    gercek (ground truth) kutularla
# ============================================================

def save_train_batch_examples():
    with open(os.path.join(TRAIN_DIR, "_annotations.coco.json")) as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images_by_id = {img["id"]: img for img in coco["images"]}
    sample_image_ids = random.sample(list(images_by_id.keys()), min(3, len(images_by_id)))

    for idx, image_id in enumerate(sample_image_ids):
        image_info = images_by_id[image_id]
        image = cv2.imread(os.path.join(TRAIN_DIR, image_info["file_name"]))
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
        out_path = os.path.join(TRAINING_DIR, f"train_batch{idx}.jpg")
        cv2.imwrite(out_path, image)

    print(f"  {len(sample_image_ids)} train_batch ornek goruntu kaydedildi.")


# ============================================================
# 7) val_batch GERCEK-vs-TAHMIN KARSILASTIRMASI
# ============================================================

def save_val_batch_comparisons(image_names, ground_truths, raw_predictions):
    n = min(3, len(image_names))
    sample_indices = random.sample(range(len(image_names)), n)

    for out_idx, i in enumerate(sample_indices):
        image_path = os.path.join(VALID_DIR, image_names[i])
        image = cv2.imread(image_path)

        gt_image = image.copy()
        for box in ground_truths[i].xyxy.astype(int):
            cv2.rectangle(gt_image, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), 2)
        cv2.imwrite(os.path.join(TRAINING_DIR, f"val_batch{out_idx}_labels.jpg"), gt_image)

        pred = raw_predictions[i]
        keep = pred.confidence >= OPERATING_THRESHOLD
        pred_image = image.copy()
        for box, conf in zip(pred.xyxy[keep].astype(int), pred.confidence[keep]):
            cv2.rectangle(pred_image, (box[0], box[1]), (box[2], box[3]), (255, 100, 0), 2)
            cv2.putText(pred_image, f"person {conf:.2f}", (box[0], max(box[1] - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.imwrite(os.path.join(TRAINING_DIR, f"val_batch{out_idx}_pred.jpg"), pred_image)

    print(f"  {n} val_batch gercek/tahmin karsilastirma cifti kaydedildi.")


# ============================================================
# ANA AKIS
# ============================================================

def main():
    print("1) Egitim egrileri (results.png) ciziliyor...")
    plot_training_curves()

    print("\nModel yukleniyor (validation seti icin tespitler toplanacak)...")
    model = RFDETRMedium(pretrain_weights=WEIGHTS_PATH, trust_checkpoint=True)

    print("\n2) Validation setinde ham tespitler toplaniyor...")
    image_names, ground_truths, raw_predictions = collect_validation_predictions(model)

    print("\n3) BoxP/R/F1/PR egrileri hesaplaniyor (guven esigi taraniyor)...")
    best_th, best_p, best_r, best_f1 = plot_pr_f1_curves(ground_truths, raw_predictions)
    print(f"  En iyi F1 noktasi: esik={best_th:.2f}, P={best_p:.3f}, R={best_r:.3f}, F1={best_f1:.3f}")

    print("\n4) Confusion matrix ciziliyor...")
    plot_confusion_matrix(ground_truths, raw_predictions)

    print("\n5) labels.jpg (egitim verisi ozeti) ciziliyor...")
    plot_labels_summary()

    print("\n6) train_batch ornekleri kaydediliyor...")
    save_train_batch_examples()

    print("\n7) val_batch gercek-vs-tahmin karsilastirmalari kaydediliyor...")
    save_val_batch_comparisons(image_names, ground_truths, raw_predictions)

    print(f"\nTamamlandi. Tum gorseller: {TRAINING_DIR}")


if __name__ == "__main__":
    main()
