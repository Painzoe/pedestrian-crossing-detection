"""
generate_summary_chart.py
----------------------------
Her modelin report.md'sinin EN SONUNA konacak "ozet grafik" - tum
gorsel/detayli egri grafiklerinden (results.png, BoxP/R/F1_curve.png
vb.) FARKLI olarak, TEK bir pencerede TEK bakista "bu model nasil
ogrendi" sorusuna cevap verir:
  - Sol eksen: toplam train loss VE toplam val loss (ayni cizgide,
    ust uste) - ikisi birbirinden ayrilmaya baslarsa (train dusuyor,
    val yukseliyor/duzlesiyorsa) bu OVERFITTING isaretidir.
  - Sag eksen: val precision, val recall, val mAP50 (0-1 araliginda,
    dogruluk trendini gosterir).

YOLO (Ultralytics) VE RF-DETR icin CALISIR - hangi framework oldugunu
klasordeki results.csv (YOLO) ya da metrics.csv (RF-DETR) dosyasinin
varligina bakarak kendisi anlar. Boylece 6 modelin de TEK bir script
ile, AYNI formatta ozet grafigi uretilir.

NOT: Bu script sadece MEVCUT egitim kayitlarindan (results.csv /
metrics.csv) okuma yapar, modeli YENIDEN EGITMEZ.

KULLANIM:
  python generate_summary_chart.py <model_klasoru_adi>
  ornek: python generate_summary_chart.py yolov8x_part1_finetune
  ornek: python generate_summary_chart.py rfdetr_part1_finetune_50ep
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"
MODEL_FOLDER_NAME = sys.argv[1]
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models", MODEL_FOLDER_NAME)
TRAINING_DIR = os.path.join(MODEL_DIR, "training")

YOLO_CSV = os.path.join(TRAINING_DIR, "results.csv")
RFDETR_CSV = os.path.join(TRAINING_DIR, "metrics.csv")


def load_yolo():
    df = pd.read_csv(TRAINING_DIR + "/results.csv")
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"]
    # YOLO'da tek bir "toplam loss" kolonu yok - 3 bileseni (box/cls/
    # dfl VEYA box/cls/l1, modele gore isim degisiyor) TOPLAYARAK
    # "toplam loss" cikariyoruz, hem train hem val icin.
    third_train = "train/dfl_loss" if "train/dfl_loss" in df.columns else "train/l1_loss"
    third_val = "val/dfl_loss" if "val/dfl_loss" in df.columns else "val/l1_loss"
    train_loss = df["train/box_loss"] + df["train/cls_loss"] + df[third_train]
    val_loss = df["val/box_loss"] + df["val/cls_loss"] + df[third_val]
    precision = df["metrics/precision(B)"]
    recall = df["metrics/recall(B)"]
    map50 = df["metrics/mAP50(B)"]
    return epochs, train_loss, val_loss, precision, recall, map50


def load_rfdetr():
    df = pd.read_csv(RFDETR_CSV)
    val_df = df.dropna(subset=["val/precision", "val/recall", "val/mAP_50"])
    val_df = val_df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    train_df = df.dropna(subset=["train/loss"]).groupby("epoch", as_index=False).last()

    epochs = train_df["epoch"] + 1  # RF-DETR 0-indeksli, YOLO ile tutarli olsun diye +1
    train_loss = train_df["train/loss"]
    # val kayiplari farkli epoch satirlarinda - ayni epoch indeksine hizala
    val_loss_by_epoch = dict(zip(val_df["epoch"] + 1, val_df["val/loss"]))
    val_loss = epochs.map(val_loss_by_epoch)
    precision = epochs.map(dict(zip(val_df["epoch"] + 1, val_df["val/precision"])))
    recall = epochs.map(dict(zip(val_df["epoch"] + 1, val_df["val/recall"])))
    map50 = epochs.map(dict(zip(val_df["epoch"] + 1, val_df["val/mAP_50"])))
    return epochs, train_loss, val_loss, precision, recall, map50


def main():
    if os.path.exists(YOLO_CSV):
        framework = "YOLO"
        epochs, train_loss, val_loss, precision, recall, map50 = load_yolo()
    elif os.path.exists(RFDETR_CSV):
        framework = "RF-DETR"
        epochs, train_loss, val_loss, precision, recall, map50 = load_rfdetr()
    else:
        raise FileNotFoundError(f"Ne results.csv ne metrics.csv bulundu: {TRAINING_DIR}")

    fig, ax_loss = plt.subplots(figsize=(11, 6))
    ax_acc = ax_loss.twinx()  # ayni x ekseni, farkli olcekte ikinci y ekseni

    l1, = ax_loss.plot(epochs, train_loss, color="tab:blue", linewidth=2, label="train loss (toplam)")
    l2, = ax_loss.plot(epochs, val_loss, color="tab:red", linewidth=2, label="val loss (toplam)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("Loss (toplam)", color="black")
    ax_loss.grid(alpha=0.3)

    l3, = ax_acc.plot(epochs, precision, color="tab:green", linewidth=1.5, linestyle="--", marker="o", markersize=3, label="val precision")
    l4, = ax_acc.plot(epochs, recall, color="tab:orange", linewidth=1.5, linestyle="--", marker="s", markersize=3, label="val recall")
    l5, = ax_acc.plot(epochs, map50, color="tab:purple", linewidth=1.5, linestyle="--", marker="^", markersize=3, label="val mAP50")
    ax_acc.set_ylabel("Precision / Recall / mAP50 (0-1)", color="black")
    ax_acc.set_ylim(0, 1.05)

    lines = [l1, l2, l3, l4, l5]
    ax_loss.legend(lines, [l.get_label() for l in lines], loc="center right")

    fig.suptitle(f"{MODEL_FOLDER_NAME} ({framework}) - Ozet: Train/Val Loss ve Val Dogruluk Metrikleri Tek Grafikte", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(TRAINING_DIR, "summary_curve.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"summary_curve.png kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
