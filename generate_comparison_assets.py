"""
generate_comparison_assets.py
--------------------------------
4 modeli (YOLO 25ep, YOLO 50ep, RF-DETR 25ep, RF-DETR 50ep) TEK
grafikte kiyaslayan gorseller uretir - outputs/models/COMPARISON.md
icin. Iki tur grafik:
  1) mAP50-95 - epoch egrisi (4 model UST USTE) - "kendi verisinde"
     ne kadar iyi ogrendiklerini gosterir.
  2) Video1.mp4'te (hic gormedikleri hedef kamera) ortalama kisi/kare
     bar grafigi - overfitting'in NEREDE ortaya ciktigini (validation
     egrisinde degil, gercek videoda) gorsel olarak vurgular.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "models", "comparison_assets")
os.makedirs(OUT_DIR, exist_ok=True)


def load_yolo_map(csv_path):
    df = pd.read_csv(csv_path)
    return df["epoch"], df["metrics/mAP50-95(B)"]


def load_rfdetr_map(csv_path):
    df = pd.read_csv(csv_path)
    val_df = df.dropna(subset=["val/mAP_50_95"]).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    return val_df["epoch"] + 1, val_df["val/mAP_50_95"]


def plot_map_comparison():
    yolo25_e, yolo25_v = load_yolo_map(os.path.join(BASE_DIR, "outputs/models/yolov8x_part1_finetune/training/results.csv"))
    yolo50_e, yolo50_v = load_yolo_map(os.path.join(BASE_DIR, "outputs/models/yolov8x_part1_finetune_50ep/training/results.csv"))
    rf25_e, rf25_v = load_rfdetr_map(os.path.join(BASE_DIR, "outputs/models/rfdetr_part1_finetune/training/metrics.csv"))
    rf50_e, rf50_v = load_rfdetr_map(os.path.join(BASE_DIR, "outputs/models/rfdetr_part1_finetune_50ep/training/metrics.csv"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yolo25_e, yolo25_v, label="YOLO 25 epoch", color="tab:blue", linestyle="--", marker="o", markersize=3)
    ax.plot(yolo50_e, yolo50_v, label="YOLO 50 epoch", color="tab:blue", linestyle="-", marker="o", markersize=3)
    ax.plot(rf25_e, rf25_v, label="RF-DETR 25 epoch", color="tab:orange", linestyle="--", marker="s", markersize=3)
    ax.plot(rf50_e, rf50_v, label="RF-DETR 50 epoch", color="tab:orange", linestyle="-", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95 (part_1'in KENDI validation setinde)")
    ax.set_title("4 modelin egitim egrisi - part_1 validation seti\n(Bu grafik hicbirinin Video1.mp4'teki overfitting sorununu GOSTERMIYOR - bkz. 2. grafik)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "map_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def plot_video_comparison():
    # Elle topladigimiz (video testlerinden) ortalama kisi/kare degerleri.
    data = {
        "Video1.mp4\n(hedef kamera,\nhic gormedi)": {
            "YOLO 25ep": 1.22, "YOLO 50ep": 36.24,
            "RF-DETR 25ep": 2.14, "RF-DETR 50ep": 2.00,
        },
        "video01.mp4\n(hic gormedi)": {
            "YOLO 25ep": 5.90, "YOLO 50ep": 7.84,
            "RF-DETR 25ep": 11.84, "RF-DETR 50ep": 12.05,
        },
        "part_1.mp4\n(kendi verisi)": {
            "YOLO 25ep": 11.63, "YOLO 50ep": 20.60,
            "RF-DETR 25ep": 9.82, "RF-DETR 50ep": 10.32,
        },
    }

    videos = list(data.keys())
    models = ["YOLO 25ep", "YOLO 50ep", "RF-DETR 25ep", "RF-DETR 50ep"]
    colors = ["#7fb3ff", "#0d47ff", "#ffc27f", "#ff7f0e"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, video in zip(axes, videos):
        values = [data[video][m] for m in models]
        bars = ax.bar(models, values, color=colors)
        ax.set_title(video, fontsize=11)
        ax.set_ylabel("Ortalama kisi/kare")
        ax.tick_params(axis="x", rotation=25)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Video testlerinde ortalama kisi/kare - 4 model kiyaslamasi\n"
                 "(YOLO 50ep'teki Video1.mp4 sicramasi = overfitting, gercek insan degil)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "video_test_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def main():
    print("Karsilastirma gorselleri uretiliyor...")
    plot_map_comparison()
    plot_video_comparison()
    print(f"\nTamamlandi: {OUT_DIR}")


if __name__ == "__main__":
    main()
