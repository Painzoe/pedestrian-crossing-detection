"""
detect_all_frames.py
---------------------
frames klasorundeki TUM fotograflari tek tek tarar, her birinde YOLO ile
person tespiti yapar, sonucu:
  1) kutulu (annotated) resim olarak diske kaydeder
  2) bir CSV dosyasina (frame adi, tespit edilen kisi sayisi) yazar

Bu adim hala "tek frame" mantiginin devami - henuz takip (tracking) veya
sayim (ROI gecisi) yok. Sadece modelin butun videoda tutarli calisip
calismadigini gormek icin.
"""

from ultralytics import YOLO
import cv2
import os
import pandas as pd

# ============================================================
# 1) AYARLAR
# ============================================================

# Frame'lerin bulundugu klasor
FRAMES_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/frames"

# Kutulu (annotated) resimlerin kaydedilecegi klasor
ANNOTATED_OUTPUT_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/outputs/annotated_frames"

# Ozet sonuclarin yazilacagi CSV dosyasi
CSV_OUTPUT_PATH = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/outputs/detection_summary.csv"

MODEL_NAME = "yolov8x.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.15


# ============================================================
# 2) MODELI YUKLE
# ============================================================

print(f"Model yukleniyor: {MODEL_NAME}")
model = YOLO(MODEL_NAME)


# ============================================================
# 3) FRAME LISTESINI HAZIRLA
# ============================================================

# Klasordeki tum dosyalari listele, sadece .jpg / .jpeg / .png uzantililari al
all_files = os.listdir(FRAMES_DIR)
frame_files = sorted([
    f for f in all_files
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
# sorted() kullaniyoruz cunku frame_0000, frame_0001, frame_0002... seklinde
# isimlendirilmis dosyalarin ZAMAN SIRASINA gore islenmesini istiyoruz.
# Isletim sistemi klasoru varsayilan olarak bu sirayla listelemeyebilir.

if len(frame_files) == 0:
    raise FileNotFoundError(f"'{FRAMES_DIR}' klasorunde hic fotograf bulunamadi.")

print(f"Toplam {len(frame_files)} frame bulundu. Tarama basliyor...\n")

# Kutulu resimlerin kaydedilecegi klasoru olustur (yoksa)
os.makedirs(ANNOTATED_OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4) HER FRAME ICIN TESPIT CALISTIR (ANA DONGU)
# ============================================================

# Sonuclari biriktirecegimiz liste - dongu bitince bunu CSV'ye donusturecegiz.
# Her frame icin bir "sozluk" (dictionary) ekleyecegiz: {"frame": ..., "person_count": ...}
results_summary = []

for frame_name in frame_files:
    # Klasor yolu + dosya adini birlestirip tam yolu olusturuyoruz
    frame_path = os.path.join(FRAMES_DIR, frame_name)

    # Fotografi oku
    image = cv2.imread(frame_path)
    if image is None:
        # Bazen dosya bozuk olabilir; bu durumda hata verip durmak yerine
        # sadece uyari basip bir sonraki frame'e geciyoruz.
        print(f"  [UYARI] '{frame_name}' okunamadi, atlaniyor.")
        continue

    # Tespit calistir. verbose=False -> Ultralytics'in her frame icin
    # terminale bastigi uzun teknik log satirlarini susturuyoruz,
    # kendi kisa ozetimizi biz yazdiriyoruz.
    results = model(image, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, imgsz=1280, verbose=False)
    result = results[0]
    num_people = len(result.boxes)

    # Terminale kisa ozet yaz (ilerlemeyi takip edebilmen icin)
    print(f"  {frame_name}: {num_people} kisi tespit edildi")

    # Kutulu resmi olustur ve diske kaydet (dosya adi orijinaliyle ayni kalsin)
    annotated_image = result.plot()
    output_path = os.path.join(ANNOTATED_OUTPUT_DIR, frame_name)
    cv2.imwrite(output_path, annotated_image)

    # Bu frame'in sonucunu listeye ekle
    results_summary.append({
        "frame": frame_name,
        "person_count": num_people
    })


# ============================================================
# 5) CSV OLARAK KAYDET
# ============================================================

# Liste halindeki sonuclari bir pandas DataFrame'e (yani bir tabloya) donusturuyoruz.
# Her satir bir frame, sutunlar "frame" ve "person_count" olacak.
summary_df = pd.DataFrame(results_summary)

# CSV dosyasina yaz. index=False -> pandas'in kendi ekledigi satir numaralarini
# (0, 1, 2, ...) dosyaya yazmasin, cunku bize gerekli degil.
summary_df.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"\nTum frame'ler tarandi.")
print(f"Kutulu resimler: {ANNOTATED_OUTPUT_DIR}")
print(f"Ozet CSV: {CSV_OUTPUT_PATH}")

# Hizli bir saglama/ozet - kac kisi ortalama gorunuyor, en kalabalik frame kac kisi
print(f"\nOrtalama kisi sayisi: {summary_df['person_count'].mean():.2f}")
print(f"Maksimum kisi sayisi (tek frame'de): {summary_df['person_count'].max()}")