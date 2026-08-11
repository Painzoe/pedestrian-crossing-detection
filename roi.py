"""
roi.py
-------
Amac: Bir video karesi uzerinde FARE ILE elle 4 kose noktasi tiklayarak
yaya gecidi (crossing) ROI (Region of Interest) poligonunu belirlemek,
sonra YOLOv8x ile TUM frame'lerdeki insanlari tespit edip her kisinin
bu poligonun ICINDE mi DISINDA mi oldugunu gorsel olarak ayirt etmek.

Otomatik (goruntu isleme ile zebra cizgisi bulma) yontemden vazgecildi -
ROI'yi kullanici kendi eliyle cizecek. Bu script SADECE:
  1) Elle ROI cizdirir (fare ile 4 nokta tiklama)
  2) YOLO ile TUM kisileri tespit eder
  3) Her kisi icin "ayak noktasi"nin (kutunun alt-orta noktasi) ROI
     icinde mi oldugunu kontrol eder, ICERIDEKINI YESIL, DISARIDAKINI
     GRI kutu ile isaretler

Cizgi gecis SAYIMI (in/out counter) bu adimda YOK - bu ayri, ileriki
bir asama.

NASIL CALISTIRILIR:
  python roi.py
  -> Ilk frame bir pencerede acilir, yaya gecidinin 4 kosesine SIRAYLA
     SOL TIK ile tikla. 4. tiklamadan sonra pencere kisa bir goz atma
     suresinin ardindan otomatik kapanir ve tum frame'lerde tespit baslar.
     Vazgecmek istersen ESC tusuna basabilirsin.
  -> outputs/roi_polygon.json'a kaydedilen poligon bir sonraki
     calistirmada OTOMATIK olarak tekrar kullanilir (yeniden tiklamana
     gerek kalmaz). Yeniden cizmek istersen asagidaki FORCE_REDRAW_ROI
     degiskenini True yap, ya da roi_polygon.json dosyasini sil.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
import pandas as pd

# ============================================================
# 1) AYARLAR
# ============================================================

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

ANNOTATED_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "roi_frames2")
ROI_JSON_PATH = os.path.join(BASE_DIR, "outputs", "roi_polygon2.json")
ROI_PREVIEW_PATH = os.path.join(BASE_DIR, "outputs", "roi_preview2.jpg")
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "roi_summary2.csv")

# detector.py ile TUTARLI tespit ayarlari. Ayni model + ayni conf/imgsz
# kullanmazsak, buradaki kisi sayilari detector.py'nin ciktisiyla
# kiyaslanamaz hale gelir.
# NOT: conf=0.15 kullaniyoruz (0.3 degil) - bu kamera acisi/kalitesi
# (dusuk cozunurluk, tepeden aci, golgeli goruntu) icin test ederek
# dogrulandi: 0.3'te golgeli/kismen gorunen kisiler YOLO tarafindan
# kacirilabiliyordu (bir test karesinde 2 kisi acikca gorunurken
# Toplam: 0 cikmisti). 0.15 bu videoda daha guvenilir sonuc veriyor.
MODEL_NAME = "yolov8x.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 1280

# ROI kac kose noktasindan olusacak. Basit tutmak icin sabit 4 (dortgen) -
# serbest (N kenarli) poligon simdilik gerekmiyor.
NUM_ROI_POINTS = 4

# outputs/roi_polygon.json zaten varsa, bu True olmadikca ONU kullanir,
# yeniden tiklama istemez. Yeniden cizmek istersen True yap.
FORCE_REDRAW_ROI = False


# ============================================================
# 2) FRAME LISTESINI HAZIRLA
# ============================================================

if not os.path.exists(FRAMES_DIR):
    raise FileNotFoundError(f"'{FRAMES_DIR}' klasoru bulunamadi.")

frame_files = sorted([
    f for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(frame_files) == 0:
    raise FileNotFoundError(f"'{FRAMES_DIR}' klasorunde hic fotograf bulunamadi.")

print(f"Toplam {len(frame_files)} frame bulundu.")

# ROI'yi cizerken uzerine tiklanacak referans kare: ilk frame.
reference_frame_path = os.path.join(FRAMES_DIR, frame_files[0])
reference_image = cv2.imread(reference_frame_path)
if reference_image is None:
    raise RuntimeError(f"Referans kare okunamadi: {reference_frame_path}")


# ============================================================
# 3) FARE ILE ELLE ROI CIZME
# ============================================================

def draw_roi_with_mouse(image, num_points):
    """Verilen goruntuyu bir pencerede acar, kullanicinin SOL TIK ile
    isaretledigi noktalari sirayla toplar. num_points kadar nokta
    toplaninca pencere otomatik kapanir ve toplanan noktalar
    (num_points, 2) boyutunda bir numpy dizisi olarak donulur."""

    clicked_points = []
    # display_image uzerine tiklama izlerini (daire/cizgi) kalici olarak
    # cizecegiz - boylece her yeni karede onceki tiklamalar kaybolmuyor.
    display_image = image.copy()

    def on_mouse_click(event, x, y, flags, param):
        # Sadece SOL TIK'i dinliyoruz. num_points'e ulasildiktan sonra
        # gelen fazladan tiklamalari yok sayiyoruz (liste tasmasin diye).
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < num_points:
            clicked_points.append((x, y))
            cv2.circle(display_image, (x, y), 5, (0, 255, 255), -1)
            # Onceki nokta ile bu noktayi birlestirerek olusan poligonun
            # kenarini anlik olarak gosteriyoruz.
            if len(clicked_points) > 1:
                cv2.line(display_image, clicked_points[-2], clicked_points[-1], (0, 255, 255), 2)
            # Son (num_points.) nokta tiklaninca poligonu kapatan kenari da ciz.
            if len(clicked_points) == num_points:
                cv2.line(display_image, clicked_points[-1], clicked_points[0], (0, 255, 255), 2)

    window_name = f"ROI Cizimi - {num_points} kosecik tikla (sirayla)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print(f"\n'{window_name}' penceresi acildi.")
    print(f"Yaya gecidinin {num_points} kosesine SIRAYLA SOL TIKLA.")
    print("Son noktadan sonra pencere otomatik kapanacak. Vazgecmek icin ESC.\n")

    while True:
        # Talimat yazisini (kac nokta tiklandi) her karede tazeliyoruz.
        frame_to_show = display_image.copy()
        cv2.putText(frame_to_show, f"{len(clicked_points)}/{num_points} nokta tiklandi - ESC: iptal",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, frame_to_show)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC -> kullanici vazgecti
            cv2.destroyWindow(window_name)
            raise RuntimeError("ROI cizimi ESC ile iptal edildi.")

        if len(clicked_points) == num_points:
            # Cizilen son kenari kullaniciya kisa bir sure gostermek icin bekle.
            cv2.waitKey(800)
            break

    cv2.destroyWindow(window_name)
    return np.array(clicked_points, dtype=np.int32)


if os.path.exists(ROI_JSON_PATH) and not FORCE_REDRAW_ROI:
    # Daha once cizilmis bir ROI varsa onu tekrar kullan - her test
    # calistirmasinda yeniden tiklamak zorunda kalma.
    with open(ROI_JSON_PATH, "r") as f:
        saved = json.load(f)
    roi_polygon = np.array(saved["polygon"], dtype=np.int32)
    print(f"Kayitli ROI kullanildi: {ROI_JSON_PATH} "
          f"(yeniden cizmek icin FORCE_REDRAW_ROI = True yap)")
else:
    roi_polygon = draw_roi_with_mouse(reference_image, NUM_ROI_POINTS)

    os.makedirs(os.path.dirname(ROI_JSON_PATH), exist_ok=True)
    with open(ROI_JSON_PATH, "w") as f:
        json.dump({"polygon": roi_polygon.tolist()}, f, indent=2)
    print(f"ROI poligonu kaydedildi: {ROI_JSON_PATH}")


# ============================================================
# 4) DOGRULAMA GORSELI (ROI'nin dogru yerde olup olmadigini goz ile
#    kontrol edebilmek icin referans kareye cizip kaydediyoruz)
# ============================================================

preview_image = reference_image.copy()
cv2.polylines(preview_image, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=3)
cv2.imwrite(ROI_PREVIEW_PATH, preview_image)
print(f"Dogrulama gorseli kaydedildi: {ROI_PREVIEW_PATH}")


# ============================================================
# 5) YOLO ILE KISI TESPITI + ROI ICI/DISI AYRIMI (TUM FRAME'LERDE)
# ============================================================

print(f"\nModel yukleniyor: {MODEL_NAME}")
model = YOLO(MODEL_NAME)

os.makedirs(ANNOTATED_OUTPUT_DIR, exist_ok=True)

results_summary = []

for frame_name in frame_files:
    frame_path = os.path.join(FRAMES_DIR, frame_name)
    image = cv2.imread(frame_path)
    if image is None:
        print(f"  [UYARI] '{frame_name}' okunamadi, atlaniyor.")
        continue

    results = model(image, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
                     imgsz=IMG_SIZE, verbose=False)
    boxes = results[0].boxes

    total_count = len(boxes)
    roi_count = 0

    annotated = image.copy()
    cv2.polylines(annotated, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # "Ayak noktasi": kutunun alt-orta noktasi. Kutunun merkezi degil
        # bunu kullaniyoruz cunku kutu bası da kapsar; kisinin ZEMINDE
        # nerede durdugunu en iyi ayaklarinin degdigi nokta gosterir.
        foot_x = (x1 + x2) / 2
        foot_y = y2

        is_inside = cv2.pointPolygonTest(roi_polygon, (foot_x, foot_y), False) >= 0
        if is_inside:
            roi_count += 1

        color = (0, 200, 0) if is_inside else (150, 150, 150)  # ROI ici: yesil, disi: gri
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(annotated, (int(foot_x), int(foot_y)), 4, color, -1)

    cv2.putText(annotated, f"Toplam: {total_count}  ROI ici: {roi_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    output_path = os.path.join(ANNOTATED_OUTPUT_DIR, frame_name)
    cv2.imwrite(output_path, annotated)

    results_summary.append({
        "frame": frame_name,
        "total_person_count": total_count,
        "roi_person_count": roi_count
    })

    print(f"  {frame_name}: toplam={total_count}, roi_ici={roi_count}")


# ============================================================
# 6) CSV OLARAK KAYDET
# ============================================================

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"\nTum frame'ler tarandi.")
print(f"Kutulu/ROI'li gorseller: {ANNOTATED_OUTPUT_DIR}")
print(f"ROI dogrulama gorseli: {ROI_PREVIEW_PATH}")
print(f"Ozet CSV: {CSV_OUTPUT_PATH}")
print(f"\nOrtalama ROI ici kisi sayisi: {summary_df['roi_person_count'].mean():.2f}")
print(f"Maksimum ROI ici kisi sayisi (tek frame'de): {summary_df['roi_person_count'].max()}")
