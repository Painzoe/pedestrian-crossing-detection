"""
detector.py
------------
Bu script'in amaci: YOLOv8 modelini kullanarak TEK BIR fotografta
insan (person) tespiti yapmak ve sonucu gorsellestirmek.

Bu, projenin ilk test adimi - henuz ROI, sayim, takip yok.
Sadece "model dogru sekilde insanlari buluyor mu?" sorusuna cevap ariyoruz.
"""

from ultralytics import YOLO   # YOLO modelini yuklemek ve calistirmak icin
import cv2                      # Goruntu okuma, cizim ve kaydetme icin (OpenCV)
import os                       # Dosya yolu islemleri icin

# ============================================================
# 1) AYARLAR (buralari kendi ortamina gore degistirebilirsin)
# ============================================================

# Test etmek istedigin fotografin tam yolu.
# Simdilik frames klasorundeki ilk kareyi kullaniyoruz.
INPUT_IMAGE_PATH = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/frames/frame_0000.jpg"

# Sonucu (kutulu goruntuyu) nereye kaydedecegimiz.
OUTPUT_IMAGE_PATH = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/outputs/detector_test_result.jpg"

# YOLO modeli - "yolov8n.pt" = "nano" versiyon, yani en kucuk ve en hizli model.
# Ilk calistirmada bu dosya otomatik olarak internetten indirilecek (~6 MB).
# Dogruluk/hiz dengesini ileride yolov8s.pt veya yolov8m.pt ile degistirebiliriz.
MODEL_NAME = "yolov8n.pt"

# COCO veri setinde "person" sinifinin ID'si 0'dir.
# Bu degeri filtre olarak verecegiz, boylece model sadece insanlari arayacak
# (araba, bisiklet vs. diger 79 sinifi gormezden gelecek).
PERSON_CLASS_ID = 0

# Tespitin "guvenilir" sayilmasi icin gereken minimum dogruluk orani (0-1 arasi).
# 0.5 = model yuzde 50'den fazla eminse o tespiti kabul et.
# Bu degeri sonra ihtiyaca gore ayarlayacagiz (cok dusukse yanlis tespitler artar,
# cok yuksekse gercek insanlari kacirabiliriz).
CONFIDENCE_THRESHOLD = 0.5


# ============================================================
# 2) MODELI YUKLE
# ============================================================

print(f"Model yukleniyor: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
# Not: Bu satir calisinca terminalde indirme ilerlemesi gorebilirsin (ilk seferde).


# ============================================================
# 3) FOTOGRAFI OKU
# ============================================================

# Fotografin gercekten var olup olmadigini kontrol ediyoruz.
# Yanlis yol yazdiysak, YOLO'ya bos/hatali veri gondermek yerine
# burada anlasilir bir hata mesaji verip duruyoruz.
if not os.path.exists(INPUT_IMAGE_PATH):
    raise FileNotFoundError(f"Fotograf bulunamadi: {INPUT_IMAGE_PATH}")

# cv2.imread ile fotografi bir "matris" (piksel dizisi) olarak bellege okuyoruz.
image = cv2.imread(INPUT_IMAGE_PATH)
print(f"Fotograf okundu. Boyut: {image.shape[1]}x{image.shape[0]} piksel")
# image.shape -> (yukseklik, genislik, renk_kanali_sayisi) doner
# ornek: (1080, 1920, 3) -> 1920x1080, RGB (3 kanal)


# ============================================================
# 4) TESPIT (DETECTION) CALISTIR
# ============================================================

# model() fonksiyonu fotografi YOLO'ya verir ve sonuc doner.
# classes=[PERSON_CLASS_ID] -> sadece insan sinifini ara demek
# conf=CONFIDENCE_THRESHOLD -> bu esigin altindaki tespitleri otomatik ele
results = model(image, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD)

# results bir liste doner (birden fazla fotograf verirsek her biri icin 1 eleman).
# Biz tek fotograf verdigimiz icin sadece ilk elemani (results[0]) kullaniyoruz.
result = results[0]

# Kac tane insan bulundu?
detected_boxes = result.boxes  # Bulunan tum kutularin (bounding box) listesi
num_people = len(detected_boxes)
print(f"Tespit edilen insan sayisi: {num_people}")


# ============================================================
# 5) SONUCLARI GORSELLESTIR VE KAYDET
# ============================================================

# Her bir tespit icin detayli bilgi yazdiralim (ogrenme amacli)
for i, box in enumerate(detected_boxes):
    # box.xyxy -> [x_min, y_min, x_max, y_max] formatinda koordinatlar
    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
    confidence = box.conf[0].item()  # bu tespitin guven skoru (0-1 arasi)

    print(f"  Kisi {i + 1}: Guven={confidence:.2f}, "
          f"Konum=({x_min:.0f}, {y_min:.0f}) - ({x_max:.0f}, {y_max:.0f})")

# result.plot() -> YOLO'nun kendi cizim fonksiyonu.
# Bulunan tum kutulari, guven skorlarini fotografin uzerine otomatik cizer.
annotated_image = result.plot()

# Cikti klasoru yoksa olustur (ilk calistirmada hata almamak icin)
output_dir = os.path.dirname(OUTPUT_IMAGE_PATH)
os.makedirs(output_dir, exist_ok=True)

# Sonucu diske kaydet
cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_image)
print(f"\nSonuc kaydedildi: {OUTPUT_IMAGE_PATH}")
print("Bu dosyayi acip kutularin dogru insanlari cevreledigini kontrol et.")