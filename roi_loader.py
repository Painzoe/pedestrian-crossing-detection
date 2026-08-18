"""
roi_loader.py
--------------
Ayri bir projeden (yaya yolu segmentasyonu) gelen ROI (Region of
Interest - "ilgi alani", ornegin bir yaya gecidi/crosswalk poligonu)
JSON dosyalarini okumak icin YARDIMCI fonksiyonlar. SAYIM veya ID
atama mantigi BURADA YOK - main.py bu dosyayi import edip kendi
tespit/takip dongusunde kullanacak (bkz. main.py'deki ID-once-ROI-
sonra sirasi ile ilgili NOT, dosyanin en altinda).

JSON FORMATI (ROI projesinden gelen ornek):
{
  "video": "video03_detected_conf042.mp4",
  "frame_resolution": {"width": 1920, "height": 1080},   # ISTEGE BAGLI
  "rois": [
    {
      "name": "crosswalk_1",
      "polygon_normalized": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    }
  ]
}

Koordinatlar NORMALIZE (0-1 arasi, genislik/yukseklige bolunmus) -
boylece ROI, cizildigi videonun cozunurlugunden BAGIMSIZ tanimlanabiliyor.
"""

import json

import cv2
import numpy as np


def load_rois(json_path, frame_width, frame_height):
    """
    ROI JSON dosyasini okur, icindeki NORMALIZE (0-1) koordinatlari
    VERILEN frame_width/frame_height'a gore GERCEK PIKSEL koordinatlarina
    cevirir.

    frame_width/frame_height BURADA (fonksiyonu cagiran main.py'de)
    islenecek videonun GERCEK cozunurlugu olmali - JSON'daki
    "frame_resolution" alani (varsa) SADECE bir UYARI vermek icin
    kullanilir, donusum icin degil. Boylece ROI, farkli cozunurlukte
    bir videoya uygulansa bile (ayni sahnenin farkli kirpilmis/olceklenmis
    bir kaydi gibi) dogru piksel konumuna oturur.

    Donus degeri: {roi_adi: piksel_poligonu (Nx2 numpy array, int32)}
    seklinde bir sozluk - is_point_in_roi() ve cv2 fonksiyonlarinin
    bekledigi format bu.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON'da "frame_resolution" varsa ve bize verilen gercek video
    # boyutundan FARKLIYSA, sessizce yanlis sonuc uretmek yerine
    # kullaniciyi uyariyoruz - poligon yine de dogru cevriliyor (normalize
    # koordinatlar zaten cozunurluk-bagimsiz), bu SADECE bir bilgi notu.
    resolution_info = data.get("frame_resolution")
    if resolution_info is not None:
        json_width = resolution_info.get("width")
        json_height = resolution_info.get("height")
        if (json_width, json_height) != (frame_width, frame_height):
            print(
                f"[UYARI] roi_loader: ROI JSON'u {json_width}x{json_height} "
                f"cozunurluk icin hazirlanmis, ama verilen video "
                f"{frame_width}x{frame_height}. Koordinatlar normalize "
                f"oldugu icin donusum yine de dogru yapilacak, ancak "
                f"kaynaklarin farkli olmasi (orn. yanlis video/ROI "
                f"eslesmesi) beklenmedik bir durum olabilir - kontrol edin."
            )

    rois_by_name = {}
    for roi in data["rois"]:
        name = roi["name"]
        polygon_normalized = np.array(roi["polygon_normalized"], dtype=np.float32)
        # Normalize (0-1) x'i genislikle, y'yi yukseklikle carpip piksel
        # konumuna ceviriyoruz. cv2.pointPolygonTest int32 bekledigi icin
        # (bkz. is_point_in_roi) burada dogrudan int32'ye yuvarliyoruz.
        polygon_pixels = polygon_normalized * np.array([frame_width, frame_height], dtype=np.float32)
        rois_by_name[name] = polygon_pixels.astype(np.int32)

    return rois_by_name


def expand_roi(polygon, expand_ratio=0.30):
    """
    Verilen poligonu KENDI MERKEZINDEN (centroid) disariya dogru,
    YUZDE bazli bir oranla genisletir - PIKSEL bazli sabit bir genisletme
    DEGIL. Sebep: sabit piksel genisletme, farkli cozunurluk/mesafedeki
    kameralarda (orn. kucuk/uzak bir crosswalk'ta asiri genisleme, buyuk/
    yakin bir crosswalk'ta yetersiz genisleme) TUTARSIZ sonuc verir; oran
    bazli genisletme her poligonun kendi boyutuna GORECELI olarak
    olcekleniyor, boylece tutarli kalir.

    polygon: Nx2 (piksel) koordinat dizisi (load_rois()'un dondurdugu format).
    expand_ratio: 0.30 = her nokta, merkezden uzakligini %30 ARTIRACAK
                  sekilde disari itilir (0.0 = degisiklik yok).

    Donus: ayni sekilde Nx2 int32 numpy array (genisletilmis poligon).
    """
    polygon = np.asarray(polygon, dtype=np.float32)
    centroid = polygon.mean(axis=0)

    # Her noktayi merkezden gecen dogru boyunca disari kaydiriyoruz:
    # yeni_nokta = merkez + (eski_nokta - merkez) * (1 + expand_ratio)
    expanded = centroid + (polygon - centroid) * (1.0 + expand_ratio)

    return expanded.astype(np.int32)


def is_point_in_roi(point, roi_polygon):
    """
    Verilen (x, y) noktasinin roi_polygon poligonunun ICINDE olup
    olmadigini dondurur (sinir cizgisinin TAM UZERINDE olmak da "icinde"
    sayilir).

    OpenCV'nin cv2.pointPolygonTest fonksiyonunu kullaniyoruz:
    - measureDist=False verildiginde SADECE isaret donduruyor
      (1.0 = icinde, -1.0 = disinda, 0.0 = sinir uzerinde) - mesafe
      hesaplamadigi icin measureDist=True'ya gore daha hizli.
    - point/roi_polygon float32'ye cevriliyor - pointPolygonTest tam
      sayi (int32) poligonlarla da calisir ama nokta/poligon farkli
      dtype'da olursa OpenCV hata verebiliyor, o yuzden ikisini de
      ayni (float32) tipe sabitliyoruz.
    """
    roi_polygon = np.asarray(roi_polygon, dtype=np.float32)
    point = (float(point[0]), float(point[1]))

    result = cv2.pointPolygonTest(roi_polygon, point, measureDist=False)
    return result >= 0
