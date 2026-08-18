"""
main.py
--------
Projenin BIRLESIK pipeline giris noktasi - detector.py, detector_rfdetr.py
ve tracker.py'nin YERINI ALIR (o dosyalar SILINMEDI, referans olarak
duruyorlar; ama bundan sonraki tum calismalar icin ASIL script budur).

COZULEN SORUN (bkz. outputs/BOUNDING_BOX_TITRESIM_RAPORU.md):
detector.py ve detector_rfdetr.py HER kareyi bagimsiz isliyordu - bir
kisi bir karede kacirilirsa (modellerin kare-basina recall'i sinirli
oldugu icin sik oluyor, bkz. outputs/models/COMPARISON.md) kutusu o
karede TAMAMEN kayboluyor, sonraki karede yeniden tespit edilince
SIFIRDAN bir kutu gibi beliriyordu -> gozle "titresim" olarak
algilaniyordu. tracker.py bunu ID surekliligiyle KISMEN cozuyordu ama
kutu YINE DE sadece o karede taze bir tespit varsa ciziliyordu (ByteTrack
ID'yi hafizada canli tutsa bile).

main.py, ByteTrack'in "lost_tracks" (kaybolmus ama lost_track_buffer
suresi dolmadigi icin hala hafizada tutulan track'ler) listesini
kullanarak, bir kisi birkac kare tespit edilemese bile SON BILINEN
(Kalman filtresiyle tahmin edilen) konumunda kutusunu CIZMEYE devam
ediyor (box-hold / persistence) - titresimi cozen kisim budur (bkz.
build_held_detections()).

NOT: --roi-json VERILMEDIYSE ID numaralari BILEREK ekrana yazilmiyor -
etiket sabit "person" kalir, ByteTrack'in tracker_id'si SADECE box-hold
icin (hangi track'in son ne zaman gorulduğunu bilmek amaciyla) arka
planda kullanilir. --roi-json VERILDIYSE ID'ler etikette gorunur hale
gelir (bkz. asagida ROI SAYIMI bolumu) - ciunku o zaman hangi kisinin
sayildigini gozle takip edebilmek faydali.

ROI SAYIMI (--roi-json): roi_loader.py'deki load_rois/expand_roi/
is_point_in_roi fonksiyonlarini kullanarak, her ROI (orn. bir yaya
gecidi/crosswalk poligonu) icin ESSIZ (unique) kac farkli track ID'sinin
en az bir kere o ROI icinde goruldugunu sayar. ONEMLI: ID atama, ROI'den
TAMAMEN BAGIMSIZ olarak TUM tespitlere yapilir (yukaridaki assign_ids_
to_all_tracks); ROI SADECE "bu ID su an icerde mi, sayilsin mi" kararı
icin SONRADAN kullanilir. Bu sirayi TERSINE cevirip (once ROI'ye gore
filtrele, sonra ID ata) yapmak, ROI kenarinda yuruyen birinin ROI
disina kisa sureli cikinca ID'sini kaybedip yeniden girince YENI ID
almasina, yani CIFT SAYIMA yol acar.

KESINLESEN MODEL KARARI: RF-DETR-Medium, 50 epoch, conf=0.42 (AutoBatch
ile egitildi) - projede kullanilacak TEK model, --model/--epochs/--conf
VARSAYILANLARI buna gore ayarli. Diger denenen modeller/varyantlar
(yolov8x, yolo26x, rfdetr'nin 25ep ve batch1 denemeleri) SILINMEDI,
referans/arsiv olarak outputs/models/ altinda duruyor ve CLI'dan hala
--model/--epochs ile secilebiliyor, ama bundan sonraki calismalar
icin varsayilan/onerilen SADECE rfdetr+50ep.

KULLANIM:
  python main.py <video_yolu> [--model {yolov8x,yolo26x,rfdetr}]
                  [--epochs {25,50}] [--conf ESIK] [--imgsz BOYUT]
                  [--output CIKTI_YOLU] [--hold-frames KARE_SAYISI]
  (--model/--epochs verilmezse varsayilan: rfdetr, 50 epoch, conf=0.42)

ORNEK (varsayilan/kesinlesen model):
  python main.py videos/Video1.mp4

ORNEK (referans amacli baska bir model denemek icin):
  python main.py videos/Video1.mp4 --model yolo26x --epochs 25
"""

from ultralytics import YOLO
from rfdetr import RFDETRMedium
import supervision as sv
import numpy as np
import torch
import cv2
import os
import argparse
from collections import defaultdict

import roi_loader

BASE_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection"

PERSON_CLASS_ID = 0
IMG_SIZE = 1280

# ============================================================
# MODEL SECIM TABLOSU - outputs/models/ altindaki klasor yapisina
# BIREBIR karsilik gelir. Yeni bir fine-tune denemesi eklenirse
# (orn. 75 epoch) buraya tek bir satir eklemek yeterli.
#
# KESINLESEN KARAR: parse_args()'daki --model/--epochs varsayilanlari
# ("rfdetr", 50) olarak ayarli - yani asagidaki ("rfdetr", 50) satiri
# artik projenin TEK resmi modeli. Digerleri (yolov8x, yolo26x,
# rfdetr'nin 25ep/batch1 denemeleri) referans/karsilastirma amacli
# burada DURUYOR ve --model/--epochs ile hala secilebiliyor, ama
# bundan sonraki calismalar icin kullanilmasi ONERILMIYOR.
#
# default_conf degerleri projede zaten dogrulanmis/kullanilan esikler:
# YOLOv8x ve YOLO26x-25ep icin 0.15 (detector.py, tracker.py, roi.py'de
# de ayni), RF-DETR-25ep icin 0.5 (compare_at_threshold.py'deki MODELS
# sozlugunden - modellerin guven skorlari AYNI OLCEKTE DEGIL, oyle
# secilmisti). YOLO26x-50ep (0.45) ve RF-DETR-50ep (0.42) degerleri ise
# F1 egrisinden bulunan optimal esikler (bkz. report_conf045.md /
# report_conf042.md, ilgili model klasorlerinde).
# ============================================================
MODEL_REGISTRY = {
    ("yolov8x", 25): {
        "type": "yolo",
        "weights": "outputs/models/yolov8x_part1_finetune/weights/best.pt",
        "default_conf": 0.15,
    },
    ("yolov8x", 50): {
        "type": "yolo",
        "weights": "outputs/models/yolov8x_part1_finetune_50ep/weights/best.pt",
        "default_conf": 0.15,
    },
    ("yolo26x", 25): {
        "type": "yolo",
        "weights": "outputs/models/yolo26x_part1_finetune/weights/best.pt",
        "default_conf": 0.15,
    },
    ("yolo26x", 50): {
        "type": "yolo",
        "weights": "outputs/models/yolo26x_part1_finetune_50ep/weights/best.pt",
        "default_conf": 0.45,
    },
    ("rfdetr", 25): {
        "type": "rfdetr",
        "weights": "outputs/models/rfdetr_part1_finetune/weights/best.pth",
        "default_conf": 0.5,
    },
    ("rfdetr", 50): {
        "type": "rfdetr",
        "weights": "outputs/models/rfdetr_part1_finetune_50ep/weights/best.pth",
        "default_conf": 0.42,
    },
}

# Box-hold: bir track o karede tespitle eslesmese bile, ByteTrack'in
# Kalman filtresiyle tahmin ettigi son konumunda kutusu KAC KARE daha
# cizilmeye devam etsin. ~30fps bir videoda 8 kare =~ 0.27 saniye - kisa
# sureli kacirilan tespitleri kopruler, ama sahneden gercekten cikmis
# birini gereginden uzun bir "hayalet kutu" olarak ekranda TUTMAZ.
# CLI'dan --hold-frames ile kolayca degistirilebilir.
#
# ONEMLI: bu deger SADECE GORSEL kutu tutma suresidir - ByteTrack'in ID
# HAFIZASI (lost_track_buffer) ile ARTIK BAGLI DEGIL, bkz. asagidaki
# DEFAULT_ID_MEMORY_FRAMES. Ikisi BILINCLI olarak AYRI iki parametre:
# ekrandaki "hayalet kutu" suresi ile arka plandaki "bu ID'yi ne kadar
# hatirlayayim" suresi FARKLI amaclar/farkli kullanici beklentileri.
DEFAULT_BOX_HOLD_FRAMES = 8

# ID HAFIZASI (ByteTrack'in lost_track_buffer'i): bir kisi kamyon/baska bir
# kisi tarafindan 1-3sn ORTULSE bile (kutu ekrandan hold_frames sonra
# kaybolur ama) ID arka planda bu kadar kare boyunca "hatirlanir" - kisi
# tekrar gorununce (reconcile_stranded_new_tracks ile) AYNI ID'yi geri
# alir, YENI ID almaz -> uzun oklüzyonlarda CIFT SAYIM onlenir. 90 kare
# =~ 3sn @30fps (kamyon gecisi/kisi-kisi ortmesi gibi senaryolari
# kapsayacak sekilde secildi). hold_frames'den (GORSEL) BAGIMSIZ, cok
# daha BUYUK bir deger - kutu ekranda 8 kare sonra kaybolur, ama ID
# hafizasi 90 kareye kadar canli kalir.
DEFAULT_ID_MEMORY_FRAMES = 90

# ROI sayimi icin gurultu filtresi: bir ID'nin NIHAI sayima girebilmesi icin
# video boyunca EN AZ BIR KERE ulasmasi gereken ARDISIK TAZE (held degil)
# tespit sayisi. Onceki teshiste (video03) track'lerin %46'sinin SADECE 1
# karede goruntu verdigi bulunmustu (gurultu/flicker) - N=5 (~0.17sn @30fps)
# bu tek/iki-kare gurultuyu buyuk olcude eler, ama kisa sureli gercekten
# kamerada kalan bir yayayi elemeyecek kadar toleransli.
DEFAULT_ROI_MIN_CONSECUTIVE_FRAMES = 5

# PARCA BIRLESTIRME (--merge-fragments, opsiyonel, varsayilan KAPALI) icin
# esikler - bkz. merge_fragmented_tracks() docstring'i. Kalabalik/uzun-kalis
# sahnelerde (orn. part1'deki turistik crosswalk, fotograf cekilirken
# defalarca tikanip yeni ID alan insanlar) reconcile_stranded_new_tracks
# tek-tur oldugu icin kurtaramadigi parcalanmalari, video BITTIKTEN SONRA
# TUM trajectory'yi gorerek duzeltmeye calisir.
#
# max_gap_seconds: DEFAULT_ID_MEMORY_FRAMES (90 kare =~3sn) ile AYNI
# pencere - teshiste (bkz. main.py'nin gecmisi) "yeni ID" olaylarinin
# sadece %4'u gercekten hafiza suresi doldugu icin olustu, geri kalani
# (parcalanma dahil) zaten bu pencerenin ICINDE gerceklesiyor.
DEFAULT_FRAGMENT_MERGE_MAX_GAP_SECONDS = 3.0
# max_dist_factor: bir onceki track'in son hiziyla EKSTRAPOLE edilen
# konumun, sonraki track'in gercek baslangic konumundan ne kadar
# UZAKLASABILECEGI - kutu capinin (avg_diag) kac kati kadar tolerans
# birakiliyor. Olcum gurultusunu (Kalman tahmini kusursuz degil) kapsayacak
# ama TAMAMEN FARKLI bir konumdaki birini kapsamayacak kadar siki.
FRAGMENT_MERGE_MAX_DIST_FACTOR = 1.5
# min_speed_for_direction: hiz vektoru bu esigin (piksel/kare) ALTINDAYSA
# yon karsilastirmasi ATLANIYOR - neredeyse durgun (fotograf cekerken sabit
# duran) biri icin "yon" olcumu gurultuden ibaret olur, boyle durumlarda
# SADECE mesafe testine guveniliyor.
FRAGMENT_MERGE_MIN_SPEED_FOR_DIRECTION = 2.0
# min_direction_cos_sim: cikis hizi ile giris hizi arasindaki kosinus
# benzerligi (1.0 = ayni yon, 0.0 = dik, -1.0 = ters yon) bu esigin
# ALTINDAYSA birlestirme REDDEDILIYOR - art arda gecen IKI FARKLI insanin
# (biri girerken digeri ayni noktadan cikarken) yanlislikla birlesmesini
# onlemek icin ana savunma hatti budur (sadece mesafeye guvenmek yeterli
# degil, bkz. konusma).
FRAGMENT_MERGE_MIN_DIRECTION_COS_SIM = 0.3

IOU_MATCH_THRESHOLD = 0.3  # tracker.py'deki AYNI deger - bir tespiti bir track'e baglamak icin gereken min ortusme


def compute_iou(box_a, box_b):
    """tracker.py'den BIREBIR alindi - iki kutunun (x1,y1,x2,y2) ne kadar
    ortustugunu 0-1 arasi bir sayiyla olcer. 1 = tam ust uste, 0 = hic kesismiyor."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def reconcile_stranded_new_tracks(tracker, known_ids_before):
    """
    HUNGARIAN REKABET DUZELTMESI (ID enflasyonu teshisinden - bkz. konusma
    gecmisi): ByteTrack'in kendi ic eslestirmesi (core.py Step 2), tek bir
    GLOBAL Hungarian atamasi yapiyor - kalabalik bir anda IKI tespit AYNI
    lost track'e "aday" olduysa, sadece maliyeti en dusuk olan CIFT
    kazaniyor. Diger tespit, o track'le TEK BASINA degerlendirilse (yeterli
    IoU ile) eslesecek olsa bile REKABETI KAYBEDIP yepyeni bir track olarak
    aciliyor (kanitlandi: bazen IoU=0.746 gibi COK yuksek bir aday bile
    kaybediliyor). Bu, lost_track_buffer'i artirmakla COZULMEZ (track zaten
    hafizadaydi) - sorun sure degil, REKABET.

    COZUM (bilinen "ikinci tur/second-chance" eslestirme teknigi,
    DeepSORT'un cascade matching'ine benzer, ama kutuphane kodunu
    DEGISTIRMEDEN, sadece public API ile main.py tarafinda uygulaniyor):
    update_with_tensors() bittikten SONRA cagrilir. Bu karede YENI acilan
    (daha once hic gorulmemis internal_track_id'li) track'leri, HALA
    lost_tracks'te kalan (bu karede REKABETTEN dolayi kimseyle eslesmemis)
    track'lerle IKINCI bir IoU turunda tekrar kontrol eder. Yeterince yakin
    (IOU_MATCH_THRESHOLD ustu) bir eski track bulunursa: ByteTrack'in
    KENDI kullandigi re_activate() mekanizmasiyla eski (dogru) track'i bu
    karenin tespitiyle "diriltir", yanlislikla acilan yeni track'i siler -
    boylece DOGRU/ESKI ID geri kullanilir, kalabalikta kaybedilen eslesme
    kurtarilir.

    known_ids_before: bu karenin update_with_tensors() cagrisindan HEMEN
    ONCE (cagiran assign_ids_to_all_tracks tarafindan) alinmis, o ana kadar
    BILINEN (tracked+lost) TUM internal_track_id'lerin kumesi - "YENI acilan"
    track'leri tespit etmek icin referans noktasi.
    """
    newly_created = [t for t in tracker.tracked_tracks if t.internal_track_id not in known_ids_before]
    if not newly_created or not tracker.lost_tracks:
        return

    claimed_lost_indices = set()
    for new_track in newly_created:
        best_iou = IOU_MATCH_THRESHOLD
        best_lost_idx = -1
        for i_lost, lost_track in enumerate(tracker.lost_tracks):
            if i_lost in claimed_lost_indices:
                continue
            iou = compute_iou(new_track.tlbr, lost_track.tlbr)
            if iou > best_iou:
                best_iou = iou
                best_lost_idx = i_lost
        if best_lost_idx == -1:
            continue

        old_track = tracker.lost_tracks[best_lost_idx]
        claimed_lost_indices.add(best_lost_idx)
        # ByteTrack'in KENDI kullandigi mekanizma: eski track'in Kalman
        # durumunu bu karenin (yanlislikla yeni acilan track'e ait) kutusuyla
        # gunceller, state=Tracked yapar - internal_track_id DEGISMEZ (eski
        #/dogru ID korunur).
        old_track.re_activate(new_track, tracker.frame_id)
        tracker.tracked_tracks = [t for t in tracker.tracked_tracks if t is not new_track]
        tracker.tracked_tracks.append(old_track)

    if claimed_lost_indices:
        tracker.lost_tracks = [
            t for i, t in enumerate(tracker.lost_tracks) if i not in claimed_lost_indices
        ]


def assign_ids_to_all_tracks(tracker, detections):
    """tracker.py'den BIREBIR alindi (detayli aciklama icin bkz. o dosya).
    supervision'in tracker.update_with_detections() fonksiyonu YERINE
    kullaniliyor - "en az N ardisik kare" sartini uygulamadan o an takip
    edilen HERKESE ID atar.

    main.py'de bu fonksiyonun dondurdugu tracker_id'ler ekrana YAZILMIYOR -
    SADECE ByteTrack'in ic durumunu (tracker.tracked_tracks / lost_tracks)
    guncellemek ve o karede GERCEKTEN eslesen tespitleri filtrelemek icin
    kullaniliyor. build_held_detections() bu guncellenmis durumu okuyarak
    box-hold kutularini uretiyor.
    """
    known_ids_before = (
        {t.internal_track_id for t in tracker.tracked_tracks} |
        {t.internal_track_id for t in tracker.lost_tracks}
    )

    tensors = np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))
    tracker.update_with_tensors(tensors=tensors)

    # Bkz. reconcile_stranded_new_tracks() docstring'i - ID enflasyonunun
    # asil buyuk sebeplerinden birini (Hungarian rekabet kaybi) burada
    # duzeltiyoruz, all_tracks'i OKUMADAN ONCE.
    reconcile_stranded_new_tracks(tracker, known_ids_before)

    all_tracks = tracker.tracked_tracks

    tracker_ids = np.full(len(detections), -1, dtype=int)
    used_track_indices = set()

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


def build_held_detections(tracker, hold_frames):
    """
    BOX-HOLD MANTIGI - titresim sorununu cozen asil kisim.

    ByteTrack, bir kisi o karede tespit edilemediginde track'i hemen
    silmiyor; internal olarak `lost_tracks` listesinde tutuyor (ta ki
    `lost_track_buffer` kadar kare boyunca hic eslesme olmayana kadar -
    bkz. supervision/tracker/byte_tracker/core.py). Bu track'lerin
    `tlbr` (kutu) degeri, Kalman filtresi sayesinde kaybolduktan sonra
    da guncelleniyor (son bilinen hiz/yon ile tahmin ediliyor).

    detector.py / detector_rfdetr.py ve eski tracker.py bu "lost_tracks"
    bilgisini HIC kullanmiyordu - kutu SADECE o karede taze bir tespit
    varsa ciziliyordu (bkz. outputs/BOUNDING_BOX_TITRESIM_RAPORU.md).
    Burada, kaybolalı `hold_frames` kareden AZ olan track'lerin son
    bilinen konumunu "tutulan" (held) kutu olarak dondurup, ana dongude
    taze tespitlerle BIRLESTIRIP ciziyoruz - boylece bir kisi birkac
    kare kacirilsa bile kutusu ekranda kalmaya devam ediyor.
    """
    held_boxes = []
    # held_ids: her held kutunun hangi track_id'ye ait oldugunu tutuyoruz -
    # ID render (opsiyonel) VE ROI sayimi (bir kisi kisa sureli kacirilsa
    # bile son bilinen konumu ROI icindeyse sayilmaya devam etsin) icin
    # gerekli.
    held_ids = []
    for track in tracker.lost_tracks:
        frames_since_seen = tracker.frame_id - track.frame_id
        if 0 < frames_since_seen <= hold_frames:
            held_boxes.append(track.tlbr)
            held_ids.append(track.internal_track_id)

    if not held_boxes:
        return sv.Detections.empty()

    return sv.Detections(xyxy=np.array(held_boxes, dtype=np.float32),
                          tracker_id=np.array(held_ids, dtype=int))


def merge_fragmented_tracks(track_positions, candidate_ids, fps,
                             max_gap_seconds=DEFAULT_FRAGMENT_MERGE_MAX_GAP_SECONDS,
                             max_dist_factor=FRAGMENT_MERGE_MAX_DIST_FACTOR,
                             min_speed_for_direction=FRAGMENT_MERGE_MIN_SPEED_FOR_DIRECTION,
                             min_direction_cos_sim=FRAGMENT_MERGE_MIN_DIRECTION_COS_SIM):
    """
    OFFLINE PARCA BIRLESTIRME - reconcile_stranded_new_tracks()'in (Hungarian
    ikinci-tur duzeltmesi) tek-turlu/gerceklestigi anda karar verdigi icin
    KURTARAMADIGI parcalanmalari (ozellikle kalabalik/uzun-kalis sahnelerde,
    3+ kisilik rekabet zincirlerinde) video BITTIKTEN SONRA, ID'nin TUM
    trajectory'sini gorerek duzeltmeye calisir - bkz. konusma gecmisi (part1
    turistik crosswalk'ta fotograf cekilirken tekrar tekrar sayi artmasi).

    FIKIR: iki farkli track_id, A (once biten) ve B (sonra baslayan), su
    UCU sart birden saglaniyorsa AYNI fiziksel kisinin iki parcasi sayilir:
      1) ZAMAN: B'nin ilk goruldugu kare ile A'nin son goruldugu kare
         arasindaki fark <= max_gap_seconds (bkz. DEFAULT_FRAGMENT_MERGE_
         MAX_GAP_SECONDS ustundeki not - id-memory-frames penceresiyle ayni).
      2) KONUM: A'nin kaybolmadan onceki SON HIZIYLA o sureye EKSTRAPOLE
         edilen konum, B'nin GERCEK ilk konumuna yakin (kutu capina
         goreceli bir esik icinde) - SADECE "yakinlik" degil, "hareketin
         DEVAMI" test ediliyor.
      3) YON: (hizlar yeterince yuksekse) A'nin cikis yonuyle B'nin giris
         yonu tutarli.

    NEDEN SADECE MESAFE YETERSIZ: art arda gecen IKI FARKLI insan da (biri
    cikarken digeri ayni noktadan girerken - crosswalk giris/cikislari
    ORTAK oldugu icin bu SIK olur) zaman+mesafe testini tek basina
    gecebilir, bu da YANLISLIKLA iki farkli kisiyi birlestirip EKSIK
    sayima yol acar. Hiz/yon ekstrapolasyonu (madde 2-3) bu riski
    azaltan ana savunma: rastgele yakin duran iki farkli insanin hareket
    vektorlerinin TESADUFEN de tutarli olma ihtimali, sadece "yakin olmak"
    tan cok daha dusuktur.

    track_positions: {track_id: [(frame_idx, foot_x, foot_y, box_diag), ...]}
                      SADECE taze/fresh tespitler (held/Kalman-tahmini DEGIL
                      - main.py ana dongusunden, bkz. cagiran kod).
    candidate_ids: SADECE birlestirme icin degerlendirilecek ID kumesi -
                   ROI gurultu filtresini (--roi-min-frames) GECMIS ID'ler
                   verilmeli; gurultu track'lerini birlestirmeye calismanin
                   anlami yok (zaten sayima girmiyorlar).

    Donus: {orijinal_id: canonical_id} sozlugu - birlesmeyen ID'ler kendi
    kendine esler (canonical_id == orijinal_id).
    """
    tracks = {}
    for tid in candidate_ids:
        points = sorted(track_positions.get(tid, []))
        if not points:
            continue
        entry = points[0]
        exit_ = points[-1]
        entry_ref = points[min(4, len(points) - 1)]
        exit_ref = points[max(0, len(points) - 5)]
        avg_diag = sum(p[3] for p in points) / len(points)

        entry_dt = entry_ref[0] - entry[0]
        entry_vel = (
            ((entry_ref[1] - entry[1]) / entry_dt, (entry_ref[2] - entry[2]) / entry_dt)
            if entry_dt > 0 else (0.0, 0.0)
        )
        exit_dt = exit_[0] - exit_ref[0]
        exit_vel = (
            ((exit_[1] - exit_ref[1]) / exit_dt, (exit_[2] - exit_ref[2]) / exit_dt)
            if exit_dt > 0 else (0.0, 0.0)
        )

        tracks[tid] = {
            "entry_frame": entry[0], "entry_pos": (entry[1], entry[2]), "entry_vel": entry_vel,
            "exit_frame": exit_[0], "exit_pos": (exit_[1], exit_[2]), "exit_vel": exit_vel,
            "avg_diag": max(avg_diag, 1.0),
        }

    max_gap_frames = max_gap_seconds * fps

    # ADAY CIFTLER: B'nin girisi A'nin cikisindan SONRA ve pencere icinde
    # olan, mesafe+yon testini gecen (A, B) ciftleri - mesafeye gore
    # (en yakin/en guvenilir once) SIRALANIYOR.
    candidates = []
    for id_a, ta in tracks.items():
        for id_b, tb in tracks.items():
            if id_a == id_b:
                continue
            gap = tb["entry_frame"] - ta["exit_frame"]
            if not (0 < gap <= max_gap_frames):
                continue

            pred_x = ta["exit_pos"][0] + ta["exit_vel"][0] * gap
            pred_y = ta["exit_pos"][1] + ta["exit_vel"][1] * gap
            dist = ((pred_x - tb["entry_pos"][0]) ** 2 + (pred_y - tb["entry_pos"][1]) ** 2) ** 0.5
            scale = (ta["avg_diag"] + tb["avg_diag"]) / 2.0
            if dist > max_dist_factor * scale:
                continue

            speed_a = (ta["exit_vel"][0] ** 2 + ta["exit_vel"][1] ** 2) ** 0.5
            speed_b = (tb["entry_vel"][0] ** 2 + tb["entry_vel"][1] ** 2) ** 0.5
            cos_sim = None
            if speed_a >= min_speed_for_direction and speed_b >= min_speed_for_direction:
                cos_sim = (
                    (ta["exit_vel"][0] * tb["entry_vel"][0] + ta["exit_vel"][1] * tb["entry_vel"][1])
                    / (speed_a * speed_b)
                )
                if cos_sim < min_direction_cos_sim:
                    continue

            candidates.append((dist, id_a, id_b, gap, cos_sim))

    # ACGOZLU (greedy) en-iyi-once eslesme: bir track'in CIKISI en fazla BIR
    # sonraki track'e "devam" olarak baglanabilir, bir track'in GIRISI de en
    # fazla BIR onceki track'ten "devam" alabilir (bir kisi ayni anda ikiye
    # BOLUNEMEZ, iki kisi TEK kisiye BIRLESEMEZ).
    candidates.sort(key=lambda c: c[0])
    exit_used = set()
    entry_used = set()
    parent = {tid: tid for tid in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted_pairs = []
    for dist, id_a, id_b, gap, cos_sim in candidates:
        if id_a in exit_used or id_b in entry_used:
            continue
        exit_used.add(id_a)
        entry_used.add(id_b)
        parent[id_b] = find(id_a)
        accepted_pairs.append({"from": id_a, "to": id_b, "gap_frames": gap,
                                "dist_px": dist, "direction_cos_sim": cos_sim})

    return {tid: find(tid) for tid in tracks}, accepted_pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Birlesik tespit+takip pipeline'i: secilen modelle kisi "
                     "tespiti yapar, ByteTrack ile takip eder, box-hold ile "
                     "titresimi azaltir."
    )
    parser.add_argument("video_path", help="Girdi video dosyasinin yolu")
    # KESINLESEN MODEL KARARI: RF-DETR-Medium, 50 epoch, conf=0.42 (AutoBatch
    # ile egitildi). Diger model aileleri (yolov8x, yolo26x) ve RF-DETR'nin
    # 25 epoch/batch1 varyantlari SILINMEDI, hala --model/--epochs ile
    # CLI'dan secilebilir - sadece artik VARSAYILAN degiller.
    parser.add_argument("--model", default="rfdetr", choices=["yolov8x", "yolo26x", "rfdetr"],
                         help="Kullanilacak model ailesi (varsayilan: rfdetr - projede "
                              "kesinlesen tek model, bkz. MODEL_REGISTRY ustundeki not)")
    parser.add_argument("--epochs", type=int, default=50, choices=[25, 50],
                         help="outputs/models/ altinda hangi fine-tune klasoru kullanilsin "
                              "(varsayilan: 50 - RF-DETR icin kesinlesen deneme; YOLO "
                              "ailelerinde 50 epoch'un genel bir faydasi yok, YOLOv8x'te "
                              "ayrica overfit var, bkz. COMPARISON.md)")
    parser.add_argument("--conf", type=float, default=None,
                         help="Guven esigi (belirtilmezse secilen modelin projede "
                              "dogrulanmis varsayilani kullanilir, bkz. MODEL_REGISTRY)")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE,
                         help=f"Tespit icin goruntu boyutu, SADECE YOLO ailesinde gecerli "
                              f"(varsayilan: {IMG_SIZE})")
    parser.add_argument("--output", default=None,
                         help="Cikti video yolu (belirtilmezse outputs/pipeline_videos/ "
                              "altina otomatik uretilir)")
    parser.add_argument("--hold-frames", type=int, default=DEFAULT_BOX_HOLD_FRAMES,
                         help=f"Box-hold: bir kisi kac kare tespit edilemese de son "
                              f"bilinen konumunda cizilmeye devam etsin - SADECE GORSEL "
                              f"(varsayilan: {DEFAULT_BOX_HOLD_FRAMES})")
    parser.add_argument("--id-memory-frames", type=int, default=DEFAULT_ID_MEMORY_FRAMES,
                         help=f"ID HAFIZASI (ByteTrack lost_track_buffer): bir kisi kamyon/baska "
                              f"bir kisi tarafindan ORTULSE bile ID'sinin arka planda kac kare "
                              f"'hatirlanacagi' - --hold-frames'DEN BAGIMSIZ (kutu ekrandan cok "
                              f"once kaybolabilir ama ID hafizada kalmaya devam eder, kisi geri "
                              f"gelince AYNI ID'yi alir) (varsayilan: {DEFAULT_ID_MEMORY_FRAMES}, "
                              f"~3sn @30fps)")
    parser.add_argument("--roi-json", default=None,
                         help="ROI poligonlarini iceren JSON dosyasinin yolu (roi_loader.py "
                              "formati, bkz. o dosyanin ust kismi). Verilirse video uzerinde "
                              "ROI'ler cizilir ve her ROI icin ESSIZ (unique) ID sayisi "
                              "sayilir/ekrana yazilir. Verilmezse ROI islevi tamamen devre disi "
                              "kalir (varsayilan davranis degismez).")
    parser.add_argument("--roi-expand", type=float, default=0.0,
                         help="ROI poligonlarini kendi merkezinden bu ORANDA disari genislet "
                              "(bkz. roi_loader.expand_roi) - varsayilan 0.0 (genisletme yok, "
                              "JSON'daki poligon oldugu gibi kullanilir). Sadece --roi-json "
                              "verildiyse etkili.")
    parser.add_argument("--roi-min-frames", type=int, default=DEFAULT_ROI_MIN_CONSECUTIVE_FRAMES,
                         help="ROI sayimi icin bir ID'nin NIHAI sayima girebilmesi icin video "
                              "boyunca en az bir kere ulasmasi gereken ARDISIK TAZE (held "
                              "degil) tespit sayisi - kisa omurlu/gurultu track'lerin sayima "
                              f"karismasini engeller (varsayilan: {DEFAULT_ROI_MIN_CONSECUTIVE_FRAMES}). "
                              "Sadece --roi-json verildiyse etkili.")
    parser.add_argument("--merge-fragments", action="store_true",
                         help="ROI sayiminda, ayni kisinin kalabalikta/uzun kalista "
                              "(bkz. reconcile_stranded_new_tracks) PARCALANMIS (farkli "
                              "ID almis) track'lerini, video bittikten sonra hareket "
                              "yonu/hizi ekstrapolasyonuyla TEK kisi olarak birlestirir "
                              "(bkz. merge_fragmented_tracks()). Varsayilan: KAPALI - "
                              "sadece --roi-json verildiyse etkili.")
    return parser.parse_args()


def resolve_model(model_family, epochs):
    """MODEL_REGISTRY'den agirlik dosyasi yolunu bulur, dosyanin
    GERCEKTEN var oldugunu dogrular (yoksa net bir hata verir)."""
    key = (model_family, epochs)
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Desteklenmeyen kombinasyon: model={model_family}, epochs={epochs}")
    info = MODEL_REGISTRY[key]
    weights_path = os.path.join(BASE_DIR, info["weights"])
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model agirliklari bulunamadi: {weights_path}\n"
            f"(outputs/models/ altindaki klasor yapisi degismis olabilir)"
        )
    return info["type"], weights_path, info["default_conf"]


def get_device():
    """GPU varsa kullan, yoksa CPU'ya dus - ekrana HANGI cihazin
    kullanildigini acikca yazdirir (train_yolo26.py'deki desenle ayni,
    boylece "GPU kullanildigindan emin ol" kontrolu terminal ciktisindan
    dogrudan yapilabilir)."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[CIHAZ] GPU kullaniliyor: {device_name}")
        return 0
    print("[CIHAZ] GPU bulunamadi, CPU kullanilacak (YAVAS olabilir).")
    return "cpu"


def load_predictor(model_type, weights_path, device):
    """
    Model ailesinden BAGIMSIZ, ortak bir arayuz uretir:
        predict(frame_bgr, conf, imgsz) -> sv.Detections
    Boylece asagidaki ana isleme dongusu hangi model kullanildigini
    bilmek zorunda kalmiyor (YOLO ve RF-DETR icin ayni kod calisiyor).
    """
    if model_type == "yolo":
        model = YOLO(weights_path)

        def predict(frame_bgr, conf, imgsz):
            results = model(frame_bgr, classes=[PERSON_CLASS_ID], conf=conf,
                             imgsz=imgsz, device=device, verbose=False)
            return sv.Detections.from_ultralytics(results[0])

        return predict

    elif model_type == "rfdetr":
        # trust_checkpoint=True: kendi egittigimiz (Roboflow'un resmi
        # dagitmadigi) bir checkpoint oldugu icin gerekli (detector_rfdetr.py'den).
        model = RFDETRMedium(pretrain_weights=weights_path, trust_checkpoint=True)

        def predict(frame_bgr, conf, imgsz):
            # RF-DETR RGB bekliyor, OpenCV BGR okuyor - cevirmezsek
            # renkler ters olur, model dogru calismaz (detector_rfdetr.py'den).
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return model.predict(frame_rgb, threshold=conf)

        return predict

    raise ValueError(f"Bilinmeyen model tipi: {model_type}")


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video bulunamadi: {args.video_path}")

    model_type, weights_path, default_conf = resolve_model(args.model, args.epochs)
    conf = args.conf if args.conf is not None else default_conf

    if model_type == "rfdetr" and args.imgsz != IMG_SIZE:
        print("[UYARI] --imgsz sadece YOLO ailesinde gecerli, RF-DETR icin yoksayiliyor.")

    if args.output:
        output_path = args.output
    else:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        output_dir = os.path.join(BASE_DIR, "outputs", "pipeline_videos")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{video_basename}_{args.model}_{args.epochs}ep.mp4"
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Model: {args.model} ({args.epochs} epoch) -> {weights_path}")
    print(f"Guven esigi: {conf}")
    print(f"Box-hold (GORSEL kutu tutma): {args.hold_frames} kare")
    print(f"ID hafizasi (arka planda, lost_track_buffer): {args.id_memory_frames} kare")

    device = get_device()
    predict = load_predictor(model_type, weights_path, device)

    # Tek sabit renk (eski detector_rfdetr.py ciktilarindaki AYNI mor,
    # supervision'in varsayilan paletindeki ilk renk: #a351fb). Asagida
    # class_id OLMAYAN, elle birlestirilmis (fresh+held) bir sv.Detections
    # ciziyoruz - tek sv.Color vermek color_lookup'i atlamiyor (yine
    # class_id arayip hata veriyor), o yuzden TEK renkli bir ColorPalette
    # + color_lookup=INDEX kullaniyoruz: INDEX kutunun sıra numarasına
    # gore paletten renk secer, palette 1 renk oldugu icin (idx % 1 = 0)
    # HER ZAMAN ayni rengi verir.
    BOX_COLOR = sv.Color.from_hex("#a351fb")
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette(colors=[BOX_COLOR]),
                                     color_lookup=sv.ColorLookup.INDEX)
    # ID BILEREK render edilmiyor - etiket olarak sabit "person" metni
    # kullaniliyor (bkz. dosya basindaki NOT). Held ve taze kutular
    # GORSEL OLARAK AYNI ciziliyor (kullanici tercihi) - ayrim SADECE
    # asagidaki konsol istatistiklerinde tutuluyor.
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette(colors=[BOX_COLOR]),
                                         color_lookup=sv.ColorLookup.INDEX)

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

    # --roi-json VERILDIYSE: ROI poligonlarini GERCEK video cozunurlugune
    # gore piksele cevir (roi_loader.load_rois), her ROI icin bos bir
    # "essiz ID" kumesi ac. roi_polygons None KALIRSA (varsayilan/ROI
    # verilmedi) asagidaki tum ROI adimlari atlanir - eski davranis
    # (ROI'siz main.py) AYNEN korunuyor.
    roi_polygons = None
    roi_counted_ids = None
    if args.roi_json:
        roi_polygons = roi_loader.load_rois(args.roi_json, width, height)
        if args.roi_expand > 0:
            roi_polygons = {
                name: roi_loader.expand_roi(polygon, args.roi_expand)
                for name, polygon in roi_polygons.items()
            }
        roi_counted_ids = {name: set() for name in roi_polygons}
        print(f"ROI yuklendi: {list(roi_polygons.keys())} "
              f"(genisletme orani: {args.roi_expand}, min ardisik taze kare: {args.roi_min_frames})")

    # GURULTU FILTRESI icin: her ID'nin O ANKI ardisik TAZE (held degil)
    # tespit serisi (bir kare bile "taze" olarak gorunmezse 0'a doner) ve
    # video boyunca ULASTIGI EN UZUN seri. ROI sayimindan BAGIMSIZ, TUM
    # tespitler icin tutuluyor - bir ID'nin nihai ROI sayimina girebilmesi
    # icin bu EN UZUN serinin en az --roi-min-frames olmasi gerekecek
    # (asagida rapor asamasinda filtreleniyor, box-hold/gorsellestirmeye
    # DOKUNMUYOR).
    consecutive_fresh_streak = {}
    max_consecutive_fresh_streak = {}

    # --merge-fragments icin: her ID'nin SADECE taze (held/Kalman-tahmini
    # DEGIL) tespitlerinin (kare, ayak_x, ayak_y, kutu_capi) gecmisi - bkz.
    # merge_fragmented_tracks(). ROI verilmese bile ucretsiz toplaniyor
    # (hesaplama maliyeti ihmal edilebilir), sadece rapor asamasinda
    # --roi-json + --merge-fragments ikisi de verilmisse kullaniliyor.
    track_positions = defaultdict(list)

    # frame_rate'i VIDEONUN GERCEK fps'i ile veriyoruz - eski tracker.py
    # bunu hic vermiyordu (sabit 30 varsayiliyordu), farkli fps'li bir
    # videoda hem box-hold hem ID hafizasi suresi (kare cinsinden) yanlis
    # hesaplanirdi.
    #
    # lost_track_buffer = args.id_memory_frames - ID HAFIZASI, --hold-frames'
    # DEN TAMAMEN BAGIMSIZ bir parametre (bkz. DEFAULT_ID_MEMORY_FRAMES
    # ustundeki NOT). max(args.hold_frames, ...) SADECE bir GUVENLIK TABANI:
    # lost_track_buffer, hold_frames'den KUCUK OLURSA ByteTrack kendi ic
    # mantiginda track'i hold_frames'e ulasmadan siler, build_held_detections()
    # onu hic goremez (box-hold bozulur) - varsayilan degerlerle (90 vs 8)
    # bu taban zaten devreye girmez, sadece kullanici --id-memory-frames'i
    # --hold-frames'den KUCUK bir deger verirse yanlislikla box-hold'u
    # bozmasini engeller.
    tracker = sv.ByteTrack(
        track_activation_threshold=conf - 0.1,
        lost_track_buffer=max(args.hold_frames, args.id_memory_frames),
        frame_rate=fps,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    total_fresh_boxes = 0
    total_held_boxes = 0
    frames_with_hold = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = predict(frame, conf, args.imgsz)
        fresh_detections = assign_ids_to_all_tracks(tracker, detections)
        held_detections = build_held_detections(tracker, args.hold_frames)

        total_fresh_boxes += len(fresh_detections)
        total_held_boxes += len(held_detections)
        if len(held_detections) > 0:
            frames_with_hold += 1

        # Cizim icin taze + held kutulari BIRLESTIRIYORUZ - ikisi de
        # ekranda AYNI gorunecek (renk/stil farki YOK, kullanici tercihi).
        # sv.Detections.merge() confidence/tracker_id gibi alanlarin
        # ikisinde de ayni sekilde dolu/bos olmasini istiyor (aksi halde
        # ValueError firlatiyor) - biz sadece xyxy'ye ihtiyac duydugumuz
        # icin (ID/etiket olarak sabit "person" yaziyoruz) iki kutu
        # dizisini dogrudan np.vstack ile birlestirip TEMIZ bir
        # sv.Detections olusturuyoruz, bu sorunu bastan onluyor.
        combined_xyxy = np.vstack([fresh_detections.xyxy, held_detections.xyxy])
        draw_detections = sv.Detections(xyxy=combined_xyxy)
        fresh_ids = list(fresh_detections.tracker_id) if fresh_detections.tracker_id is not None else []
        held_ids = list(held_detections.tracker_id) if held_detections.tracker_id is not None else []
        combined_ids = fresh_ids + held_ids

        # ARDISIK TAZE SERI GUNCELLEMESI (ROI gurultu filtresi icin) - bu
        # karede TAZE gorunen ID'lerin serisi +1 artiyor, GORUNMEYEN
        # (held dahil - held bir "taze" GORME degil, sadece Kalman tahmini)
        # daha once baslamis serilerin HEPSI 0'a sifirlaniyor. En uzun
        # ulasilan seri ayrica ayri bir sozlukte tutuluyor (rapor
        # asamasinda filtre icin).
        for tid, box in zip(fresh_ids, fresh_detections.xyxy):
            foot_x = (box[0] + box[2]) / 2.0
            foot_y = box[3]
            box_diag = ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5
            track_positions[tid].append((frame_index, foot_x, foot_y, box_diag))

        fresh_ids_this_frame = set(fresh_ids)
        for tid in fresh_ids_this_frame:
            consecutive_fresh_streak[tid] = consecutive_fresh_streak.get(tid, 0) + 1
            if consecutive_fresh_streak[tid] > max_consecutive_fresh_streak.get(tid, 0):
                max_consecutive_fresh_streak[tid] = consecutive_fresh_streak[tid]
        for tid in list(consecutive_fresh_streak.keys()):
            if tid not in fresh_ids_this_frame:
                consecutive_fresh_streak[tid] = 0
        # ID'ler VARSAYILAN olarak (dosya basindaki NOT'a gore) render
        # edilmiyor - sabit "person" etiketi kullaniliyor. SADECE ROI
        # sayimi aktifken (--roi-json verildiyse) ID'leri gosteriyoruz,
        # ciunku o zaman hangi kisinin sayildigini gozle takip etmek
        # faydali (bkz. asagidaki ROI sayim adimi).
        if roi_polygons is not None:
            labels = [f"ID:{tid}" for tid in combined_ids]
        else:
            labels = ["person"] * len(draw_detections)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=draw_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=draw_detections, labels=labels)

        # ============================================================
        # ROI SAYIMI - ONEMLI SIRALAMA: ID atama (assign_ids_to_all_tracks)
        # yukarida, ROI'den TAMAMEN BAGIMSIZ sekilde TUM tespitlere
        # zaten yapildi. ROI burada SADECE "bu ID su an ROI icinde mi,
        # sayilmali mi" kararı icin kullaniliyor - ID atamayi ROI'ye gore
        # FILTRELEMIYORUZ. Bu sira onemli: aksi halde ROI kenarina yakin
        # yuruyen biri ROI disina kisa sureli ciktiginda ID'sini kaybedip
        # tekrar girince YENI ID alir, bu da CIFT SAYIMA yol acar.
        if roi_polygons is not None:
            for box, track_id in zip(combined_xyxy, combined_ids):
                # Kutunun ALT-ORTA noktasi (ayak hizasi) - bir kisinin
                # "nerede durdugu" kutu merkezinden cok ayak noktasiyla
                # daha dogru temsil edilir (kamera acisi nedeniyle kutu
                # merkezi genelde govde/kafa hizasinda kalir).
                foot_point = ((box[0] + box[2]) / 2.0, box[3])
                for roi_name, roi_polygon in roi_polygons.items():
                    if roi_loader.is_point_in_roi(foot_point, roi_polygon):
                        roi_counted_ids[roi_name].add(int(track_id))

            for roi_name, roi_polygon in roi_polygons.items():
                cv2.polylines(annotated_frame, [roi_polygon.reshape(-1, 1, 2)],
                               isClosed=True, color=(0, 255, 255), thickness=2)
                label_pos = (int(roi_polygon[:, 0].min()), int(roi_polygon[:, 1].min()) - 10)
                # Ekrandaki canli sayac da GURULTU FILTRELI (en az
                # --roi-min-frames ardisik taze kare) sayiyi gosteriyor -
                # aksi halde video uzerindeki rakam, en sonda kaydedilen
                # rapordaki (dogru/filtreli) rakamdan FARKLI gorunurdu.
                filtered_count = sum(
                    1 for tid in roi_counted_ids[roi_name]
                    if max_consecutive_fresh_streak.get(tid, 0) >= args.roi_min_frames
                )
                cv2.putText(annotated_frame, f"{roi_name}: {filtered_count}",
                            label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated_frame)

        frame_index += 1
        if frame_index % 30 == 0 or frame_index == total_frames:
            print(f"  {frame_index}/{total_frames} kare islendi "
                  f"(taze: {len(fresh_detections)}, held: {len(held_detections)})")

    cap.release()
    writer.release()

    print("\n" + "=" * 50)
    print("BOX-HOLD OZETI")
    print("=" * 50)
    print(f"Toplam kare: {frame_index}")
    print(f"Toplam taze tespit kutusu: {total_fresh_boxes}")
    print(f"Toplam held (tutulan) kutu: {total_held_boxes}")
    print(f"Box-hold devreye giren kare sayisi: {frames_with_hold} "
          f"(%{100 * frames_with_hold / max(frame_index, 1):.1f})")

    if roi_polygons is not None:
        # GURULTU FILTRESI: bir ID, NIHAI sayima SADECE video boyunca en az
        # bir kere --roi-min-frames ARDISIK TAZE (held degil) tespit serisine
        # ulastiysa giriyor - kisa omurlu/gurultu track'ler burada eleniyor.
        # ONEMLI: bu filtre SADECE raporlama/sayim asamasinda uygulaniyor,
        # box-hold/gorsellestirme (yukaridaki cizim) buna gore DEGISMEDI.
        filtered_roi_ids = {
            roi_name: {tid for tid in ids
                       if max_consecutive_fresh_streak.get(tid, 0) >= args.roi_min_frames}
            for roi_name, ids in roi_counted_ids.items()
        }

        # --merge-fragments: N=5 gurultu filtresini GECMIS (yani zaten
        # "gercek" sayilan) ID'ler arasinda, hiz/yon ekstrapolasyonuyla
        # parcalanmis olanlari birlestir - bkz. merge_fragmented_tracks().
        # Gurultu filtresinden AYRI/SONRAKI bir adim: ikisinin etkisini
        # raporda karistirmadan ayri ayri gosterebilmek icin
        # filtered_roi_ids (merge ONCESI) ayrica saklaniyor.
        merge_map = {}
        merge_pairs = []
        merged_roi_ids = filtered_roi_ids
        if args.merge_fragments:
            candidate_ids = set()
            for ids in filtered_roi_ids.values():
                candidate_ids |= ids
            merge_map, merge_pairs = merge_fragmented_tracks(track_positions, candidate_ids, fps)
            merged_roi_ids = {
                roi_name: {merge_map.get(tid, tid) for tid in ids}
                for roi_name, ids in filtered_roi_ids.items()
            }

        # TOPLAM: butun ROI'lerdeki essiz (FILTRELENMIS VE varsa BIRLESTIRILMIS)
        # ID'lerin BIRLESIMI (union) - bir kisi birden fazla ROI'de gorulsse
        # (orn. iki crosswalk'tan da gecmisse) TOPLAM'da SADECE 1 kere
        # sayiliyor, ciunku soru "kac FARKLI kisi gecti" (ROI'lerin toplami
        # degil).
        all_counted_ids = set()
        for ids in merged_roi_ids.values():
            all_counted_ids |= ids

        report_lines = [
            "ROI SAYIM RAPORU",
            "=" * 50,
            f"Video: {args.video_path}",
            f"Model: {args.model} ({args.epochs} epoch), conf={conf}",
            f"ROI JSON: {args.roi_json}",
            f"Gurultu filtresi: en az {args.roi_min_frames} ardisik taze kare (--roi-min-frames)",
            f"Parca birlestirme (--merge-fragments): "
            f"{'AKTIF (max bosluk: ' + str(DEFAULT_FRAGMENT_MERGE_MAX_GAP_SECONDS) + 'sn)' if args.merge_fragments else 'KAPALI'}",
            "",
        ]
        for roi_name, ids in merged_roi_ids.items():
            raw_count = len(roi_counted_ids[roi_name])
            n_filtered_count = len(filtered_roi_ids[roi_name])
            gurultu_elenen = raw_count - n_filtered_count
            merge_azaltan = n_filtered_count - len(ids)
            extra = f", parca birlestirmeyle azalan: {merge_azaltan}" if args.merge_fragments else ""
            report_lines.append(
                f"{roi_name}: {len(ids)} essiz kisi (ID'ler: {sorted(ids)}) "
                f"[filtre oncesi ham sayi: {raw_count}, elenen gurultu: {gurultu_elenen}{extra}]"
            )
        report_lines.append("-" * 50)
        report_lines.append(f"TOPLAM (en az bir ROI'de sayilan essiz kisi, filtreli): {len(all_counted_ids)}")
        report_lines.append("(Not: bir kisi birden fazla ROI'de gorunmusse TOPLAM'da sadece 1 kere sayilir)")

        if args.merge_fragments:
            groups = defaultdict(list)
            for tid, root in merge_map.items():
                groups[root].append(tid)
            merged_groups = {root: sorted(members) for root, members in groups.items() if len(members) > 1}
            if merged_groups:
                report_lines.append("-" * 50)
                report_lines.append("Birlestirilen ID gruplari (ayni fiziksel kisi sayildi):")
                for root, members in sorted(merged_groups.items()):
                    report_lines.append(f"  {members} -> tek kisi (temsilci ID: {root})")

                # DENETIM (audit) izi - HER ikili birlestirme kararinin
                # dayandigi olcumleri gosteriyor (bkz. merge_fragmented_tracks
                # docstring'i): gap_frames (bosluk), dist_px (ekstrapole
                # edilen konumla gercek konum arasi fark), direction_cos_sim
                # (yon tutarliligi, None = hiz cok dusuk oldugu icin yon
                # kontrolu atlandi). Supheli bir birlestirmeyi (orn. gercekte
                # iki farkli insan) bu satirlara bakarak gozden gecirmek
                # icin - "neden birlesti" sorusunun cevabi burada.
                report_lines.append("")
                report_lines.append("Birlestirme denetim izi (her ikili karar):")
                for pair in merge_pairs:
                    cos_txt = (f"{pair['direction_cos_sim']:.2f}" if pair["direction_cos_sim"] is not None
                                else "N/A (dusuk hiz, yon kontrolu atlandi)")
                    report_lines.append(
                        f"  {pair['from']} -> {pair['to']}: bosluk={pair['gap_frames']:.0f} kare "
                        f"({pair['gap_frames'] / fps:.2f}sn), tahmin-gercek fark={pair['dist_px']:.1f}px, "
                        f"yon benzerligi={cos_txt}"
                    )

        report_text = "\n".join(report_lines)

        print("\n" + report_text)

        # Raporu, video ile AYNI klasore, video ile AYNI isimle (sadece
        # sonuna _roi_report.txt eklenerek) kaydediyoruz - boylece hangi
        # videonun raporu oldugu dosya adindan belli, terminal ciktisi
        # kaybolsa/kaydirilsa bile rapor KALICI olarak diskte duruyor.
        report_path = os.path.splitext(output_path)[0] + "_roi_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\nROI raporu kaydedildi: {report_path}")

    print(f"\nCikti video: {output_path}")


if __name__ == "__main__":
    main()
