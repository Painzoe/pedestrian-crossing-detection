# 4 Model Karşılaştırması — YOLOv8x vs RF-DETR, 25 vs 50 epoch

Bu dosya, `part_1` verisiyle eğitilen 4 farklı model denemesini bir arada özetler:

| # | Model | Klasör | report.md |
|---|---|---|---|
| 1 | YOLOv8x, 25 epoch | `yolov8x_part1_finetune/` | [rapor](yolov8x_part1_finetune/report.md) |
| 2 | YOLOv8x, 50 epoch | `yolov8x_part1_finetune_50ep/` | [rapor](yolov8x_part1_finetune_50ep/report.md) — **OVERFIT, kullanma** |
| 3 | RF-DETR-Medium, 25 epoch | `rfdetr_part1_finetune/` | [rapor](rfdetr_part1_finetune/report.md) |
| 4 | RF-DETR-Medium, 50 epoch | `rfdetr_part1_finetune_50ep/` | [rapor](rfdetr_part1_finetune_50ep/report.md) |

---

## En önemli bulgu: epoch artışı YOLO'yu bozdu, RF-DETR'i etkilemedi

![video test comparison](comparison_assets/video_test_comparison.png)

`Video1.mp4` (modelin HİÇBİRİNİN görmediği, projenin asıl hedef kamerası) üzerinde ortalama kişi/kare:

| | 25 epoch | 50 epoch | Değişim |
|---|---|---|---|
| **YOLOv8x** | 1.22 | **36.24** | **29 kat artış — overfitting, gerçek insan değil** |
| **RF-DETR** | 2.14 | 2.00 | Neredeyse sabit — sağlıklı |

**Neden önemli:** Bu proje "hangi model daha doğru" sorusundan önce, "hangi model daha GÜVENİLİR/ÖNGÖRÜLEBİLİR" sorusuna da cevap veriyor. YOLO'yu daha uzun eğitmek performansı artıracağı yerde tamamen kullanılamaz hale getirdi — bunu sadece gerçek video testiyle (formal validation metrikleriyle DEĞİL) yakaladık.

---

## Eğitim eğrileri — neden bu sorunu ÖNCEDEN göremedik

![map comparison](comparison_assets/map_comparison.png)

Her 4 modelin de kendi `part_1` validation setindeki mAP50-95 eğrisi — **hepsi normal/iyi görünüyor, YOLO 50ep'in Video1.mp4'te çökeceğine dair hiçbir işaret yok.** Bunun sebebi basit: validation seti de `part_1`'in AYNI kamerasından, yani "bu model kendi alanında ne kadar iyi" ölçüyor, "başka bir kamerada ne kadar iyi" sorusuna cevap veremez. RF-DETR'in eğrisi de (turuncu) YOLO'ya (mavi) göre çok daha hızlı ve DÜZGÜN yakınsıyor — ~15-20 epoch'ta zaten doygun, sonrası neredeyse düz çizgi. YOLO'nun eğrisi (özellikle 50ep, düz mavi çizgi) çok daha gürültülü/dalgalı, istikrarsız bir öğrenme sürecine işaret ediyor.

**Ders:** Formal metrikler (mAP, precision, recall) TEK BAŞINA yeterli değil — gerçek, farklı bir kamerada video testi yapmadan bir modelin genelleme yeteneğini bilemezsin.

---

## Tüm sayısal metrikler bir arada

| Metrik | YOLO 25ep | YOLO 50ep | RF-DETR 25ep | RF-DETR 50ep |
|---|---|---|---|---|
| Precision (kendi val. seti) | 0.709 | 0.719 | 0.718 | 0.706 |
| Recall (kendi val. seti) | 0.674 | 0.721 | 0.762 | 0.765 |
| mAP50 (kendi val. seti) | 0.707 | 0.720 | 0.765 | 0.756 |
| mAP50-95 (kendi val. seti) | 0.387 | 0.402 | 0.433 | 0.433 |
| Eğitim süresi | ~8 dk | ~16 dk | ~11 dk | ~22 dk |
| Inference hızı (part_1.mp4) | ~10 fps | ~9.9 fps | ~23 fps | ~23 fps |
| **Video1.mp4 ort. kişi/kare** | 1.22 | **36.24 (bozuk)** | 2.14 | 2.00 |
| video01.mp4 ort. kişi/kare | 5.90 | 7.84 | 11.84 | 12.05 |
| part_1.mp4 ort. kişi/kare | 11.63 | 20.60 (şüpheli) | 9.82 | 10.32 |

---

## Genel sonuç ve öneri

1. **RF-DETR-Medium, 25 epoch** (`rfdetr_part1_finetune/`) şu ana kadarki en dengeli/güvenilir seçim: iyi metrikler, hızlı inference, hem kendi verisinde hem yabancı kameralarda tutarlı, fazla eğitime karşı dayanıklı.
2. **YOLOv8x, 50 epoch** kullanılmamalı — ciddi overfitting var.
3. **YOLOv8x, 25 epoch** hâlâ makul bir yedek seçenek, ama RF-DETR'in gerisinde.
4. RF-DETR için 50 epoch'un 25'e göre somut bir faydası çıkmadı (sayılar hemen hemen aynı) — bu veri boyutunda (240 görsel) 25 epoch yeterli görünüyor, ekstra eğitim süresi (2 kat) karşılığını vermiyor.

**Sıradaki mantıklı adım:** Modeli değil, VERİYİ büyütmek — `Video1.mp4`'ten ve Bellevue veri setinden yeni örnekler etiketleyip `part_1` ile birlikte eğitmek, RF-DETR'in zaten güçlü olan genelleme yeteneğini daha da pekiştirmek.
