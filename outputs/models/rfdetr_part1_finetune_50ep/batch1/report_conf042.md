# Model Raporu — rfdetr_part1_finetune_50ep / batch1 / conf=0.42 denemesi

*Bu dosya `outputs/models/rfdetr_part1_finetune_50ep/batch1/report_conf042.md` konumunda duruyor. Eğitim/ağırlıklara HİÇ dokunulmadı — bu, MEVCUT `weights/best.pth` (gerçek batch=1 eğitimi) üzerinde SADECE inference güven eşiğini değiştiren bir denemedir, yeniden eğitim YAPILMADI.*

**Ne test edildi:** [batch=1 eğitilmiş RF-DETR-50ep modelinin](report_batch1.md) `weights/best.pth` ağırlığı, F1-optimal eşik **conf=0.42** ile `Video1.mp4`, `part_1.mp4`, `video02.mp4` ve `video03.mp4` üzerinde test edildi. imgsz, model seçimi gibi başka HİÇBİR ayara dokunulmadı.

---

## 1. Ayarlar

| Ayar | Değer |
|---|---|
| Ağırlık | `weights/best.pth` (gerçek batch=1 eğitimi, epoch 50, [rapor](report_batch1.md)) |
| Confidence threshold | **0.42** (F1 eğrisinden optimal) — önceki/varsayılan: 0.5 |
| Model/imgsz/vs. | DEĞİŞTİRİLMEDİ |

---

## 2. Eğitim ve Validation Eğrileri (DEĞİŞMEDİ)

Bu denemede yeniden eğitim yapılmadı — sadece mevcut `best.pth` ağırlığının inference eşiği değişti. Aşağıdaki grafikler [batch=1 raporundakiyle](report_batch1.md) BİREBİR AYNI, referans için tekrar gösteriliyor:

![results](training/results.png)

---

## 3. Ek görsel analizler (DEĞİŞMEDİ)

### 3.1 Güven eşiği eğrileri

| Dosya | Ne gösteriyor |
|---|---|
| ![P](training/BoxP_curve.png) `BoxP_curve.png` | Precision - eşik |
| ![R](training/BoxR_curve.png) `BoxR_curve.png` | Recall - eşik |
| ![F1](training/BoxF1_curve.png) `BoxF1_curve.png` | F1 - eşik (optimal eşik 0.42 buradan bulundu) |
| ![PR](training/BoxPR_curve.png) `BoxPR_curve.png` | Precision-Recall |

### 3.2 Confusion Matrix

![confusion matrix](training/confusion_matrix_normalized.png)

---

## 4. Özet grafik

![summary](training/summary_curve.png)

---

## 5. Inference Testleri

| Video | Ort. kişi/kare (conf=0.42) | Ort. kişi/kare (eski eşik, conf=0.5) | Fark | Not |
|---|---|---|---|---|
| `Video1.mp4` (hedef kamera) | 2.49 | 2.21 | +0.28 | Küçük artış, sağlıklı aralıkta |
| `part_1.mp4` (kendi verisi) | 12.36 | 10.68 | +1.68 | [AutoBatch'teki](../report_conf042.md) +1.76'ya çok yakın bir artış |
| `video02.mp4` (yeni video) | 12.54 | *(daha önce test edilmedi)* | — | H.264'e çevrilip test edildi (AV1 codec sorunu) |
| `video03.mp4` (yeni video) | 4.62 | *(daha önce test edilmedi)* | — | — |

Çıktı videolar: `inference_tests_conf042/Video1/`, `inference_tests_conf042/part_1/`, `inference_tests_conf042/video02/`, `inference_tests_conf042/video03/`

**Titreşim/görsel gözlem:** `part_1.mp4`'te aynı zaman damgasında (t≈90sn, [AutoBatch'teki](../report_conf042.md) ile AYNI kare) eski (0.5) ve yeni (0.42) eşiği karşılaştırdım — bu spesifik karede görünür bir fark yoktu (aynı kutular, aynı skorlar), ortalama kişi/kare'deki artış videonun başka karelerinde gerçekleşiyor olmalı. Genel eğilim AutoBatch versiyonuyla tutarlı: sayısal artış var, gözlemlenen karede yeni gürültü yok.

---

## Genel değerlendirme

RF-DETR-50ep batch=1'de de conf eşiğini 0.42'ye düşürmek, [AutoBatch'teki bulguyla](../report_conf042.md) tutarlı şekilde küçük-orta bir artış getiriyor (özellikle part_1.mp4'te), hedef kamerada etkisi sınırlı. batch=1 ile AutoBatch arasında bu eşik denemesinde belirgin bir davranış farkı yok — ikisi de aynı yönde, benzer büyüklükte tepki veriyor.
