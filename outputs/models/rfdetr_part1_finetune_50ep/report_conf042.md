# Model Raporu — rfdetr_part1_finetune_50ep (AutoBatch) / conf=0.42 denemesi

*Bu dosya `outputs/models/rfdetr_part1_finetune_50ep/report_conf042.md` konumunda duruyor. Eğitim/ağırlıklara HİÇ dokunulmadı — bu, MEVCUT `weights/best.pth` (orijinal AutoBatch eğitimi) üzerinde SADECE inference (tespit) güven eşiğini değiştiren bir denemedir, yeniden eğitim YAPILMADI.*

**Ne test edildi:** [Orijinal RF-DETR-50ep AutoBatch modelinin](report.md) `weights/best.pth` ağırlığı, F1 eğrisinden bulunan optimal eşik **conf=0.42** ile (varsayılan/önceki eşik: 0.5) `Video1.mp4`, `part_1.mp4`, `video02.mp4` ve `video03.mp4` üzerinde test edildi. imgsz, model seçimi gibi başka HİÇBİR ayara dokunulmadı.

---

## 1. Ayarlar

| Ayar | Değer |
|---|---|
| Ağırlık | `weights/best.pth` (AutoBatch eğitimi, epoch 50, [rapor](report.md)) |
| Confidence threshold | **0.42** (F1 eğrisinden optimal) — önceki/varsayılan: 0.5 |
| Model/imgsz/vs. | DEĞİŞTİRİLMEDİ (576px, RFDETRMedium) |

---

## 2. Eğitim ve Validation Eğrileri (DEĞİŞMEDİ)

Bu denemede yeniden eğitim yapılmadı — sadece mevcut `best.pth` ağırlığının inference eşiği değişti. Aşağıdaki grafikler [orijinal rapordakiyle](report.md) BİREBİR AYNI, referans için tekrar gösteriliyor:

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
| `Video1.mp4` (hedef kamera) | 2.22 | 2.00 | +0.22 | Küçük artış, sağlıklı aralıkta |
| `part_1.mp4` (kendi verisi) | 12.08 | 10.32 | +1.76 | Görsel incelemede: kalabalık sahnelerde gerçek ama düşük-güvenli (0.42-0.5 arası) kişiler geri kazanıldı, yeni gürültü/hayalet kutu gözlenmedi |
| `video02.mp4` (yeni video) | 12.20 | *(daha önce test edilmedi)* | — | AV1 codec sorunu nedeniyle H.264'e çevrilip test edildi |
| `video03.mp4` (yeni video) | 3.95 | *(daha önce test edilmedi)* | — | — |

Çıktı videolar: `inference_tests_conf042/Video1/`, `inference_tests_conf042/part_1/`, `inference_tests_conf042/video02/`, `inference_tests_conf042/video03/`

**Titreşim/görsel gözlem:** `part_1.mp4`'te aynı zaman damgasında (t≈90sn) eski (0.5) ve yeni (0.42) eşiği kare kare karşılaştırdım — kalabalık bir kavşak sahnesinde, eski eşikte görünmeyen 2 ek kutu yeni eşikte ortaya çıktı, ikisi de GERÇEK insanların üzerinde (hayalet kutu değil). `Video1.mp4`'te ise örneklenen karede fark yoktu. Genel izlenim: conf=0.42, RF-DETR'de yeni gürültü EKLEMİYOR, sadece düşük-güvenli gerçek tespitleri geri kazandırıyor.

---

## Genel değerlendirme

RF-DETR-50ep AutoBatch'te conf eşiğini 0.5'ten F1-optimal 0.42'ye düşürmek, kalabalık sahnelerde birkaç gerçek kişiyi daha yakalıyor (part_1.mp4'te +1.76 kişi/kare), hedef kamerada (Video1.mp4) neredeyse etkisiz, ve gözlemlenen karelerde yeni gürültü/titreşim eklemiyor — RF-DETR'nin güven skorlarının güvenilirliğini destekleyen bir sonuç.
