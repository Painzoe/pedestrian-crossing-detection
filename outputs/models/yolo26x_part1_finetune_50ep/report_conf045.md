# Model Raporu — yolo26x_part1_finetune_50ep (AutoBatch) / conf=0.45 denemesi

*Bu dosya `outputs/models/yolo26x_part1_finetune_50ep/report_conf045.md` konumunda duruyor. Eğitim/ağırlıklara HİÇ dokunulmadı — bu, MEVCUT `weights/best.pt` (orijinal AutoBatch eğitimi) üzerinde SADECE inference güven eşiğini değiştiren bir denemedir, yeniden eğitim YAPILMADI.*

**Ne test edildi:** [Orijinal YOLO26x-50ep AutoBatch modelinin](report.md) `weights/best.pt` ağırlığı, F1 eğrisinden bulunan optimal eşik **conf=0.45** (0.454'e en yakın pratik değer) ile (varsayılan/önceki eşik: 0.15) `Video1.mp4`, `part_1.mp4`, `video02.mp4` ve `video03.mp4` üzerinde test edildi. imgsz, model seçimi gibi başka HİÇBİR ayara dokunulmadı.

---

## ÖNEMLİ BULGU: Yüksek eşik gürültüyü temizliyor ama Video1.mp4'teki asıl "kaçırma" sorununu ÇÖZMÜYOR

Görsel karşılaştırma (aynı kareyi conf=0.15 ve conf=0.45'te izledim), conf=0.15'teki tek tespitin (`person 0.16`, gerçek bir insan ÜZERİNDE DEĞİL, muhtemelen gürültü) conf=0.45'te elendiğini gösterdi — beklenen/olumlu bir temizlik. AMA aynı karedeki İKİ GERÇEK insan (yaya geçidini geçen) HİÇBİR eşikte tespit edilmiyor. Yani ortalama kişi/kare sayısındaki düşüş (1.35→0.48) "daha temiz/doğru" bir sonuç değil, zaten var olan bir "kaçırma" (recall) sorununun üstünü örtüyor — model gerçek insanları baştan beri görmüyordu, yüksek eşik sadece gürültü kutularını da eledi.

---

## 1. Ayarlar

| Ayar | Değer |
|---|---|
| Ağırlık | `weights/best.pt` (AutoBatch eğitimi, epoch 38, [rapor](report.md)) |
| Confidence threshold | **0.45** (F1 eğrisinden optimal, 0.454'e en yakın pratik değer) — önceki/varsayılan: 0.15 |
| Model/imgsz/vs. | DEĞİŞTİRİLMEDİ |

---

## 2. Eğitim ve Validation Eğrileri (DEĞİŞMEDİ)

Bu denemede yeniden eğitim yapılmadı — sadece mevcut `best.pt` ağırlığının inference eşiği değişti. Aşağıdaki grafikler [orijinal rapordakiyle](report.md) BİREBİR AYNI, referans için tekrar gösteriliyor:

![results](training/results.png)

---

## 3. Ek görsel analizler (DEĞİŞMEDİ)

### 3.1 Güven eşiği eğrileri

| Dosya | Ne gösteriyor |
|---|---|
| ![P](training/BoxP_curve.png) `BoxP_curve.png` | Precision - eşik |
| ![R](training/BoxR_curve.png) `BoxR_curve.png` | Recall - eşik |
| ![F1](training/BoxF1_curve.png) `BoxF1_curve.png` | F1 - eşik (optimal eşik 0.45 buradan bulundu) |
| ![PR](training/BoxPR_curve.png) `BoxPR_curve.png` | Precision-Recall |

### 3.2 Confusion Matrix

![confusion matrix](training/confusion_matrix_normalized.png)

---

## 4. Özet grafik

![summary](training/summary_curve.png)

---

## 5. Inference Testleri

| Video | Ort. kişi/kare (conf=0.45) | Ort. kişi/kare (eski eşik, conf=0.15) | Fark | Not |
|---|---|---|---|---|
| `Video1.mp4` (hedef kamera) | 0.48 | 1.35 | -0.87 | ⚠️ Bkz. yukarıdaki önemli bulgu |
| `part_1.mp4` (kendi verisi) | 15.41 | 23.52 | -8.11 | Beklenen düşüş, eşik gürültüyü elemiş |
| `video02.mp4` (yeni video) | 8.06 | *(daha önce test edilmedi)* | — | — |
| `video03.mp4` (yeni video) | 0.49 | *(daha önce test edilmedi)* | — | Çok düşük, bu sahnede model zorlanıyor |

Çıktı videolar: `inference_tests_conf045/Video1/`, `inference_tests_conf045/part_1/`, `inference_tests_conf045/video02/`, `inference_tests_conf045/video03/`

**Titreşim/görsel gözlem:** `Video1.mp4`'te aynı zaman damgasında (t=10sn) eski (0.15) ve yeni (0.45) eşiği karşılaştırdım. Eski eşikte tek düşük-güvenli bir kutu vardı (`person 0.16`), görüntünün sağ-alt köşesinde, ekranda görünen hiçbir insanın üzerinde değil. Yeni eşikte bu kutu elendi — beklenen davranış. Ama sahnedeki asıl iki gerçek insan ne eski ne yeni eşikte tespit edilmedi.

---

## Genel değerlendirme

YOLO26x-50ep AutoBatch'te conf eşiğini 0.45'e çıkarmak sayısal olarak "temiz" bir sonuç veriyor ama görsel doğrulama gösteriyor ki bu, gerçek bir iyileşme değil — model zaten Video1.mp4'teki insanları görmüyordu, yüksek eşik sadece düşük-güvenli gürültü kutularını da elemiş oldu. Bu, projedeki tekrar eden derse (formal/sayısal metriklerin tek başına yeterli olmadığına) bir örnek daha.
