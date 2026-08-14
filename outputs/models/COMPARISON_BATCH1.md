# Model Karşılaştırması — batch_size=1 Denemesi (6 Model)

Bu dosya, [orijinal COMPARISON.md](COMPARISON.md)'deki 6 modelin (YOLOv8x/YOLO26x/RF-DETR × 25ep/50ep) **`batch_size=1`** (gerçek/ham, gizli gradyan biriktirme kapalı — `nbs=1`) ile yeniden eğitilmiş hallerini özetler. Orijinal AutoBatch sonuçlarına HİÇ dokunulmadı, hepsi kendi `batch1/` alt klasöründe ayrı duruyor.

| # | Model | Rapor |
|---|---|---|
| 1 | YOLOv8x, 25 epoch, batch=1 | [rapor](yolov8x_part1_finetune/batch1/report_batch1.md) |
| 2 | YOLOv8x, 50 epoch, batch=1 | [rapor](yolov8x_part1_finetune_50ep/batch1/report_batch1.md) |
| 3 | YOLO26x, 25 epoch, batch=1 | [rapor](yolo26x_part1_finetune/batch1/report_batch1.md) |
| 4 | YOLO26x, 50 epoch, batch=1 | [rapor](yolo26x_part1_finetune_50ep/batch1/report_batch1.md) |
| 5 | RF-DETR-Medium, 25 epoch, batch=1 | [rapor](rfdetr_part1_finetune/batch1/report_batch1.md) |
| 6 | RF-DETR-Medium, 50 epoch, batch=1 | [rapor](rfdetr_part1_finetune_50ep/batch1/report_batch1.md) |

**Metodoloji notu:** İlk YOLOv8x-25ep denemesinde Ultralytics'in gizli `nbs=64` mekanizması yüzünden "batch=1" aslında etkin batch=64 çıkıyordu — bu bulunup `nbs=1` ile düzeltildi, tüm YOLO denemeleri bu düzeltilmiş ayarla yapıldı ([bkz. detay](yolov8x_part1_finetune/batch1/report_batch1.md)). RF-DETR'de karşılığı `grad_accum_steps=1`.

---

## En önemli bulgu: Mimari (LayerNorm vs BatchNorm) batch=1'e verdiği tepkide kökten farklı — ve BatchNorm ailesi içinde bile SONUÇ ÖNGÖRÜLEMEZ

### Hedef kamerada (Video1.mp4) batch=1'in etkisi

| Model | Orijinal (AutoBatch) | batch=1 | Değişim |
|---|---|---|---|
| YOLOv8x 25ep | 1.22 | 2.11 | İyileşme yönünde |
| **YOLOv8x 50ep** | **36.24** (halüsinasyon) | **0.23** (neredeyse sıfır) | **TERS yönde bir başka aşırılık** ⚠️ doğrulama gerekiyor |
| YOLO26x 25ep | 0.32 (zayıf) | 2.24 | Güçlü iyileşme |
| YOLO26x 50ep | 1.35 (zayıf) | 2.59 | Güçlü iyileşme |
| RF-DETR 25ep | 2.14 | 2.33 | Neredeyse aynı |
| RF-DETR 50ep | 2.00 | 2.21 | Neredeyse aynı |

**Okuma:**
- **RF-DETR (LayerNorm):** batch=1'e karşı tamamen kayıtsız — hem 25 hem 50 epoch'ta sonuç orijinalle pratik olarak aynı. LayerNorm'un örnek-bazlı normalizasyonu beklendiği gibi çalıştı.
- **YOLO26x (BatchNorm), her iki epoch sayısında da:** batch=1, orijinalin hedef kamerada zayıf/eksik-tespit sorununu TUTARLI şekilde iyileştirdi (0.32→2.24, 1.35→2.59) — RF-DETR aralığına yaklaştı.
- **YOLOv8x (BatchNorm), epoch sayısına göre TAMAMEN FARKLI davrandı:** 25ep'te iyileşme (1.22→2.11, YOLO26x'e benzer), ama 50ep'te orijinalin "halüsinasyon" sorununu (36.24) "neredeyse hiç tespit yok" sorununa (0.23) çevirdi — iki AŞIRI UÇ arasında, ama hiçbiri sağlıklı değil.

**Sonuç:** Batch boyutunun etkisi mimara (BatchNorm vs LayerNorm) göre kökten farklı, ama BatchNorm ailesi (YOLO) içinde bile MODEL/EPOCH'A ÖZGÜ — batch=1 bazen düzeltiyor (YOLO26x), bazen sorunun yönünü değiştiriyor ama çözmüyor (YOLOv8x-50ep). Bu, batch boyutunun CNN/BatchNorm modellerinde sadece "gürültü miktarını" değil, modelin HANGİ YÖNDE hata yapacağını da etkileyebildiğinin kanıtı. **⚠️ Bu tablodaki tüm Video1.mp4 sonuçları henüz görsel olarak doğrulanmadı** — sayısal ortalamalar tek başına "gerçek tespit mi gürültü mü" sorusuna kesin cevap vermiyor, ayrı bir görsel inceleme gerekiyor.

---

## part_1.mp4 (kendi verisi) ve video01.mp4 (görülmemiş, hedef olmayan kamera)

| Model | part_1 orijinal | part_1 batch=1 | video01 orijinal | video01 batch=1 |
|---|---|---|---|---|
| YOLOv8x 25ep | 11.63 | 25.14 ⚠️ | 5.90 | 7.43 |
| YOLOv8x 50ep | 20.60 | 20.30 | 7.84 | 6.20 |
| YOLO26x 25ep | 18.39 | 18.99 | 7.61 | 7.99 |
| YOLO26x 50ep | 23.52 | 18.55 | 9.44 | 5.32 |
| RF-DETR 25ep | 9.82 | 10.72 | 11.84 | 10.19 |
| RF-DETR 50ep | 10.32 | 10.68 | 12.05 | 10.34 |

**Gözlem:** RF-DETR'de fark her zaman küçük (±1 civarı). YOLO ailesinde dalgalanma daha büyük, en dikkat çekeni YOLOv8x-25ep'in `part_1.mp4`'te 2.2 kat artışı (11.63→25.14) — bu da henüz görsel doğrulanmadı.

---

## Kendi validation setindeki (mAP50-95, best epoch) etkisi — hepsi ya aynı ya da hafif iyi

| Model | Orijinal (AutoBatch) | batch=1 | Fark |
|---|---|---|---|
| YOLOv8x 25ep | 0.387 | 0.399-0.404 | hafif iyi |
| YOLOv8x 50ep | 0.402 | 0.435 | belirgin iyi |
| YOLO26x 25ep | 0.394 | 0.384 | hafif düşük |
| YOLO26x 50ep | 0.432 | 0.443 | hafif iyi |
| RF-DETR 25ep | 0.433 | 0.419-0.427 | ihmal edilebilir fark |
| RF-DETR 50ep | 0.433 | 0.430 | ihmal edilebilir fark |

**Ders (bir öncekinin tekrarı, batch=1'de de doğrulandı):** Kendi validation setindeki metrikler TÜM modellerde "iyi/normal" görünüyor — hiçbiri Video1.mp4'teki büyük sapmaları (ne YOLOv8x-50ep'in çöküşünü, ne YOLO26x'in iyileşmesini) önceden haber vermiyor. Formal metrikler tek başına yeterli değil, bu projede tekrar tekrar doğrulanan bir bulgu.

---

## Eğitim istikrarı — BatchNorm ailesinde epoch-1 sarsıntısı, LayerNorm'da yok

- **RF-DETR (her iki epoch sayısı):** Sorunsuz, düzgün yakınsama. Sadece `val/loss_ce` gürültülü görünüyor ama bu, precision/recall/mAP'yi etkilemeyen, kendinden-emin-yanlış tahminleri ağır cezalandıran bir cross-entropy artefaktı (detaylı açıklama ilgili raporlarda).
- **YOLOv8x-25ep:** Epoch 1'de neredeyse toplam çöküş (precision 0.004, mAP50 0.000), epoch 5+'ten itibaren toparlanma.
- **YOLOv8x-50ep / YOLO26x-25ep / YOLO26x-50ep:** Daha yumuşak ama benzer bir dip/toparlanma deseni var (epoch 5-10 civarı en düşük nokta, sonra istikrarlı yükseliş).

Bu, BatchNorm'un tek-örneklik istatistiklerle (batch=1) beklenen dengesizliğinin somut kanıtı — LayerNorm'un (RF-DETR) bu sorunu yaşamadığı iddiasını doğruluyor.

---

## Doygunluk (saturation) bulgusu — batch=1'de de aynı desen sürüyor

50 epoch'luk denemelerde `best.pt`/`best.pth` çoğunlukla son epoch DEĞİL:

| Model (50ep) | best ağırlık epoch'u | Orijinal AutoBatch'teki karşılığı |
|---|---|---|
| YOLOv8x batch=1 | 36 (ama 50'ye çok yakın, gerçek doygunluk değil) | — |
| YOLO26x batch=1 | **40** | Orijinalde **38** — neredeyse aynı |
| RF-DETR batch=1 | **47** | Orijinalde net bir epoch numarası verilmemişti, ama yakın davranış |

**Ders:** Doygunluk noktası, batch boyutundan bağımsız, veri setinin (240 görsel) kendi sınırlarından kaynaklanıyor — bu projedeki en tutarlı, mimariler ve batch boyutları arasında tekrar eden bulgu.

---

## Genel sonuç

1. **RF-DETR (LayerNorm) batch boyutuna karşı tamamen dayanıklı** — hem epoch sayısına hem batch boyutuna karşı en öngörülebilir, en güvenilir mimari olmaya devam ediyor.
2. **YOLO ailesi (BatchNorm) batch=1'e karşı öngörülemez tepki veriyor** — bazen (YOLO26x, her iki epoch) hedef kamerada belirgin iyileşme, bazen (YOLOv8x-50ep) orijinal sorunu (halüsinasyon) başka bir aşırılığa (neredeyse-sıfır tespit) çeviriyor. Bu, "batch=1 kötüdür/iyidir" gibi tek yönlü bir genelleme yapılamayacağını, etkinin model/epoch kombinasyonuna özgü olduğunu gösteriyor.
3. **Formal validation metrikleri (mAP, precision, recall) hiçbir modelde hedef kamera davranışını önceden haber vermiyor** — gerçek video testi hâlâ vazgeçilmez.
4. **Açık kalan iş:** Video1.mp4'teki YOLOv8x-50ep (0.23) ve YOLO26x (2.24/2.59) sonuçlarının görsel doğrulaması henüz yapılmadı — çıktı videolar (`inference_tests/Video1/Video1_detected.mp4`, ilgili model klasörlerinde) izlenmeden bu sayıların "gerçek tespit" mi "gürültü/kaçırma" mı olduğu kesinleşmeyecek.
