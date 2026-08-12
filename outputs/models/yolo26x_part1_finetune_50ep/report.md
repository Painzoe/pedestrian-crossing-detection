# Model Raporu — yolo26x_part1_finetune_50ep

*Bu dosya `outputs/models/yolo26x_part1_finetune_50ep/report.md` konumunda duruyor.*

**Ne eğitildi:** `yolo26x.pt`, `part_1` verisiyle **50 epoch** fine-tune edildi — [25 epoch'luk denemenin](../yolo26x_part1_finetune/report.md) tekrarı, epoch sayısı 2 katına çıkarıldı.

---

## Sonuç özeti (best.pt — epoch 50, final)

| Metrik | 25 epoch | 50 epoch |
|---|---|---|
| Precision | 0.663 | 0.704 |
| Recall | 0.713 | 0.767 |
| mAP50 | 0.706 | 0.746 |
| mAP50-95 | 0.394 | 0.431 |

Kendi validation setinde YOLOv8x'te gördüğümüze benzer şekilde her şey "iyi" görünüyor — ama biliyoruz ki bu tek başına güvenilir değil.

---

## Inference Testleri

| Video | Ort. kişi/kare (conf=0.15) | Ort. kişi/kare (conf=0.3) | Yorum |
|---|---|---|---|
| `Video1.mp4` (hedef kamera) | **1.35** | *(test edilmedi)* | Görsel kontrol: bilinen karede (2 gerçek kişi) YİNE HİÇBİRİNİ bulamadı, sadece köşede belirsiz 1 düşük-güvenli kutu var. Düşük sayı "temiz" değil, "kaçırma". |
| `video01.mp4` | 9.44 | *(test edilmedi)* | — |
| `part_1.mp4` (kendi verisi) | 23.52 | **18.56** | Eşiği artırmak biraz azalttı ama YOLOv8x'teki gibi net bir düzelme değil (YOLOv8x'te 20.60→11.66, neredeyse 25ep seviyesine inmişti; burada 23.52→18.56, hâlâ yüksek) |

**Video1.mp4 karşılaştırması (25 vs 50 epoch):**

| | 25 epoch | 50 epoch |
|---|---|---|
| Ort. kişi/kare | 0.32 | 1.35 |
| Bilinen karede gerçek kişileri buluyor mu? | Hayır | Hayır |

50 epoch'ta sayı yükseldi (0.32→1.35) ama bu "düzelme" değil — hâlâ gerçek insanları kaçırıyor, sadece başka (muhtemelen yanlış) yerlerde birkaç düşük güvenli kutu daha üretmiş.

---

## Genel değerlendirme

YOLO26x, 50 epoch'ta ne YOLOv8x-50ep'in çarpıcı overfitting patlamasını (36.24) yaşadı, ne de gerçek bir düzelme gösterdi. `part_1.mp4`'te (kendi verisi) hem 25 hem 50 epoch'ta YOLOv8x'ten daha fazla kutu üretiyor (18-24 arası), `Video1.mp4`'te ise (hedef kamera) her iki epoch sayısında da gerçek kişileri güvenilir şekilde bulamıyor. Genel sonuç değişmedi: **RF-DETR hâlâ en güvenilir seçim.** Detaylar için [COMPARISON.md](../COMPARISON.md).

*(Eğitim grafikleri için `training/` klasörüne, [YOLO26x 25 epoch raporundaki](../yolo26x_part1_finetune/report.md) görsel açıklamalarına bakabilirsin — yapı aynı.)*
