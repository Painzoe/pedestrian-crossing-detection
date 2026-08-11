"""
extract_frames.py
-------------------
data/raw_videos/ klasorundeki TUM video dosyalarini tek tek isler,
her birini SANIYEDE 1 KARE olacak sekilde JPG fotograflara boler.

Her video kendi ismiyle ayni isimde bir alt klasore kaydedilir, ornek:
  data/raw_videos/castro_10dk.mp4
    -> data/dataset_frames/castro_10dk/frame_0000.jpg
    -> data/dataset_frames/castro_10dk/frame_0001.jpg
    -> ...

Boylece hangi frame'in hangi videodan geldigini karistirmadan takip
edebiliyoruz - ileride bir sorun cikarsa (orn. bir videonun kalitesi
kotu) o videonun frame'lerini tek seferde bulup cikarmak kolay olur.
"""

import cv2
import os

# ============================================================
# 1) AYARLAR
# ============================================================

# Ham videolarin bulundugu klasor
RAW_VIDEOS_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/data/raw_videos"

# Cikan frame'lerin kaydedilecegi ana klasor (her video kendi alt klasorunu alacak)
OUTPUT_FRAMES_DIR = "/home/painzoe/PycharmProjects/pedestrian-crossing-detection/data/dataset_frames"

# Saniyede kac kare alinacak. 1 = saniyede 1 kare (senin secimin).
FRAMES_PER_SECOND = 1

# Islenecek video uzantilari
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")


# ============================================================
# 2) VIDEO LISTESINI HAZIRLA
# ============================================================

if not os.path.exists(RAW_VIDEOS_DIR):
    raise FileNotFoundError(f"'{RAW_VIDEOS_DIR}' klasoru bulunamadi.")

all_files = os.listdir(RAW_VIDEOS_DIR)
video_files = sorted([
    f for f in all_files
    if f.lower().endswith(VIDEO_EXTENSIONS)
])

if len(video_files) == 0:
    raise FileNotFoundError(f"'{RAW_VIDEOS_DIR}' klasorunde hic video dosyasi bulunamadi.")

print(f"Toplam {len(video_files)} video bulundu: {video_files}\n")

os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)


# ============================================================
# 3) HER VIDEO ICIN FRAME CIKAR
# ============================================================

# Genel ozet icin: her videodan kac frame cikti, sonda raporlayacagiz
summary = {}

for video_name in video_files:
    video_path = os.path.join(RAW_VIDEOS_DIR, video_name)

    # Video dosya adindan uzantiyi cikarip klasor ismi olarak kullaniyoruz
    # ornek: "castro_10dk.mp4" -> "castro_10dk"
    video_basename = os.path.splitext(video_name)[0]
    video_output_dir = os.path.join(OUTPUT_FRAMES_DIR, video_basename)
    os.makedirs(video_output_dir, exist_ok=True)

    # Videoyu OpenCV ile ac
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [UYARI] '{video_name}' acilamadi, atlaniyor.")
        continue

    # Videonun kendi orijinal FPS'ini (saniyede kac kare oldugunu) oku.
    # Bu onemli: video 30 fps de olsa 25 fps de olsa, biz "saniyede 1 kare"
    # istiyoruz, bu yuzden videonun gercek fps degerine gore kac karede
    # bir alacagimizi (frame_interval) hesaplamamiz lazim.
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        # Bazi bozuk/eksik videolarda fps bilgisi okunamayabilir.
        # Boyle bir durumda 30 fps varsayiyoruz (en yaygin deger).
        print(f"  [UYARI] '{video_name}' icin FPS okunamadi, 30 varsayiliyor.")
        video_fps = 30.0

    # Kac karede bir frame alacagimiz. Ornek: video 30 fps ise ve
    # saniyede 1 kare istiyorsak, her 30 karede bir frame kaydederiz.
    frame_interval = int(round(video_fps / FRAMES_PER_SECOND))

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"'{video_name}': {video_fps:.1f} fps, {total_frames_in_video} toplam kare")

    frame_index = 0       # videodaki gercek kare sayaci (her kare icin +1)
    saved_count = 0        # kac frame kaydettigimiz

    while True:
        ret, frame = cap.read()
        # ret=False -> video bitti demek, donguden cik
        if not ret:
            break

        # Sadece frame_interval'a tam bolunen karelerde kayit yapiyoruz.
        # Ornek: frame_interval=30 ise, 0. 30. 60. 90. ... kareler kaydedilir.
        if frame_index % frame_interval == 0:
            output_filename = f"frame_{saved_count:04d}.jpg"
            output_path = os.path.join(video_output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    summary[video_name] = saved_count
    print(f"  -> {saved_count} frame kaydedildi: {video_output_dir}\n")


# ============================================================
# 4) OZET
# ============================================================

print("=" * 50)
print("FRAME CIKARMA OZETI")
print("=" * 50)
total_frames = 0
for video_name, count in summary.items():
    print(f"  {video_name}: {count} frame")
    total_frames += count

print(f"\nToplam frame sayisi: {total_frames}")
print(f"Tum frame'ler burada: {OUTPUT_FRAMES_DIR}")