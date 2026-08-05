import cv2
import os

video_path = "videos/Video1.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video açılamadı!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"✅ Video başarıyla açıldı")
print(f"FPS: {fps}")
print(f"Toplam frame: {total_frames}")
print(f"Süre: {duration:.2f} saniye")

second = 0

while second < duration:

    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)

    ret, frame = cap.read()

    if not ret:
        break

    filename = os.path.join(
        output_dir,
        f"frame_{second:04d}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(f"Kaydedildi: {filename}")

    second += 1

cap.release()

print("🎉 Frame extraction tamamlandı!")