import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # sort.py dosyasını dahil edin

# Load a YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')

# Initialize SORT tracker
tracker = Sort()

def detect_lane(image):
    # Gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenarları algıla (Canny kenar tespiti kullanarak)
    edges = cv2.Canny(blurred, 350, 400)

    # ROI (Region of Interest) belirle
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 1),
        (width, height * 1),
        (width, height * 0.001),
        (0, height * 0.001),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough dönüşümü ile çizgileri algıla
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=150)

    return lines

# Video dosyasını açma
video_path = "road.mp4"
cap = cv2.VideoCapture(video_path)

frame_id = 0
previous_positions = {}

# Videoyu kare kare işleme
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Her kare için yol tespiti yap
    lines = detect_lane(frame)

    # Şeritleri orijinal frame üzerinde çiz
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Görüntü boyutlarını al
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # YOLO modelini kullanarak tahmin yapma
    results = model.predict(source=frame, show=False)

    # Tespit edilen nesnelerin bilgilerini al
    detections = []
    for result in results:
        boxes = result.boxes  # Get boxes object
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
            confidence = box.conf[0]
            if confidence > 0.5:  # Güven eşiği
                detections.append([x1, y1, x2, y2, confidence])

    # Takip için SORT'a tespitleri gönder
    if len(detections) > 0:
        tracks = tracker.update(np.array(detections))
    else:
        tracks = tracker.update()

    # Koordinat sistemini çiz
    cv2.line(frame, (0, 0), (width, 0), (255, 0, 0), 2)  # Yatay merkez çizgisi
    cv2.line(frame, (0, 0), (0, height), (255, 0, 0), 2)  # Dikey merkez çizgisi

    # Her izlenen nesne için
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        obj_center_x = int((x1 + x2) / 2)
        obj_center_y = int((y1 + y2) / 2)

        # Yeni koordinat sistemine göre Y koordinatlarını hesapla
        new_center_y = obj_center_y - center_y
        label = f"ID {track_id}"

        # Araba etrafına dikdörtgen çizme
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Hız ve yön hesaplama
        if frame_id > 0:
            if track_id in previous_positions:
                prev_pos = previous_positions[track_id]
                curr_pos = (obj_center_x, new_center_y)

                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                direction = (dx, dy)

                distance = np.sqrt(dx**2 + dy**2)
                fps = cap.get(cv2.CAP_PROP_FPS)
                speed = distance * fps  # assuming distance is in pixels and speed is in pixels/second

                print(f"Araba {label}: Yön: {direction}, Hız: {speed:.2f} piksel/s")

            previous_positions[track_id] = (obj_center_x, new_center_y)
        else:
            previous_positions[track_id] = (obj_center_x, new_center_y)

    frame_id += 1

    # Güncellenmiş frame'i gösterme
    cv2.imwrite("Frame.mp4", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çıkış
        break

cap.release()
cv2.destroyAllWindows()
