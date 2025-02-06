from ultralytics import YOLO
import cv2
import os

# Nastavenie modelu
yolo_model = YOLO('oceaneye-ai.pt')  # Nahraď "model.pt" svojim modelom

# Cesta k vstupnému videu
input_video_path = 'VID10.mp4'  # Nahraď svojím súborom
output_video_path = "output_filtered.mp4"

# Otvorenie vstupného videa
cap = cv2.VideoCapture(input_video_path)

# Získanie základných parametrov videa
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Nastavenie výstupného videa
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detekcia objektov v aktuálnom frame
    results = yolo_model(frame)
    detected = False
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Index detekovanej triedy
            label = yolo_model.names[cls]  # Názov triedy
            
            if label.lower() == "mrenka":
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordináty rámčeka
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kreslenie rámčeka
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        if detected:
            break
    
    # Ak bola detekovaná "mrenka", uložíme frame do výstupného videa
    if detected:
        out.write(frame)

# Uvoľnenie zdrojov
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video uložené: {output_video_path}")
