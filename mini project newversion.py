import cv2
import torch
from ultralytics import YOLO

input_video  = r"C:\Users\user\Downloads\test3.mp4"
output_video = r"C:\Users\user\Downloads\output_counted3.mp4"

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ------------------------------
# Load YOLO11
# ------------------------------
model = YOLO("yolo11n.pt").to(device)

# ------------------------------
# Video IO
# ------------------------------
cap = cv2.VideoCapture(input_video)
assert cap.isOpened(), "❌ Input video not opened"

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

assert out.isOpened(), "❌ VideoWriter not opened"

# ------------------------------
# Counting setup
# ------------------------------
count_line_y = height // 2
count = 0
tracked_ids = set()

# ------------------------------
# Processing loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.25,
        iou=0.5,
        tracker="bytetrack.yaml",
        device=0
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cy = int((y1 + y2) / 2)

            # Count crossing
            if cy > count_line_y and obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                count += 1

            # Draw bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {int(obj_id)}",
                        (x1, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    # Draw line + count
    cv2.line(frame, (0, count_line_y),
             (width, count_line_y), (0,0,255), 2)

    cv2.putText(frame, f"Count: {count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (255,0,0), 3)

    # 🔴 SAVE FRAME
    out.write(frame)

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved at:", output_video)
