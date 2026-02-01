import cv2
import pickle
import cvzone
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
VIDEO_PATH = "carPark.mp4"
SLOT_FILE = "parking_slots.pkl"

# Pixel-based thresholds (tuned)
FG_STRONG = 900
FG_MEDIUM = 600
EDGE_THRESHOLD = 60

# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------
model = YOLO("yolov8n.pt")
print("✅ YOLO model loaded")

# -------------------------------------------------
# LOAD PARKING SLOTS
# -------------------------------------------------
with open(SLOT_FILE, "rb") as f:
    slots = pickle.load(f)

print(f"✅ Loaded {len(slots)} parking slots")

# -------------------------------------------------
# VIDEO CAPTURE
# -------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Cannot open video")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def box_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (xB - xA) * (yB - yA) if xA < xB and yA < yB else 0


def is_occupied(roi, yolo_overlap):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(gray, 40, 120)

    fg_pixels = cv2.countNonZero(fg)
    edge_pixels = cv2.countNonZero(edges)

    # -------- Final Decision --------
    if fg_pixels > FG_STRONG:
        return True

    if fg_pixels > FG_MEDIUM and edge_pixels > EDGE_THRESHOLD:
        return True

    if yolo_overlap:
        return True

    return False

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    free_count = 0

    # ---------------- YOLO DETECTION ----------------
    vehicle_boxes = []
    results = model(frame, conf=0.4, stream=True, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] in ["car", "truck", "bus"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)

    # ---------------- SLOT CHECK ----------------
    for (x, y, w, h) in slots:
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        slot_box = (x, y, x + w, y + h)

        yolo_overlap = False
        for vb in vehicle_boxes:
            if box_overlap(slot_box, vb) > 0:
                yolo_overlap = True
                break

        occupied = is_occupied(roi, yolo_overlap)

        if occupied:
            color = (0, 0, 255)  # RED
        else:
            color = (0, 255, 0)  # GREEN
            free_count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ---------------- DISPLAY ----------------
    cvzone.putTextRect(
        frame,
        f"Vacant Slots: {free_count}/{len(slots)}",
        (50, 50),
        scale=2,
        thickness=3,
        offset=10
    )

    cv2.imshow("Parking Slot Detection (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------
# CLEANUP
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("🛑 Program terminated")
