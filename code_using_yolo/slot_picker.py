import cv2
import pickle
import os

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = "carParkImg.png"
SLOT_FILE = "parking_slots.pkl"

# Slot size (TUNED for aerial/top-down view)
SLOT_WIDTH = 107
SLOT_HEIGHT = 48

# -------------------------------
# LOAD EXISTING SLOTS (if any)
# -------------------------------
if os.path.exists(SLOT_FILE):
    with open(SLOT_FILE, "rb") as f:
        slots = pickle.load(f)
else:
    slots = []

# -------------------------------
# MOUSE CALLBACK
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    global slots

    if event == cv2.EVENT_LBUTTONDOWN:
        slots.append((x, y, SLOT_WIDTH, SLOT_HEIGHT))
        print(f"Added slot at ({x}, {y})")

        with open(SLOT_FILE, "wb") as f:
            pickle.dump(slots, f)

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, (sx, sy, w, h) in enumerate(slots):
            if sx < x < sx + w and sy < y < sy + h:
                removed = slots.pop(i)
                print(f"Removed slot at ({removed[0]}, {removed[1]})")

                with open(SLOT_FILE, "wb") as f:
                    pickle.dump(slots, f)
                break

# -------------------------------
# LOAD IMAGE
# -------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("❌ carParkImg.png not found")

cv2.namedWindow("Slot Picker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Slot Picker", mouse_callback)

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    display = img.copy()

    for (x, y, w, h) in slots:
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow("Slot Picker", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
print("✅ Slot selection finished")
