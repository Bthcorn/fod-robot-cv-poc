"""List available camera indices (finds iPhone Continuity Camera too)."""

import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    print(f"index {i}: {'available' if ok else 'none'}")
    cap.release()
