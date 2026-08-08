"""List available camera indices (finds iPhone Continuity Camera too)."""

import cv2

N_INDICES = 5


def main():
    for i in range(N_INDICES):
        cap = cv2.VideoCapture(i)
        print(f"index {i}: {'available' if cap.isOpened() else 'none'}")
        cap.release()


if __name__ == "__main__":
    main()
