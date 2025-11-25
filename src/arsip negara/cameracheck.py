import cv2
def list_cameras(max_idx=8):
    available = []
    for i in range(max_idx+1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # try DirectShow backend (works well on Windows)
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            available.append(i)
        cap.release()
    return available

print("Available camera indices:", list_cameras(8))
