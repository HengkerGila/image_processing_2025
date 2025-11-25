import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-left
    rect[2] = pts[np.argmax(s)]   # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def extract_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cv2.imshow("Edges", edges)
    



    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # remove small contours and large contours
    contours = [cnt for cnt in contours if 30000 < cv2.contourArea(cnt) < 50000]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:

        # draw the contour on the original frame for debugging
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

        # put text on the frame
        cv2.putText(frame, f"Area: {cv2.contourArea(cnt):.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # Quadrilateral found
            pts = approx.reshape(4, 2)
            rect = order_points(pts)

            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            
            maxWidth = int(max(widthA, widthB))
            maxHeight = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))


            # Auto orient: keep card portrait
            if warped.shape[1] > warped.shape[0]:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            card_ratio = warped.shape[1] / warped.shape[0]

            if (card_ratio > 0.69 or card_ratio < 0.8):
                return warped

    return None


cap = cv2.VideoCapture(0)
container = np.zeros((400, 400, 3), dtype=np.uint8)
frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    card = extract_card(frame)
    card_ratio = card.shape[1] / card.shape[0] if card is not None else 0

    if (card is not None) :
        container = card.copy()
        print(f"Card detected with ratio: {container.shape[0]} {container.shape[1]} = {card_ratio:.2f}")

    cv2.imshow("Original", frame)
    cv2.imshow("Card Normalized", container)

    if cv2.waitKey(0) == ord('q'):
        frame_count += 1
        # continue for next frame
        continue
    elif cv2.waitKey(0) == ord('c'):
        # back to the previous frame
        frame_count -= 1
        if frame_count < 0:
            frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        continue

    cap.release()
    cv2.destroyAllWindows()