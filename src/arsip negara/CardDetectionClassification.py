import cv2
import numpy as np
import os

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def nothing(x):
    pass

def load_templates(dataset_dir):

    all_templates = {}

    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        template_imgs = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue    

                # Preprocessing: Thresholding
                _, thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                template_imgs.append(thresh_img)

        if template_imgs:
            all_templates[folder_name] = template_imgs
            print(f"[LOADED] {len(template_imgs)} templates for '{folder_name}'")

    print(f"[TOTAL] Loaded templates for {len(all_templates)} card types.")
    return all_templates

# def classify_card(warped_card_img, all_templates):
#     """
#     Mengklasifikasikan kartu utuh menggunakan template matching
#     dengan membandingkannya dengan 52 template kartu utuh.
#     """
#     # Pra-proses kartu dari kamera agar sesuai dengan template
#     # (Template sudah di-grayscale dan di-threshold saat dimuat)
#     warped_gray = cv2.cvtColor(warped_card_img, cv2.COLOR_BGR2GRAY)
    
#     # Gunakan nilai threshold yang SAMA seperti saat memuat template
#     _, warped_thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
#     best_match = ("UNKNOWN", 0.0)
    
#     # Loop melalui 52 template dan temukan yang paling cocok
#     for card_name, template_img in all_templates.items():
#         # Pastikan template dan gambar memiliki ukuran yang sama
#         # (Seharusnya sudah, tapi resize adalah pengaman)
#         if warped_thresh.shape != template_img.shape:
#              # Ini seharusnya tidak terjadi jika card_width_warped konsisten
#              template_img = cv2.resize(template_img, (warped_thresh.shape[1], warped_thresh.shape[0]))

#         res = cv2.matchTemplate(warped_thresh, template_img, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, _ = cv2.minMaxLoc(res)
        
#         if max_val > best_match[1]:
#             best_match = (card_name, max_val)

#     classified_card = best_match[0]
#     confidence = best_match[1]
    
#     # Ambang batas kepercayaan, bisa Anda sesuaikan
#     if confidence < 0.6: 
#         classified_card = "UNKNOWN"
        
#     return classified_card, confidence

def classify_card(warped_card_img, all_templates):
    """
    Mengklasifikasikan kartu utuh menggunakan template matching
    dengan membandingkannya terhadap semua template per jenis kartu.
    """
    warped_gray = cv2.cvtColor(warped_card_img, cv2.COLOR_BGR2GRAY)
    _, warped_thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    best_match = ("UNKNOWN", 0.0)

    # Loop untuk setiap jenis kartu (misal '2_diamond', 'Q_spade', dst)
    for card_name, template_list in all_templates.items():
        for template_img in template_list:  # Loop setiap gambar template di kelas itu
            if not isinstance(template_img, np.ndarray):
                continue

            # Pastikan ukuran sama
            if warped_thresh.shape != template_img.shape:
                template_resized = cv2.resize(template_img, (warped_thresh.shape[1], warped_thresh.shape[0]))
            else:
                template_resized = template_img

            # Template Matching
            result = cv2.matchTemplate(warped_thresh, template_resized, cv2.TM_CCOEFF_NORMED)
            cv2.imshow("Template Match Debug", result)
            _, score, _, _ = cv2.minMaxLoc(result)

            # Simpan jika lebih baik dari sebelumnya
            if score > best_match[1]:
                best_match = (card_name, score)

    classified_card = best_match[0]
    confidence = best_match[1]

    # Ambang batas kepercayaan minimum
    if confidence < 0.6:
        classified_card = "UNKNOWN"

    return classified_card, confidence


cap = cv2.VideoCapture(1)

all_templates = load_templates("dataset")

kernel = np.ones((5,5),np.uint8)

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 60, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 30, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 40, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 100, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    ret, img = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask_green = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_cards = cv2.bitwise_not(mask_green)

    mask_opened = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel)
    mask_opclo = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    res_opclo = cv2.bitwise_and(img, img, mask=mask_opclo)

    contours, _ = cv2.findContours(mask_opclo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_value = img.copy()
    grayscale_value_mask = img.copy()

    detected_count = 0

    c_contour = []
    for contour in contours:
        if cv2.contourArea(contour) > 3000:
            c_contour.append(contour)

    if c_contour:
        for contour in c_contour: 
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                src_pts = order_points(approx.reshape(4, 2))
                width, height = 200, 300
                dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(img, matrix, (width, height))
                best_card, confidence = classify_card(warped, all_templates)
                cv2.putText(warped, f"{best_card} ({confidence:.2f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow("Detected", warped)

            cv2.drawContours(card_value, [contour], -1, (0, 255, 0), 2)
    
    cv2.imshow('Mask', mask_opclo)
    
    #cv2.imshow('Opened-Closed', cv2.flip(res_opclo, 1))
    cv2.imshow('Nilai', cv2.flip(card_value, 1))
    #cv2.imshow('Value Mask', cv2.flip(grayscale_value_mask, 1))
    

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()