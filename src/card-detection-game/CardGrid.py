import cv2
import numpy as np
import os
from keras.models import load_model

def save_cards_to_txt(cards, combo, filename="cards_state.txt"):
    """
    Menyimpan hasil deteksi ke file txt:
    COMBO <nama_combo>
    CARDS <label1> <label2> ...
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"COMBO {combo}\n")
        if cards:
            f.write("CARDS " + " ".join(cards) + "\n")
        else:
            f.write("CARDS\n")


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

# Filter rasio kartu
def is_valid_card_ratio(width, height, err=0.75):
    """
    Mengecek apakah rasio width:height mendekati 2:3.
    err = toleransi error, default 25%
    """
    target_ratio = 2/3     # ≈ 0.6666
    ratio = width / height

    return (target_ratio * (1 - err) <= ratio <= target_ratio * (1 + err))

# Card parser
def parse_card(label):
    """
    Input: '10_heart'
    Output: rank(int), suit(str)
    """
    rank_str, suit = label.split("_")

    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }

    return rank_map[rank_str], suit

# Hand detector
# def detect_hand_limited(cards):
#     """
#     cards = list of card labels, contoh:
#     ["10_heart", "J_heart", "Q_heart", "K_heart", "A_heart"]
#     """

#     # Butuh minimal 2 kartu untuk pair / double pair
#     if len(cards) < 2:
#         return "Tidak Valid"

#     # Parsing rank & suit
#     ranks = []
#     suits = []

#     for c in cards:
#         r, s = parse_card(c)
#         ranks.append(r)
#         suits.append(s)

#     ranks_sorted = sorted(ranks)
#     from collections import Counter
#     rank_count = Counter(ranks)
#     suit_count = Counter(suits)

#     # ------------------------
#     # 1. Pair
#     # ------------------------
#     if list(rank_count.values()).count(2) == 1 and len(cards) == 2:
#         return "Pair"

#     # ------------------------
#     # 2. Double Pair
#     # ------------------------
#     if list(rank_count.values()).count(2) == 2 and len(cards) == 4:
#         return "Double Pair"

#     # ------------------------
#     # 3. Straight Flush / Royal Flush
#     # hanya untuk 5 kartu
#     # ------------------------
#     if len(cards) == 5:

#         # cek flush
#         is_flush = (len(suit_count) == 1)

#         # cek straight
#         uniq = sorted(set(ranks_sorted))
#         is_straight = False
#         if len(uniq) == 5 and uniq[-1] - uniq[0] == 4:
#             is_straight = True

#         if is_flush and is_straight:
#             if uniq == [10, 11, 12, 13, 14]:
#                 return "Royal Flush"
#             return "Straight Flush"

#     # ------------------------
#     # Tidak memenuhi aturan apapun
#     # ------------------------
#     return "Tidak Valid"

lookup_table = {

    # --- Pair ---
    "Pair": lambda ranks, suits: (
        len(ranks) == 2 and
        len(set(ranks)) == 1
    ),

    # --- Double Pair ---
    "Double Pair": lambda ranks, suits: (
        len(ranks) == 4 and
        sorted([ranks.count(x) for x in set(ranks)]) == [2, 2]
    ),

    # --- Royal Flush ---
    "Royal Flush": lambda ranks, suits: (
        len(ranks) == 5 and
        len(set(suits)) == 1 and
        sorted(ranks) == [10, 11, 12, 13, 14]
    ),

    # --- Straight Flush ---
    "Straight Flush": lambda ranks, suits: (
        len(ranks) == 5 and
        len(set(suits)) == 1 and
        len(set(ranks)) == 5 and
        max(ranks) - min(ranks) == 4
    ),
}


def detect_hand_LUT(cards):
    """
    Menggunakan lookup table untuk mengidentifikasi kombinasi Capsa.
    Hanya: Pair, Double Pair, Straight Flush, Royal Flush.
    Jika tidak cocok → "Tidak Valid".
    """

    if len(cards) < 2:
        return "Tidak Valid"

    # Parse ranks & suits
    ranks = []
    suits = []

    for c in cards:
        r, s = parse_card(c)
        ranks.append(r)
        suits.append(s)

    # Coba cocokkan dengan lookup table
    for combo_name, rule in lookup_table.items():
        if rule(ranks, suits):
            return combo_name

    return "Tidak Valid"

# def load_templates(dataset_dir):

#     all_templates = {}

#     for folder_name in os.listdir(dataset_dir):
#         folder_path = os.path.join(dataset_dir, folder_name)
#         if not os.path.isdir(folder_path):
#             continue

#         template_imgs = []
#         for file_name in os.listdir(folder_path):
#             if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 img_path = os.path.join(folder_path, file_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is None:
#                     continue    

#                 # Preprocessing: Thresholding
#                 _, thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#                 template_imgs.append(thresh_img)

#         if template_imgs:
#             all_templates[folder_name] = template_imgs
#             print(f"[LOADED] {len(template_imgs)} templates for '{folder_name}'")

#     print(f"[TOTAL] Loaded templates for {len(all_templates)} card types.")
#     return all_templates

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

# def classify_card(warped_card_img, all_templates):
#     """
#     Mengklasifikasikan kartu utuh menggunakan template matching
#     dengan membandingkannya terhadap semua template per jenis kartu.
#     """
#     warped_gray = cv2.cvtColor(warped_card_img, cv2.COLOR_BGR2GRAY)
#     _, warped_thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#     best_match = ("UNKNOWN", 0.0)

#     # Loop untuk setiap jenis kartu (misal '2_diamond', 'Q_spade', dst)
#     for card_name, template_list in all_templates.items():
#         for template_img in template_list:  # Loop setiap gambar template di kelas itu
#             if not isinstance(template_img, np.ndarray):
#                 continue

#             # Pastikan ukuran sama
#             if warped_thresh.shape != template_img.shape:
#                 template_resized = cv2.resize(template_img, (warped_thresh.shape[1], warped_thresh.shape[0]))
#             else:
#                 template_resized = template_img

#             # Template Matching
#             result = cv2.matchTemplate(warped_thresh, template_resized, cv2.TM_CCOEFF_NORMED)
#             cv2.imshow("Template Match Debug", result)
#             _, score, _, _ = cv2.minMaxLoc(result)

#             # Simpan jika lebih baik dari sebelumnya
#             if score > best_match[1]:
#                 best_match = (card_name, score)

#     classified_card = best_match[0]
#     confidence = best_match[1]

#     # Ambang batas kepercayaan minimum
#     if confidence < 0.6:
#         classified_card = "UNKNOWN"

#     return classified_card, confidence


cap = cv2.VideoCapture(1)

# ====== CONFIG CNN KARTU ======
DirektoriDataSet = "dataset"  # folder yang berisi 2_club, 2_diamond, dst

# LabelKelas diambil dari nama folder seperti di program training
LabelKelas = tuple(
    sorted(
        [
            d
            for d in os.listdir(DirektoriDataSet)
            if os.path.isdir(os.path.join(DirektoriDataSet, d))
        ]
    )
)
print("LabelKelas:", LabelKelas)

# Load model CNN yang sudah kamu latih (hasil TrainingCNN)
ModelCNN = load_model("Hasil.h5")
print("Model CNN loaded.")
IMG_SIZE = 128  


# all_templates = load_templates("dataset")

kernel = np.ones((5,5),np.uint8)

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 60, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 30, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 40, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 100, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

def KlasifikasiCitraTunggal(warped_card_img, LabelKelas, ModelCNN, threshold=0.5):
    """
    Mirip Listing 8.2 tapi untuk SATU citra kartu (BGR, dari OpenCV).
    Output: nama kelas (mis. '2_club') dan confidence.
    """
    img = cv2.resize(warped_card_img, (IMG_SIZE, IMG_SIZE))

    # Normalisasi 0..1, tipe float32 
    img = np.asarray(img) / 255.0
    img = img.astype("float32")

    # Bentuk batch 1 citra
    X = np.expand_dims(img, axis=0)  # shape: (1, 128, 128, 3)

    # Prediksi
    hs = ModelCNN.predict(X, verbose=0)[0]  # vektor probabilitas (n_kelas,)

    # Menentukan kelas dengan ambang batas
    if hs.max() > threshold:
        idx = int(np.argmax(hs))
        kelas = LabelKelas[idx]
    else:
        idx = -1
        kelas = "UNKNOWN"

    confidence = float(hs.max())
    return kelas, confidence


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

    result_frame = img.copy()

    warped_cards = []
    detected_count = 0
    recognized_cards = []

    for contour in contours:
        if cv2.contourArea(contour) < 3000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            if not is_valid_card_ratio(w, h):
                continue
            src_pts = order_points(approx.reshape(4, 2))
            width, height = 200, 300

            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, matrix, (width, height))
            if warped.shape[0] < warped.shape[1]:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)


            # Klasifikasi kartu menggunakan CNN
            best_card, confidence = KlasifikasiCitraTunggal(
                warped,
                LabelKelas,
                ModelCNN,
                threshold=0.5   # bisa kamu naik/turunkan
            )

            # Simpan kartu yang dikenali untuk deteksi combo
            if best_card != "UNKNOWN":
                recognized_cards.append(best_card)

            cv2.putText(
                warped,
                f"{best_card} ({confidence:.2f})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            warped_cards.append(warped)
            cv2.drawContours(result_frame, [approx], -1, (0, 255, 0), 2)
            detected_count += 1

    # DETEKSI COMBO KARTU
    if recognized_cards:
        combo = detect_hand_LUT(recognized_cards)
        print("Combo:", combo)

        # Simpan ke TXT untuk “API” ke game raylib
        save_cards_to_txt(recognized_cards, combo, "cards_state.txt")
        
        # Tampilkan di layar
        cv2.putText(result_frame, f"Hand: {combo}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)


    # Warped GRID
    if warped_cards:

        standard_h = 300
        standard_w = 200

        # Meresize semua warped cards ke ukuran standar
        resized_cards = [cv2.resize(wc, (standard_w, standard_h)) for wc in warped_cards]

        max_per_row = 5   # Maksimal 5
        rows = []

        for i in range(0, len(resized_cards), max_per_row):
            row_images = resized_cards[i:i + max_per_row]

            # Menjadi HITAM jika kurang dari max_per_row
            if len(row_images) < max_per_row:
                deficit = max_per_row - len(row_images)
                black = np.zeros_like(row_images[0])
                row_images.extend([black] * deficit)

            # Build this row (always safe)
            row = cv2.hconcat(row_images)
            rows.append(row)

        # Secara vertikal gabung semua row
        grid = cv2.vconcat(rows)

        cv2.imshow("Warped Grid", grid)
    
    # TUNJUKIN KARTU 
    cv2.putText(result_frame, f"Detected: {detected_count} cards", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
    cv2.imshow("Detected Cards", result_frame)

    
    #cv2.imshow('Mask', mask_opclo)
    
    #cv2.imshow('Opened-Closed', cv2.flip(res_opclo, 1))
    #cv2.imshow('Nilai', cv2.flip(card_value, 1))
    #cv2.imshow('Value Mask', cv2.flip(grayscale_value_mask, 1))
    

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()