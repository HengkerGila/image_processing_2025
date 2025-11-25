import cv2
import numpy as np

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

#img = cv2.imread('kartu.png')
cap = cv2.VideoCapture(1)

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

    # Filter contours by area
    c_contour = []
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            c_contour.append(contour)

    # The following block for sorting multiple cards is commented out as it is designed for static images
    # and may not be performant for live video. A simpler processing loop is used below.
    # processed_cards = []
    # if c_contour:
    #     b_box = [cv2.boundingRect(c) for c in c_contour]
    #     contours_and_boxes = sorted(zip(c_contour, b_box), key=lambda b: b[1][1])
    #     rows = []
    #     current_row = []
    #     if contours_and_boxes:
    #         ref_height = contours_and_boxes[0][1][3]
    #         current_row.append(contours_and_boxes[0][0])
    #         last_y = contours_and_boxes[0][1][1]
    #
    #         for i in range(1, len(contours_and_boxes)):
    #             contour, (x, y, w, h) = contours_and_boxes[i]
    #             if y > last_y + ref_height * 0.7:
    #                 current_row.sort(key=lambda c: cv2.boundingRect(c)[0])
    #                 rows.append(current_row)
    #                 current_row = [contour]
    #             else:
    #                 current_row.append(contour)
    #             last_y = y
    #
    #         if current_row:
    #             current_row.sort(key=lambda c: cv2.boundingRect(c)[0])
    #             rows.append(current_row)
    #
    #     flat_card_list = []
    #     for row_idx, row in enumerate(rows):
    #         for col_idx, contour in enumerate(row):
    #             value = (row_idx * 13) + col_idx
    #             flat_card_list.append({'contour': contour, 'value': value})
    #
    #     sorted_cards = sorted(flat_card_list, key=lambda c: c['value'])
    #
    #     for card_data in sorted_cards:
    #         contour = card_data['contour']
    #
    #         peri = cv2.arcLength(contour, True)
    #         geo = cv2.approxPolyDP(contour, 0.02 * peri, True)
    #
    #         if len(approx) == 4:
    #             src_pts = order_points(approx.reshape(4, 2))
    #             width, height = 250, 350
    #             dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')
    #             matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    #             warped = cv2.warpPerspective(img, matrix, (width, height))
    #
    #             card_data['warped'] = warped
    #             processed_cards.append(card_data)
    #
    #         cv2.drawContours(card_value, [contour], -1, (0, 255, 0), 3)
    #         grayscale_val = (card_data['value'] / 51.0) * 255
    #         color = (grayscale_val, grayscale_val, grayscale_val)
    #         cv2.drawContours(grayscale_value_mask, [contour], -1, color, -1)
    #         x, y, w, h = cv2.boundingRect(contour)
    #         label = f"{card_data['value'] / 51.0:.2f}"
    #         text_pos = (x + 10, y + 40)
    #         cv2.putText(card_value, label, text_pos, cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 3)

    # Simplified processing for live video
    if c_contour:
        for contour in c_contour:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                src_pts = order_points(approx.reshape(4, 2))
                width, height = 250, 350
                dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(img, matrix, (width, height))
                cv2.imshow('Warped Card', warped)

            cv2.drawContours(card_value, [contour], -1, (0, 255, 0), 3)


    cv2.imshow('Mask', mask_opclo)
    cv2.imshow('opened-closed', cv2.flip(res_opclo, 1))
    cv2.imshow('nilai', cv2.flip(card_value, 1))
    cv2.imshow('Value Mask', cv2.flip(grayscale_value_mask, 1))

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC key
        break

# The original static image display loop and spacebar functionality are commented out.
# card_index = -1
# while True:
#     key = cv2.waitKey(0) & 0xFF
#
#     if key == 27: # ESC key
#         break
#
#     if key == 32: # Spacebar
#         if not processed_cards:
#             continue
#
#         card_index = (card_index + 1) % len(processed_cards)
#         card_data = processed_cards[card_index]
#         contour = card_data['contour']
#         warped = card_data['warped']
#         x, y, w, h = cv2.boundingRect(contour)
#         og_snippet = img[y:y+h, x:x+w]
#
#         h_w, w_w, _ = warped.shape
#         h_o, w_o, _ = og_snippet.shape
#         scale = h_w / h_o
#         new_w = int(w_o * scale)
#         resized_og = cv2.resize(og_snippet, (new_w, h_w))
#
#         compare_view = cv2.hconcat([resized_og, warped])
#
#         cv2.imshow('compare', compare_view)

cap.release()
cv2.destroyAllWindows()