import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../img/Tower2-2.jpg')

kernel = np.ones((5, 5)) / 25

hasil_konvolusi = cv2.filter2D(img, -1, kernel)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1), plt.imshow(img, cmap='gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(hasil_konvolusi, cmap='gray')
plt.title('Hasil Konvolusi'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()