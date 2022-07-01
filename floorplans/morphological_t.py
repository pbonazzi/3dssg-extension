import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

toimg = os.path.join("path_to_img")
img = cv2.imread(toimg, 0)
plt.imshow(img)
plt.show()
f, axarr = plt.subplots(2, 2, figsize=(8, 8))

# plt.figure(1)
# plt.subplot(211)
kernel = np.ones((30, 30), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
# cv2.imwrite('erosion.jpg', erosion)
# plt.imshow(erosion)

# plt.subplot(212)
dilation = cv2.dilate(img, kernel, iterations=1)
# cv2.imwrite('dilation.jpg', dilation)
# plt.imshow(dilation)

# plt.subplot(213)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imwrite('opening.jpg', opening)
# plt.imshow(opening)

# plt.subplot(214)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

axarr[0, 0].imshow(closing)
axarr[0, 0].title.set_text("Closing")

axarr[0, 1].imshow(opening)
axarr[0, 1].title.set_text("Opening")

axarr[1, 0].imshow(erosion)
axarr[1, 0].title.set_text("Erosion")

axarr[1, 1].imshow(dilation)
axarr[1, 1].title.set_text("Dilation")

plt.show()
