import cv2

image = cv2.imread("image.png")  # BGR
image[:, :, 0] = 0
image[:, :, 1] = 0
cv2.imshow("Image", image)
cv2.waitKey(0)