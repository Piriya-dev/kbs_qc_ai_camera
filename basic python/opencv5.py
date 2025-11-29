import cv2

img=cv2.imread("test_images/p2.png")

#Over write Line on img
cv2.line(img, (0, 0), (450, 450), (255, 0, 0), 10,lineType=cv2.LINE_AA)
cv2.imshow("Image1", img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.circle(img, (400, 50), 50, (0, 255, 0), -1,lineType=cv2.LINE_AA)
cv2.imshow("Image1", img)
cv2.waitKey()
cv2.destroyAllWindows()