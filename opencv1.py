import cv2
img=cv2.imread("test_images/p2.png")
#print("Image shape:", img.shape)
cv2.imshow("Image1", img)
cv2.waitKey()
cv2.destroyAllWindows()

