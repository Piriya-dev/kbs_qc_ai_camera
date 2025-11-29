import cv2
img=cv2.imread("test_images/p2.png")
grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print("Image shape:", img.shape)
cv2.imshow("Image1", img)
cv2.imshow("Grey Image", grey)
cv2.waitKey()
cv2.destroyAllWindows()

