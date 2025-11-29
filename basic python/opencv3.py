import cv2
import convert_to_grey
img=cv2.imread("test_images/p2.png")

#grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resize_img = cv2.resize(img, (800, 600))
#grey = convert_to_grey.convert_to_grey(img)

#print("Image shape:", img.shape)
cv2.imshow("Image1", resize_img)
#cv2.imshow("Grey Image", grey)
cv2.waitKey()
cv2.destroyAllWindows()

#