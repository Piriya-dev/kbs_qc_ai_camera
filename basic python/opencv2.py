import cv2
img=cv2.imread("test_images/p2.png")
#save image to new folder
cv2.imwrite("output_images/p2_output.png", img)
print("Image saved to output_images/p2_output.png")

