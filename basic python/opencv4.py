import cv2

def convert_to_grey(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def resize_image(img, width, height):
    resized_img = cv2.resize(img, (width, height))
    return resized_img

def main():
    img = cv2.imread("test_images/p2.png")
    grey_img = convert_to_grey(img)
    resized_img = resize_image(img, 800, 600)

    cv2.imshow("Original Image", img)
    cv2.imshow("Grey Image", grey_img)
    cv2.imshow("Resized Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()