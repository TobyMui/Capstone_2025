import cv2

'''In this program, we access an image in our assets file and open it
    This program also teaches how to write an image'''
img = cv2.imread('assets/1I8A0018.jpg',1)
img = cv2.resize(img, (0,0),fx = 0.25, fy = 0.25)

cv2

img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


