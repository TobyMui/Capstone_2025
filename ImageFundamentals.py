import cv2

img = cv2.imread('assets/airplane.jpg', -1)
print(img.shape)

'''In the case of airplane.jpg it is represent by (4000,6000,3)
    4000 is the number of rows or height
    6000 is the width or columsn 
    3 is the number of channels per pixel.'''
