import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert


img = []

def reds_to_cyan(image):
        #Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #Define range of red color in HSV
    rd_lb = np.array([-10,50,50])
    rd_ub = np.array([10,255,255])

        #Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, rd_lb, rd_ub)

        #Change color at masked region to cyan 
    image[mask == 255] = [255,255,0]
    return image

def solidify(image, flag = 1):
        #Solidify outer border(1) and inner void(2)
    if flag != 1 and flag != 2:
        #cv2.imshow('out', image)
        cv2.imwrite('temp/solidified.jpg', image)
        return

        #Convert to Blurred Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))

        #Threshold lighter areas
    ret, thresh = cv2.cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

        #Find contours
    _, cnt, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Select Biggest Contour for outer and Smallest for inner
    if flag == 1:   c = max(cnt, key = cv2.contourArea)
    else:           c = min(cnt, key = cv2.contourArea)

        #Draw an empty black Mask
    mask = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

        #Draw solid Contour on the Mask
    cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        #Superimpose the Mask on the Image (Area to be solidified is white on mask)
    if flag == 1:   mask = invert(mask)
    image = cv2.bitwise_or(mask, image)
    
        #Dilate out thin black lines
    kernel = np.ones((3,3),np.uint8)
    image = cv2.dilate(image, kernel, iterations = 1)
    solidify(image, flag + 1)

def main():
    img = cv2.imread("../mazes/maze-2.jpg")
    #img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_AREA)
    img = reds_to_cyan(img)
    solidify(img)

if __name__ == "__main__":
    main()
