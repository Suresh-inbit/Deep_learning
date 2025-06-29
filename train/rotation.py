import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
# Read the images
template = cv.imread('mvtec_ad/grid/train/good/000.png', cv.IMREAD_GRAYSCALE)[:200, :200]
original = cv.imread('mvtec_ad/grid/train/good/005.png', cv.IMREAD_GRAYSCALE)
cv.imwrite("/dev/files/workspace/image.png", original)
# cv.waitKey(0)

def rotate_image(image, angle):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
 
    # Perform the affine transformation (rotation)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    
    # Return the rotated image
    return rotated

def template_matching(original, template):
    max_angle =-1
    curmax =0
    # Iterate over angles from 0 to 179 degrees
    for angle in range(0, 180,2):
        # Rotate the template image by the current angle
        rotated_template = rotate_image(template, angle)
        
        # Perform template matching between the rotated template and the original image. Included padding 100px to find best window to match
        res = cv.matchTemplate(rotated_template, original[:300, :300], cv.TM_CCOEFF_NORMED)

        # Check if the current match is better than the previous best match
        if np.max(res) > curmax:
            # Update the best matching angle and the highest match value
            max_angle = angle
            curmax = np.max(res)

    print(f"Best matching angle: {max_angle} degrees with loss value: {curmax}")
    return max_angle
# angle = template_matching(original, template)
# print(angle)
# cv.imshow("as", original)
# cv.imshow("asd", rotate_image(template, angle))
# cv.waitKey(0)
# exit()
for file in os.listdir('/dev/files/workspace/mvtec_ad/grid/train/good/'):
    if file.endswith('.png'):
        template = cv.imread('/dev/files/workspace/mvtec_ad/grid/train/good/'+file, cv.IMREAD_GRAYSCALE)
        angle = template_matching(original, template)
        print(angle)
        rotated = rotate_image(template, angle)
        cv.imwrite('/dev/files/workspace/mvtec_ad/grid/train/rotated/'+file, rotated)

