import os

import cv2
import numpy as np
import math

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def distance(X1, X2):
    dist = math.sqrt((X1[0] - X2[0])**2 + (X1[1] - X2[1])**2)
    return dist

def extract_bounding_boxes (input_filename, save_path, name):
    image = cv2.imread(input_filename)
    original = image.copy()
    origin_img = image
    image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(origin_img, [box], 0, (36,255,12), 3)

        width = (distance(box[0],box[1])+distance(box[2],box[3]))*0.5
        height = (distance(box[0],box[2])+distance(box[1],box[3]))*0.5
        width = int(width)
        height = int(height)

        srcPoint = np.array(box, dtype=np.float32)
        dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
        ROI = cv2.warpPerspective(original, matrix, (width, height))
        # ROI = cv2.resize(ROI, (256, 256))
        ROI = cv2.resize(ROI, (ROI.shape[0]*7, ROI.shape[1]*7))
        ROI = cv2.rotate(ROI, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imwrite(save_path + f'ROI_{ROI_number}.png', ROI)
        cv2.imwrite(save_path + 'ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

    # cv2.imwrite(save_path + f'ROI_{name}.png', origin_img)

    return origin_img, cnts, ROI_number