#!/bin/python

import os
from func import *

if __name__ == '__main__':

    load_path = 'test_dataset/'
    save_path = './result/'

    createFolder(save_path)
    image_name = os.listdir(load_path)

    for name in image_name:
        img = load_path + name
        origin_img, cnts, ROI_number = extract_bounding_boxes(img, save_path, name='')
        print("The number of ROI(Region of Interest): ", str(ROI_number))