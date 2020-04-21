import matplotlib.pyplot as plt
from NCC import align
import numpy as np
import cv2
import os

dir_name = '../hw2_data/task3_colorizing'
save_dir_name = 'output'
img_list = os.listdir(dir_name)

for img_name in img_list:
    name = img_name.split('.')
    # JPG colorizing
    if name[1] == 'jpg':
        # Read image
        img = cv2.imread(os.path.join(dir_name, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove border
        h, w = img.shape
        img = img[int(h*0.01):int(h-h*0.02), int(w*0.02):int(w-w*0.02)]

        # Divide into blue, green, red image three part
        h, w = img.shape
        height = int(h/3)
        blue = img[0:height, :]
        green = img[height:2*height, :]
        red = img[2*height:3*height, :]

        # align(base matrix, align matrix, window size)
        # max ncc => shift 17
        alignGtoB = align(blue, green, 20)
        alignRtoB = align(blue, red, 20)
        g = np.roll(green, alignGtoB, axis=(0, 1))
        r = np.roll(red, alignRtoB, axis=(0, 1))
        # g = np.roll(green, [0,0], axis = (0, 1))
        # r = np.roll(red,[0,0],axis = (0, 1))

        colored = np.dstack((r, g, blue))
        colored = colored[int(colored.shape[0]*0.05):int(colored.shape[0]-colored.shape[0]*0.05), int(colored.shape[1]*0.05):int(colored.shape[1]-colored.shape[1]*0.05)]
        plt.imshow(colored)
        plt.show()
        plt.imsave(os.path.join(save_dir_name, img_name), colored)
    # TIF colorizing
    elif name[1] == 'tif':
        img = cv2.imread(os.path.join(dir_name, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        img = img[int(h*0.01):int(h-h*0.02), int(w*0.02):int(w-w*0.02)]
        h, w = img.shape
        height = int(h/3)
        blue_ = img[0:height, :]
        green_ = img[height:2*height, :]
        red_ = img[2*height:3*height, :]

        img = cv2.resize(img, (int(w/10), int(h/10)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape
        height = int(h/3)
        blue = img[0:height, :]
        green = img[height:2*height, :]
        red = img[2*height:3*height, :]

        # max shift => shift 12
        alignGtoB = align(blue, green, 15)
        alignRtoB = align(blue, red, 15)
        g = np.roll(green_, [alignGtoB[0]*10, alignGtoB[1]*10], axis=(0, 1))
        r = np.roll(red_, [alignRtoB[0]*10, alignRtoB[1]*10], axis=(0, 1))

        colored = np.dstack((r, g, blue_))
        colored = colored[int(colored.shape[0]*0.05):int(colored.shape[0]-colored.shape[0]*0.05), int(colored.shape[1]*0.05):int(colored.shape[1]-colored.shape[1]*0.05)]
        plt.imshow(colored)
        plt.show()
        plt.imsave(os.path.join(save_dir_name, name[0]+'.jpg'), colored)
    else:
        continue
