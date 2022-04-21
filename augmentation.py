import numpy as np
#import matplotlib.pyplot as plt
import cv2
import glob
import random

import time
import os

def brightness(gray, val):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = int(random.uniform(-val, val))
    if brightness > 0:
        gray = gray + brightness
    else:
        gray = gray - brightness
    gray = np.clip(gray, 10, 255)
    return gray

def contrast(gray, min_val, max_val):
    #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    alpha = int(random.uniform(min_val, max_val)) # Contrast control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha)
    return adjusted

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def vertical_shift_down(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    img = fill(img, h, w)
    return img

def vertical_shift_up(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(0.0, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    img = fill(img, h, w)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def main():
    dir = "/dev/shm/dataset/n01930112"
    # dir = "../n01930112"
    image_dir = os.path.join(dir,"*.JPEG")
    images = sorted(glob.glob(image_dir))
    i = 0
    # print(images[0])
    # print(len(images))

    for fname in images:
        #open file with opencv

        # set timer
        start = time.time()

        #do augmentation
        img = cv2.imread(fname)
        readtime = time.time()
        # print(img.shape)
        
        img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
        img = brightness(img, 10)
        img = contrast(img, 1, 1.2)
        img = horizontal_flip(img, 1)
        img = rotation(img, 180)
        img = horizontal_shift(img, 0.2)
        #if random.uniform(0,1) > 0.5:
        img = vertical_flip(img, 1)
        
        # file_name = dir + str(i) + '.png'
        # #file_name = 'aug_image/' + str(i) + '.png'
        # cv2.imwrite(file_name, img)
        # i = i + 1
        # if i > 10000:
        #     break

        end = time.time()
        reading = (readtime-start)*1000
        transforming = (end-readtime)*1000
        print(f"read:{reading:.4f}, augment:{transforming:.4f}") # miliseconds


if __name__ == "__main__":
    # run main program
    main()
