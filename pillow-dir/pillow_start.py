from email.mime import image
import PIL
import os
import glob

from time import time

print("pillow start")

dataset_dir = "/home/hong/dataset/imagenet"
src_dir = os.path.join(dataset_dir, "whatisthis")  # change this function


image_list = []
for img_path in image_list:
    # @time-start-1
    start_time = time()
    curr_image = PIL.Image.open(img_path)
    # @time-end-1
    end_time = time()
    print(f"read:{end_time - start_time}")

