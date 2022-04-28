#!/bin/bash
# echo &PWD

# dir_path="/home/pi/dir-1/n01930112"
dir_path="/home/hong/dir6/playground/n01930112"

for entry in ${dir_path}/*
do
    echo "./hello_mmal_jpeg.bin ${entry}"
    # ./hello_mmal_jpeg.bin ${entry}
done