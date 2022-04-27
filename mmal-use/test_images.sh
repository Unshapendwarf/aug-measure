#!/bin/bash
# echo &PWD
n_b=8
n_f=32
n_cluster=128

let end=$n_cluster-1
# img_dir="/home/pi/dir-1/n01930112"
dir_path="/home/hong/dir6/playground/n01930112/"
img_dir=`ls /home/hong/dir6/playground/n01930112/`

for entry in "$img_dir"
do
    # echo "${dir_path}"
    echo "${entry}"
    echo 1
done

# for ((var=0; var<=${end}; var++))
# do
# CUDA_VISIBLE_DEVICES=1 python train.py --data_root /mnt/URP_DS/final_eval_240/ \
#       --cluster_num $var  --pretrained --pretrained_path pretrained_model/240_EDSR_${n_b}_${n_f}.pth\
#       --input_name 240_LR_avif \
#       --target_name 240_HR \
#       --model_type EDSR \
#       --patch_size 80 \
#       --model_save_root saved_models/save_model_merge/${n_cluster}/${n_b}_${n_f} \
#       --scale 3 --n_blocks $n_b --n_feats $n_f \
#       --use_cuda --num_epoch 10 --num_valid_image 3\
#       --csv_file csv-index/rand_${n_cluster}.csv \
#       --num_batch 4 \
    #   --num_update_per_epoch 1000
# done