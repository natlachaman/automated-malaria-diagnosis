#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process
#PBS -q tiny
#PBS -N test
#PBS -k oe
#PBS -l walltime=00:10:00

module load cs/keras/2.1.0-tensorflow-1.4-python-3.5
module load lib/hdf5/1.8.16-gnu-4.9

python train.py \
	--gt gt_train3_fal_wbc_prep.pickle\
	--batch_size 8\
	--epochs 120\
	--lr 0.001\
	--matching_thrs 0.5\
	--optimizer 'adam'\
	--comment 'short from scratch, 768x768, 0.5 IoU, offset 0.5, bounds (50, 230), LR=.001'\
	--img_shape 768 768 3\
    --num_classes 3\
    --no_annotation_label 0.\
    --feat_layers 'block4_conv1' 'block6' 'block8_conv2' 'block9_conv2'\
    --anchor_size_bounds 0.05 0.36\
    --anchor_ratios 1 1 1 1 1 1 1 1\
	--anchor_steps 4 8 16 32 64 128 256 512\
	--anchor_offset 0.5\
	--normalizations 20 -1 -1 -1 -1 -1 -1\
	--prior_scaling 0.1 0.1 0.2 0.2\
	--img_dir 'train3'


