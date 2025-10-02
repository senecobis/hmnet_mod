cd /home/pellerito/HMNet_pth__/experiments/segmentation

CUDA_VISIBLE_DEVICES=1 python scripts/test.py \
config/hmnet_B3_custom.py \
data/dsec/list/test/ \
data/dsec/ \
--speed_test \
--pretrained workspace/hmnet_B3_custom/VQ_2025-09-26_13-32-26/checkpoint_8.pth.tar