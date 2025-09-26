CUDA_VISIBLE_DEVICES=2,3 \
python scripts/train.py \
/home/pellerito/HMNet_pth__/experiments/segmentation/config/hmnet_B3_custom.py \
--amp --overwrite --wandb --distributed