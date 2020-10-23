CUDA_VISIBLE_DEVICES=7 python main.py \
--model EDSR --scale 4 \
--lr 1e-4 --epochs 300 --decay 150  --loss 1*L1 \
--data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 \
--batch_size 8 --patch_size 192 \
--n_threads 1 --n_GPUs 1 \
--save fdsr_train \
--searched_model fdsr_full_x4_25% \
--n_feats 256 --n_resblocks 32 --res_scale 0.1 \
--dir_data ~/workspace/datasets/ \
# Put your own dataset directory to dir_data
