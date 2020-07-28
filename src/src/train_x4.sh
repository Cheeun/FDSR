python main.py \
--save_results \
--model EDSR --scale 4 \
--lr 1e-4 --epochs 300 --decay 150  --loss 1*L1 \
--batch_size 8 --patch_size 192 \
--data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 \
--save searched_small_edsr_x4_prune \
--import_dir searched_small_edsr_x4 \
--n_feats 64 --n_resblocks 16 \
--n_threads 1 --n_GPUs 1 \
