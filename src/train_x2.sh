CUDA_VISIBLE_DEVICES=5 python main.py \
--n_feats 256 --n_resblocks 32 --res_scale 0.1 \
--batch_size 8 --patch_size 192 \
--data_test Set5+Set14+B100+Urban100 --data_range 1-800/801-810 --dir_data ~/workspace/datasets/ \
--scale 2 \
--save fdsr_train_x2 \
--searched_model fdsr_full_x2_3% \
# --test_only #uncomment for 