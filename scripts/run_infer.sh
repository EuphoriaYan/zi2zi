export CUDA_VISIBLE_DEVICES=3


python infer.py \
--model_dir \
experiment/checkpoint/experiment_1_batch_32 \
--batch_size \
32 \
--source_obj \
infer_experiment/data/train.obj \
--embedding_ids \
1 \
--save_dir \
infer