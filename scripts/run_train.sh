export CUDA_VISIBLE_DEVICES=3

experiment_dir=experiment
experiment_id=1
batch_size=32
learning_rate=1e-3
epoch=100
sample_steps=200
checkpoint_steps=500
schedule=20

python train.py \
--experiment_dir ${experiment_dir} \
--experiment_id ${experiment_id} \
--batch_size ${batch_size} \
--lr ${learning_rate} \
--epoch ${epoch} \
--sample_steps ${sample_steps} \
--checkpoint_steps ${checkpoint_steps} \
--schedule ${schedule}
