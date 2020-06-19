export CUDA_VISIBLE_DEVICES=3

experiment_dir=experiment
experiment_id=0
batch_size=32
learning_rate=1e-3
epoch=100
sample_step=100
schedule=20

python train.py \
--experiment_dir ${experiment_dir} \
--experiment_id ${experiment_id} \
--batch_size ${batch_size} \
--lr ${learning_rate} \
--epoch ${epoch} \
--sample_step ${sample_step} \
--schedule ${schedule}
