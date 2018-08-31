DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python ../train_image_classifier.py \
--train_dir $TRAIN_DIR \
--dataset_dir $DATASET_DIR \
--dataset_name imagenet \
--dataset_split_name train \
--model_name resnet_v1_50 \
--learning_rate_decay_type one_cycle \
--optimizer momentum \
--momentum 0.95 \
--min_momentum 0.85 \
--weight_decay 0.00001 \
--learning_rate 0.5 \
--max_learning_rate 1.0 \
--step_size 50000 \
--max_number_of_steps 150000 \
--train_image_size 224 \
--batch_size 64