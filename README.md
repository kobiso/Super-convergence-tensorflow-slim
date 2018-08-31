# Super-Convergence-TensorFlow-Slim
This is a Tensorflow implementation of [Super-Convergence: very fast training of neural networks using large learning rates](https://arxiv.org/abs/1708.07120) aiming to be compatible on the [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim).

## Super-Convergence
This paper suggests a different learning rate policy called *one-cycle policy* which makes network to be trained significantly faster and named this phenomenon *super-convergence*.

### Cyclical Learning Rate
[*Cyclical Learning Rate (CLR)*](https://arxiv.org/abs/1506.01186) is the previous research of the author and suggests to train a network with CLR to get better classification performance.

<div align="center">
  <img src="https://github.com/kobiso/super-convergence-tensorflow-slim/blob/master/figures/clr.png"  width="420">
</div>

### One-Cycle Policy
*Super-convergence* paper suggests *one-cycle policy* which is small modification of CLR.
It always use one cycle that is smaller than the total number of iterations/epochs and allow the learning rate to decrease several orders of magnitude less than the initial learning rate for the remaining iterations.
If using one cycle learning rate schedule, it is better to use a *cyclical momentum (CM)* that starts at the maximum momentum value and decreases with increasing learning rate.

### Learning Rate Range Test
In order to find proper learning rate range for CLR and one-cycle policy, you can use *learning rate range test*.
As the learning rate increases, it eventually becomes too large and czuses the test/validation loss to increase and the accuracy to decrease.
This point or smaller value of this point can be used as the maximum bound.
Minimum bound can be found as:
1. a factor of 3 or 4 less than the maximum bound
2. a factor of 10 or 20 less than the maximum bound if only one cycle is used
3. by a short test of hundreds of iterations with a few initial learning rates and pick the largest one that allows convergence to begin without signs of overfitting

### Test Results

<div align="center">
  <img src="https://github.com/kobiso/super-convergence-tensorflow-slim/blob/master/figures/ex.png">
</div>

<div align="center">
  <img src="https://github.com/kobiso/super-convergence-tensorflow-slim/blob/master/figures/ex2.png">
</div>

## Prerequisites
- Python 3.x
- TensorFlow 1.x
- TF-slim
  - Check the ['installation' part of TF-Slim image models README](https://github.com/tensorflow/models/tree/master/research/slim#installation).

## Prepare Data set
You should prepare your own dataset or open dataset (Cifar10, flowers, MNIST, ImageNet).
For preparing dataset, you can follow the ['preparing the datasets' part in TF-Slim image models README](https://github.com/tensorflow/models/tree/master/research/slim#preparing-the-datasets).

## Train a Model
### Learning Rate Range Test
Below script gives you an example of learning rate range test.
To perform learning rate range test, `--learning_rate_decay_type` argument should be `lr_range_test`.
Minimum learning rate and maximum learning rate should be set on `--learning_rate` and `--max_learning_rate` arguments.

```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python ../train_image_classifier.py \
--train_dir $TRAIN_DIR \
--dataset_dir $DATASET_DIR \
--dataset_name imagenet \
--dataset_split_name train \
--model_name resnet_v1_50 \
--learning_rate_decay_type lr_range_test \
--optimizer momentum \
--momentum 0.9 \
--weight_decay 0.00001 \
--learning_rate 0.00001 \
--max_learning_rate 3.0 \
--step_size 50000 \
--max_number_of_steps 100000 \
--train_image_size 224 \
--batch_size 64
```

### Train with One-Cycle Policy
Below script gives you an example of training a model with one-cycle policy.
To perform one-cycle policy, `--learning_rate_decay_type` argument should be `one_cycle`.
For cyclical learning rate, minimum learning rate and maximum learning rate should be set on `--learning_rate` and `--max_learning_rate` arguments.
For cyclical momentum, minimum momentum and maximum momentum should be set on `--min_momentum` and `--momentum` arguments.

```
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
```

### Train with Cyclical Learning Rate
Below script gives you an example of training a model with cyclical learning rate.
To perform cyclical learning rate, `--learning_rate_decay_type` argument should be `CLR`.
Minimum learning rate and maximum learning rate should be set on `--learning_rate` and `--max_learning_rate` arguments.

```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python ../train_image_classifier.py \
--train_dir $TRAIN_DIR \
--dataset_dir $DATASET_DIR \
--dataset_name imagenet \
--dataset_split_name train \
--model_name resnet_v1_50 \
--learning_rate_decay_type CLR \
--optimizer momentum \
--momentum 0.95 \
--min_momentum 0.85 \
--weight_decay 0.00001 \
--learning_rate 0.1 \
--max_learning_rate 0.5 \
--step_size 1000 \
--max_number_of_steps 10000 \
--train_image_size 224 \
--batch_size 64
```

## Evaluate a Model
To keep track of validation accuracy while training, you can use `eval_image_classifier_loop.py` which evaluate the performance at multiple checkpoints during training.
If you want to just evaluate a model once, you can use `eval_image_classifier.py`.


Below script gives you an example of evaluating a model repeatedly while training a model.

```
DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 
```

## Experiments

### Cyclical LR and Momentum

<div align="center">
  <img src="https://github.com/kobiso/super-convergence-tensorflow-slim/blob/master/figures/lr_momentum.png">
</div>

### Results

<div align="center">
  <img src="https://github.com/kobiso/super-convergence-tensorflow-slim/blob/master/figures/test.png">
</div>

LR_schedule | model | data | max_steps | step_size | batch_size | optimizer | lr | weight_decay | momentum | acc | training_time
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
constant LR | resnet50_v1 | imagenet | 1000k | - | 128 | rmsprop | 0.01 | 0.00004 | 0.9 | 0.6923 | 7d 19h
one-cycle policy | resnet50_v1 | imagenet | 250k | 100k | 128 | momentum | 0.05-1.0 | 0.00001 | 0.95-0.85 | **0.7075** | 2d 16h

## Related Works
- Blog: [Super-Convergence: very fast training of neural networks using large learning rates](https://kobiso.github.io//research/research-super-convergence/)
- Repository: [CBAM-TensorFlow-Slim](https://github.com/kobiso/CBAM-tensorflow-slim)
- Repository: [SENet-TensorFlow-Slim](https://github.com/kobiso/SENet-tensorflow-slim)

## References
- Paper: [Cyclical learning rates for training neural networks](https://arxiv.org/abs/1506.01186)
- Paper: [Super-Convergence: very fast training of neural networks using large learning rates](https://arxiv.org/abs/1708.07120)
- Paper: [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- Github: [Caffe files of NRL Technical Report](https://github.com/lnsmith54/hyperParam1)
- Github: [keras-one-cycle](https://github.com/titu1994/keras-one-cycle)
  
## Author
Byung Soo Ko / kobiso62@gmail.com
