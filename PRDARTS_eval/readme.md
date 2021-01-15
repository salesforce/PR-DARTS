
This code fold includes the pretrained models, including CIFAR10, CIFAR100,
ImageNet which locates in the "pretrain_models" fold.

For test, there are two steps.

Step 1.
For test, you should provide the data path in test_cifar.py and test_imagenet.py:
parser.add_argument('--data_dir', type=str, default='/export/home/dataset/cifar10/', help='data dir')

Moreover, please assign the GPU at line 22 by setting CUDA_VISIBLE_DEVICES=0. For imagenet, it requires
two GPUs.

Step 2.

To evalute CIFAR10, you can directly run test_cifar.py

To evalute CIFAR100, You also need to test_cifar.py. But you should first comment the setting lines 30-33 for CIFAR10
and then open the comment (lines 36-39) for CIFAR100.

To evalute ImageNet, after you set the data path and GPU properly, you can directly run test_imagenet.py

For train, there are two steps.

Step 1.
For test, you should provide the data path in train_cifar.py and train_imagenet.py:
parser.add_argument('--data_dir', type=str, default='/export/home/dataset/cifar10/', help='data dir')

Moreover, please assign the GPU at line 45 by setting CUDA_VISIBLE_DEVICES=0. For imagenet, it requires
two GPUs.

Step 2.

To train on CIFAR10, you can directly run train_cifar.py

To train on CIFAR100, You also need to train_cifar.py by using
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ./train_cifar.py \
        --data_dir /export/home/dataset/cifar100/ \
        --cutout \
        --save './results/CIFAR10_train/' \
        --cifar100 \
        --drop_path_prob 0.4

To train on ImageNet, after you set the data path and GPU properly, you can directly run train_imagenet.py