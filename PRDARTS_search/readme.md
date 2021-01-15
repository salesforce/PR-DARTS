
This code fold includes the search code of PR-DARTS.

For test, there are two steps.

Step 1.
For search, you should provide the data path in test_cifar.py:
parser.add_argument('--data_path', type=str, default='/export/home/dataset/cifar10/', help='data dir')

Moreover, please assign the GPU at line 17 by setting CUDA_VISIBLE_DEVICES=0. For imagenet, it requires
two GPUs.

For setting optimizer's parameters or architecthure's parameters, please revise them in the files in the "configs" fold.

Step 2.
You can directly run test_PRDARTS.py. After searching, it will automatically output the genotypes (or you can check at the logging file).

If you want to train this selected  genotype, you can directly replace the PRDART variable in the "genotypes.py" file of the training codes as this genotype
