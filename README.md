## Theory-Inspired Path-Regularized Differential Network Architecture Search

This is a PyTorch implementation of the [PR-DARTS paper](https://arxiv.org/pdf/2006.16537.pdf):
```
@inproceedings{zhou2020NAS,
  title={Theory-Inspired Path-Regularized Differential Network Architecture Search},
  author={Pan Zhou and Caiming Xiong and Richard Socher and Steven Hoi},
  booktitle={Neural Information Processing Systems},
  year={2020}
}
```
### Prepare
Our environment is Pytorch 1.2 and torchvision 0.4. 

### Network Architecture Search
Please find the search code in the "PRDARTS_search" fold. In this work, we search the network architecture on CIFAR10, and then evaluate the searched network on CIFAR10, CIFAR100 and ImageNet.


Step 1.
For search, you should provide the data path in test_cifar.py:
parser.add_argument('--data_path', type=str, default='/export/home/dataset/cifar10/', help='data dir')

Moreover, please assign the GPU at line 17 by setting CUDA_VISIBLE_DEVICES=0. For imagenet, it requires
two GPUs.

For setting optimizer's parameters or architecthure's parameters, please revise them in the files in the "configs" fold.

Step 2.
You can directly run test_PRDARTS.py. After searching, it will automatically output the genotypes (or you can check at the logging file).


```
python test_PRDARTS.py \
  --data_path [your imagenet-folder with train and val folders]
```

If you want to train this selected  genotype, you can directly replace the PRDART variable in the "genotypes.py" file of the training codes in  as this genotype

### Network Architecture Evaluation
Please find the search code in the "PRDARTS_eval" fold.  
For train, there are two steps.

Step 1.
For test, you should provide the data path in train_cifar.py and train_imagenet.py:
parser.add_argument('--data_dir', type=str, default='/export/home/dataset/cifar10/', help='data dir')

Moreover, please assign the GPU at line 45 by setting CUDA_VISIBLE_DEVICES=0. For imagenet, it requires
two GPUs.

Step 2.

To train on CIFAR10, you can directly run train_cifar.py
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ./train_cifar.py 
```

To train on CIFAR100, You also need to train_cifar.py by using
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python ./train_cifar.py \
        --data_dir /export/home/dataset/cifar100/ \
        --cutout \
        --save './results/CIFAR10_train/' \
        --cifar100 \
        --drop_path_prob 0.4
```

To train on ImageNet, after you set the data path and GPU properly, you can directly run train_imagenet.py
```
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python ./train_imagenet.py 
```


### Models

Our trained selected models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">CIFAR10 top1 error</th>
<th valign="bottom">CIFAR100 top1 error</th>
<th valign="bottom">ImageNet top1 error</th>
<th valign="bottom">ImageNet top5 error</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/pdf/2006.16537.pdf">PR-DARTS v2</a></td>
<td align="center">2.32 <a href="./PRDARTS_eval/pretrain_models/CIFAR10_model.pt">download</a></td>
<td align="center">16.45 <a href="./PRDARTS_eval/pretrain_models/CIFAR100_model.pt">download</a></td>
<td align="center">24.1 <a href="./PRDARTS_eval/pretrain_models/ImageNet_model2.pt">download</a></td>
<td align="center">7.3 <a href="./PRDARTS_eval/pretrain_models/ImageNet_model2.pt">download</a></td>
</tr>
</tbody></table>



### License

This project is under the MIT License. See [LICENSE](LICENSE) for details.
