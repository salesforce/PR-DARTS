import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import NetworkCIFAR as Network
import pdb

CUDA_VISIBLE_DEVICES=1
OMP_NUM_THREADS=4

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower, not used in inference phase')
parser.add_argument('--cutout', type=bool, default=False, help='use cutout')

## evaluate on CIFAR 10, if you want to evaluate CIFAR100, please comment this paragraph
# parser.add_argument('--model_path', type=str, default='./pretrain_models/CIFAR10_model.pt', help='experiment name')
# parser.add_argument('--selected_arch', type=str, default="PRDARTS", help='which architecture to use')
# parser.add_argument('--data_dir', type=str, default='/export/home/dataset/cifar10/', help='data dir')
# parser.add_argument('--cifar100', action='store_true', default=False, help='if use cifar100')

## evaluate on CIFAR 100, if you want to evaluate CIFAR100, please open the following comment
parser.add_argument('--model_path', type=str, default='./pretrain_models/CIFAR100_model.pt', help='experiment name')
parser.add_argument('--selected_arch', type=str, default="PRDARTS", help='which architecture to use')
parser.add_argument('--data_dir', type=str, default='/export/home/dataset/cifar100/', help='data dir')
parser.add_argument('--cifar100', action='store_true', default=True, help='if use cifar100')

args, unparsed = parser.parse_known_args()


if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'


def main():
    if not torch.cuda.is_available():
        sys.exit(1)

    ## step 1 construct the selected network
    genotype = eval("genotypes.%s" % args.selected_arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    ## step 2 load pretrained model parameter
    if args.cifar100:
     model = torch.nn.DataParallel(model)
     model = model.cuda()
     model.load_state_dict(torch.load(args.model_path)['net'])
    else:
     utils.load(model, args.model_path)
     model = torch.nn.DataParallel(model)
     model = model.cuda()

    model.module.drop_path_prob = 0
    model.drop_path_prob = 0

    print("param size = %fMB"%utils.count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    ## step 3 load test data
    valid_queue = load_data_cifar(args)

    ## step 4. inference on test data
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    print('-----------------------------------------------')
    print('Average Valid_acc: %f '%valid_acc)
    print('-----------------------------------------------')



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        print('Valid Step: %03d Objs: %e Overall Acc: %f'%( step, objs.avg, top1.avg))

    return top1.avg, objs.avg


def load_data_cifar(args):
    ## step 3 load test data
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        # train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data_dir, train=False, download=True, transform=valid_transform)
    else:
        # train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data_dir, train=False, download=True, transform=valid_transform)

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    return valid_queue

if __name__ == '__main__':
    main()


