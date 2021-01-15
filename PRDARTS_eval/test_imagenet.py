import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import pdb
from torch.autograd import Variable
from model import NetworkImageNet as Network

# for test, the cuda number reqires at least two.
CUDA_VISIBLE_DEVICES=6,7
OMP_NUM_THREADS=16

parser = argparse.ArgumentParser("test imagenet")
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower, not used in the inference phase')
parser.add_argument('--model_path', type=str, default='./pretrain_models/ImageNet_model2.pt', help='experiment name')
parser.add_argument('--selected_arch', type=str, default='PRDARTS', help='which architecture to use')
parser.add_argument('--data_dir', type=str, default='/export/share/datasets/vision/imagenet', help='temp data dir')


args, unparsed = parser.parse_known_args()




def main():
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    ## step 1 construct the selected network
    genotype = eval("genotypes.%s" % args.selected_arch)
    CLASSES = 1000
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)


    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    ## step 2 load pretrained model parameter
    model_CKPT = torch.load(args.model_path)
    model.load_state_dict(model_CKPT['state_dict'])
    model.module.drop_path_prob = 0
    model.drop_path_prob = 0
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    ## step 3 load test data
    valid_queue = load_data_cifar(args)


    ## step 4. inference on test data
    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    print('-----------------------------------------------')
    print('Valid_acc_top1: %f,  Valid_acc_top5: %f' % (valid_acc_top1, valid_acc_top5))
    print('-----------------------------------------------')


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % 50 == 0:
            print('VALID Step: %03d Objs: %e R1: %f R5: %f'%(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, objs.avg


def load_data_cifar(args):
    validdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_data = dset.ImageFolder(validdir,
        transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,
        ]))
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    return valid_queue

if __name__ == '__main__':
    main()
