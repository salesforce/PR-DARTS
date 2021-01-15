import os, sys, time, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
import pdb

from config_utils import load_config, dict2config
from get_dataset import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from optimizers import get_optim_scheduler
from utils import get_model_infos, obtain_accuracy
from logger_utils import AverageMeter, time_string, convert_secs2time
from models import get_cell_based_tiny_net, get_search_spaces

#CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 python test_PRDARTS.py


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()
    # pdb.set_trace()
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
        scheduler.update(None, 1.0 * step / len(xloader))
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # update the weights
        w_optimizer.zero_grad()
        _, logits, _ = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # update the architecture-weight
        a_optimizer.zero_grad()
        _, logits, activation_terms = network(inputs=arch_inputs)
        arch_loss = criterion(logits, arch_targets)

        arch_loss = arch_loss + activation_terms.mean()
        arch_loss.backward()
        a_optimizer.step()

        arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=base_losses, top1=base_top1, top5=base_top5)
            Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=arch_losses, top1=arch_top1, top5=arch_top5)
            logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
    # pdb.set_trace()
    return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)


    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)


    config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                            './configs/', config.batch_size, xargs.workers)
    logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader),
                                                                                     config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    if xargs.model_config is None:
        model_config = dict2config({'name': 'PRDARTS', 'C': xargs.channel, 'N': xargs.num_cells,
                                    'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                    'space': search_space,
                                    'affine': False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
    else:
        model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space': search_space,
                                                        'affine': False,
                                                        'track_running_stats': bool(xargs.track_running_stats)}, None)
    # pdb.set_trace()
    search_model = get_cell_based_tiny_net(model_config)
    search_model.set_k(xargs.k)
    search_model.set_threshold(xargs.threshold)
    search_model.set_regular_lambda(xargs.regular_lambda1, xargs.regular_lambda2, xargs.regular_lambda3)

    logger.log('search-model :\n{:}'.format(search_model))
    logger.log('model-config : {:}'.format(model_config))

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
    a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=xargs.arch_weight_decay)
    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('a-optimizer : {:}'.format(a_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))
    logger.log('criterion   : {:}'.format(criterion))
    flop, param = get_model_infos(search_model, xshape)
    # pdb.set_trace()

    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()


    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        genotypes = checkpoint['genotypes']
        valid_accuracies = checkpoint['valid_accuracies']
        search_model.load_state_dict(checkpoint['search_model'])
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: search_model.genotype()}


    # start training
    probabilitys= []
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        search_model.set_tau(xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1))

        search_model.set_ab(xargs.a_min + (xargs.a_max - xargs.a_min) * epoch / (total_epoch - 1),
                            xargs.b_max - (xargs.b_max - xargs.b_min) * epoch / (total_epoch - 1))

        probabilitys.append(search_model.get_probability())
        search_model.set_dropout(xargs.dropout_pro)

        logger.log('\n[Search the {:}-th epoch] {:}, tau={:},  LR={:}, a={:}, b={:}, lambda1={:}, lambda2={:}, lambda3={:}, k={:}'.format(epoch_str, need_time, \
                                                        search_model.get_tau(), min(w_scheduler.get_lr()), search_model.get_a(),search_model.get_b(),xargs.regular_lambda1,\
                                                                      xargs.regular_lambda2,xargs.regular_lambda3,xargs.k))



        search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5\
            = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str,
                          xargs.print_freq, logger)



        search_time.update(time.time() - start_time)
        logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(
            epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
        logger.log(
            '[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss,
                                                                                           valid_a_top1, valid_a_top5))
        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1
        if valid_a_top1 > valid_accuracies['best']:
            valid_accuracies['best'] = valid_a_top1
            genotypes['best'] = search_model.genotype()
            find_best = True
        else:
            find_best = False

        genotypes[epoch] = search_model.genotype()
        logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
        # save checkpoint
        save_path = save_checkpoint({'epoch': epoch + 1,
                                     'args': deepcopy(xargs),
                                     'search_model': search_model.state_dict(),
                                     'w_optimizer': w_optimizer.state_dict(),
                                     'a_optimizer': a_optimizer.state_dict(),
                                     'w_scheduler': w_scheduler.state_dict(),
                                     'genotypes': genotypes,
                                     'valid_accuracies': valid_accuracies,
                                     'probabilitys': probabilitys
                                     },
                                    model_base_path, logger)
        last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args': deepcopy(args),
            'last_checkpoint': save_path,
        }, logger.path('info'), logger)
        if find_best:
            logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str,
                                                                                                             valid_a_top1))
            copy_checkpoint(model_base_path, model_best_path, logger)
        with torch.no_grad():
            logger.log('{:}'.format(search_model.show_alphas()))
        if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch])))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log('\n' + '-' * 100)
    # check the performance from the architecture dataset
    logger.log('GGA : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum,
                                                                                genotypes[total_epoch - 1]))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[total_epoch - 1])))
    logger.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("PRDARTS")
    parser.add_argument('--data_path', type=str, default='/export/home/dataset/cifar.python', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='search on CIFAR10.')
    parser.add_argument('--search_space_name', type=str, default='darts', help='The search space name.')
    parser.add_argument('--max_nodes', type=int, default=4, help='The maximum number of nodes.')
    parser.add_argument('--channel', type=int, default=36, help='The number of channels.')
    parser.add_argument('--num_cells', type=int, default=5, help='The number of cells in one stage.')
    parser.add_argument('--track_running_stats', type=int, default=0, help='Whether use track_running_stats or not in the BN layer.')  # choices=[0,1]
    parser.add_argument('--config_path', type=str, default='./configs/PRDARTS-NASNet-optimizer.config',help='optimizer settings.')
    parser.add_argument('--model_config', type=str, default='./configs/PRDARTS-NASNet-archs.config', help='search architecture setting')

    # architecture leraning rate
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--tau_min', type=float, default=1e-1, help='The minimum tau for Gumbel')
    parser.add_argument('--tau_max', type=float, default=1, help='The maximum tau for Gumbel')
    parser.add_argument('--a_min', type=float, default=-0.1, help='The minimum tau for Gumbel')
    parser.add_argument('--a_max', type=float, default=-0.1, help='The maximum tau for Gumbel')
    parser.add_argument('--b_min', type=float, default=1.1, help='The minimum tau for Gumbel')
    parser.add_argument('--b_max', type=float, default=1.1, help='The maximum tau for Gumbel')

    parser.add_argument('--regular_lambda1', type=float, default=0.01, help='The maximum tau for Gumbel')#0.2
    parser.add_argument('--regular_lambda2', type=float, default=0.005, help='The maximum tau for Gumbel')#0.1
    parser.add_argument('--regular_lambda3', type=float, default=0.005, help='The maximum tau for Gumbel')#0.1

    parser.add_argument('--k', type=float, default=2, help='The maximum tau for Gumbel')


    parser.add_argument('--cutout_length', type=int, default=16, help='The maximum tau for Gumbel')
    parser.add_argument('--dropout_pro', type=float, default=0.40, help='The maximum tau for Gumbel')
    parser.add_argument('--threshold', type=float, default=0.20, help='The maximum tau for Gumbel')

    # log
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir', type=str, default='./output/search-cell-darts/PRDARTS-search-margin-001-skippoolinggroup',
                        help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str,
                        help='The path to load the architecture dataset (tiny-nas-benchmark).')  # default='/export/home/dataset/NAS-Bench-201-v1_0-e61699.pth ',
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=1, help='manual seed')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
    main(args)
