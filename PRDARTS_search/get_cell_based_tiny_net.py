##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from os import path as osp
import numpy as np
from typing import List, Text
import torch
import pdb
# __all__ = ['change_key', 'get_cell_based_tiny_net', 'get_search_spaces', 'get_cifar_models', 'get_imagenet_models', \
#            'obtain_model', 'obtain_search_model', 'load_net_from_checkpoint', \
#            'CellStructure', 'CellArchitectures'
#            ]

# useful modules
from config_utils import dict2config
# from .SharedUtils import change_key
from genotypes import Structure as CellStructure
from genotypes import architectures as CellArchitectures
# from .cell_searchs import CellStructure, CellArchitectures
from search_model_prdarts_nasnet import NASNetworkPRDARTS

def change_key(key, value):
  def func(m):
    if hasattr(m, key):
      setattr(m, key, value)
  return func

def parse_channel_info(xstring):
  blocks = xstring.split(' ')
  blocks = [x.split('-') for x in blocks]
  blocks = [[int(_) for _ in x] for x in blocks]
  return blocks

# Cell-based NAS Models
def get_cell_based_tiny_net(config):
  if isinstance(config, dict): config = dict2config(config, None) # to support the argument being a dict
  super_type = getattr(config, 'super_type', 'basic')
  group_names = ['DARTS-V1', 'DARTS-V2', 'GDAS', 'GGA', 'GGA2', 'PRDARTS', 'SETN', 'ENAS', 'RANDOM']

  # pdb.set_trace()
  if super_type == 'basic' and config.name in group_names:
    from .cell_searchs import nas201_super_nets as nas_super_nets
    try:
      return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space, config.affine, config.track_running_stats)
    except:
      return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space)
  elif super_type == 'nasnet-super':
    from .cell_searchs import nasnet_super_nets as nas_super_nets
    # pdb.set_trace()
    return NASNetworkPRDARTS(config.C, config.N, config.steps, config.multiplier, \
                    config.stem_multiplier, config.num_classes, config.space, config.affine, config.track_running_stats)
    # return nas_super_nets[config.name](config.C, config.N, config.steps, config.multiplier, \
    #                 config.stem_multiplier, config.num_classes, config.space, config.affine, config.track_running_stats)
  elif config.name == 'infer.tiny':
    from .cell_infers import TinyNetwork
    if hasattr(config, 'genotype'):
      genotype = config.genotype
    elif hasattr(config, 'arch_str'):
      genotype = CellStructure.str2structure(config.arch_str)
    else: raise ValueError('Can not find genotype from this config : {:}'.format(config))
    return TinyNetwork(config.C, config.N, genotype, config.num_classes)
  elif config.name == 'infer.nasnet-cifar':
    from .cell_infers import NASNetonCIFAR
    raise NotImplementedError
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name) -> List[Text]:
  if xtype == 'cell':
    from .cell_operations import SearchSpaceNames
    assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
    return SearchSpaceNames[name]
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))


def get_cifar_models(config, extra_path=None, args_par=None):
  # pdb.set_trace()
  super_type = getattr(config, 'super_type', 'basic')
  if super_type == 'basic':
    from .CifarResNet      import CifarResNet
    from .CifarDenseNet    import DenseNet
    from .CifarWideResNet  import CifarWideResNet
    if config.arch == 'resnet':
      return CifarResNet(config.module, config.depth, config.class_num, config.zero_init_residual)
    elif config.arch == 'densenet':
      return DenseNet(config.growthRate, config.depth, config.reduction, config.class_num, config.bottleneck)
    elif config.arch == 'wideresnet':
      return CifarWideResNet(config.depth, config.wide_factor, config.class_num, config.dropout)
    else:
      raise ValueError('invalid module type : {:}'.format(config.arch))
  elif super_type.startswith('infer'):
    from .shape_infers import InferWidthCifarResNet
    from .shape_infers import InferDepthCifarResNet
    from .shape_infers import InferCifarResNet
    from .cell_infers import NASNetonCIFAR
    assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
    infer_mode = super_type.split('-')[1]
    # pdb.set_trace()
    if infer_mode == 'width':
      return InferWidthCifarResNet(config.module, config.depth, config.xchannels, config.class_num, config.zero_init_residual)
    elif infer_mode == 'depth':
      return InferDepthCifarResNet(config.module, config.depth, config.xblocks, config.class_num, config.zero_init_residual)
    elif infer_mode == 'shape':
      return InferCifarResNet(config.module, config.depth, config.xblocks, config.xchannels, config.class_num, config.zero_init_residual)
    elif infer_mode == 'nasnet.cifar':
      genotype = config.genotype
      pdb.set_trace()
      if extra_path is not None:  # reload genotype by extra_path
        if not osp.isfile(extra_path): raise ValueError('invalid extra_path : {:}'.format(extra_path))
        xdata = torch.load(extra_path)
        if not args_par.non_tailor:
          current_epoch = xdata['epoch']
          genotype = xdata['genotypes'][current_epoch-1]
        else:
          def sampling_hard_gate(xins,args_par):
            # pdb.set_trace()
            logits = (np.log(xins) + 0.5) / args_par.tau
            probs = (1.0/(1+np.exp(-logits))) * (args_par.limit_b - args_par.limit_a) + args_par.limit_a
            return probs.clip(0, 1.0)

          def get_edge2index(steps):
            import torch.nn as nn
            edges = nn.ModuleDict()
            for i in range(steps):
              for j in range(2 + i):
                node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
                # stride = 2 if reduction and j < 2 else 1
                # op = MixedOp(space, C, stride, affine, track_running_stats)
                edges[node_str] = None
            edge_keys = sorted(list(edges.keys()))
            edge2index = {key: i for i, key in enumerate(edge_keys)}
            return edge2index

          def get_genotype(weights, op_names, edge2index, steps, threshold):
              gene = []
              # gene2 = []
              # pdb.set_trace()
              for i in range(steps):
                edges = []
                for j in range(2 + i):
                  node_str = '{:}<-{:}'.format(i, j)
                  ws = weights[edge2index[node_str]]
                  for k, op_name in enumerate(op_names):
                    if op_name == 'none': continue
                    edges.append((op_name, j, ws[k]))
                # edges = sorted(edges, key=lambda x: -x[-1])
                # selected_edges = edges[:2]
                # gene.append(tuple(selected_edges))

                selected_edges = []
                for ik in range(0,len(edges)):
                  if edges[ik][-1] >= threshold:
                    selected_edges.append(edges[ik])
                if len(selected_edges):
                  # pdb.set_trace()
                  #selected_edges = edges[:2]
                  #selected_edges = edges[[int(i) for i in idx]]#np.array(
                  gene.append(tuple(selected_edges))
                # pdb.set_trace()
              if len(gene) <= 0:
                raise ValueError('invalid genotype: all are too small for connection')
              return gene

          #from models import get_search_spaces
          # pdb.set_trace()
          search_space = get_search_spaces('cell', 'darts')
          steps = 4
          edge2index = get_edge2index(steps)

          # pdb.set_trace()
          gene_normal = get_genotype(sampling_hard_gate(xdata['search_model']['arch_normal_parameters'].cpu().numpy(),args_par), search_space, edge2index, steps, args_par.threshold)
          gene_reduce = get_genotype(sampling_hard_gate(xdata['search_model']['arch_reduce_parameters'].cpu().numpy(),args_par), search_space, edge2index, steps, args_par.threshold)
          genotype = {'normal': gene_normal, 'normal_concat': list(range(2 , steps + 2)), 'reduce': gene_reduce, 'reduce_concat': list(range(2 , steps + 2))}
          # pdb.set_trace()
      C = config.C if hasattr(config, 'C') else config.ichannel
      N = config.N if hasattr(config, 'N') else config.layers
      # pdb.set_trace()
      return NASNetonCIFAR(C, N, config.stem_multi, config.class_num, genotype, config.auxiliary)
    else:
      raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
  else:
    raise ValueError('invalid super-type : {:}'.format(super_type))


def get_imagenet_models(config):
  super_type = getattr(config, 'super_type', 'basic')
  if super_type == 'basic':
    from .ImagenetResNet import ResNet
    if config.arch == 'resnet':
      return ResNet(config.block_name, config.layers, config.deep_stem, config.class_num, config.zero_init_residual, config.groups, config.width_per_group)
    else:
      raise ValueError('invalid arch : {:}'.format( config.arch ))
  elif super_type.startswith('infer'): # NAS searched architecture
    assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
    infer_mode = super_type.split('-')[1]
    if infer_mode == 'shape':
      from .shape_infers import InferImagenetResNet
      from .shape_infers import InferMobileNetV2
      if config.arch == 'resnet':
        return InferImagenetResNet(config.block_name, config.layers, config.xblocks, config.xchannels, config.deep_stem, config.class_num, config.zero_init_residual)
      elif config.arch == "MobileNetV2":
        return InferMobileNetV2(config.class_num, config.xchannels, config.xblocks, config.dropout)
      else:
        raise ValueError('invalid arch-mode : {:}'.format(config.arch))
    else:
      raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
  else:
    raise ValueError('invalid super-type : {:}'.format(super_type))


# Try to obtain the network by config.
def obtain_model(config, extra_path=None,  args_par=None):
  # pdb.set_trace()
  if config.dataset == 'cifar':
    return get_cifar_models(config, extra_path, args_par)
  elif config.dataset == 'imagenet':
    return get_imagenet_models(config)
  else:
    raise ValueError('invalid dataset in the model config : {:}'.format(config))


def obtain_search_model(config):
  if config.dataset == 'cifar':
    if config.arch == 'resnet':
      from .shape_searchs import SearchWidthCifarResNet
      from .shape_searchs import SearchDepthCifarResNet
      from .shape_searchs import SearchShapeCifarResNet
      if config.search_mode == 'width':
        return SearchWidthCifarResNet(config.module, config.depth, config.class_num)
      elif config.search_mode == 'depth':
        return SearchDepthCifarResNet(config.module, config.depth, config.class_num)
      elif config.search_mode == 'shape':
        return SearchShapeCifarResNet(config.module, config.depth, config.class_num)
      else: raise ValueError('invalid search mode : {:}'.format(config.search_mode))
    elif config.arch == 'simres':
      from .shape_searchs import SearchWidthSimResNet
      if config.search_mode == 'width':
        return SearchWidthSimResNet(config.depth, config.class_num)
      else: raise ValueError('invalid search mode : {:}'.format(config.search_mode))
    else:
      raise ValueError('invalid arch : {:} for dataset [{:}]'.format(config.arch, config.dataset))
  elif config.dataset == 'imagenet':
    from .shape_searchs import SearchShapeImagenetResNet
    assert config.search_mode == 'shape', 'invalid search-mode : {:}'.format( config.search_mode )
    if config.arch == 'resnet':
      return SearchShapeImagenetResNet(config.block_name, config.layers, config.deep_stem, config.class_num)
    else:
      raise ValueError('invalid model config : {:}'.format(config))
  else:
    raise ValueError('invalid dataset in the model config : {:}'.format(config))


def load_net_from_checkpoint(checkpoint):
  assert osp.isfile(checkpoint), 'checkpoint {:} does not exist'.format(checkpoint)
  checkpoint   = torch.load(checkpoint)
  model_config = dict2config(checkpoint['model-config'], None)
  model        = obtain_model(model_config)
  model.load_state_dict(checkpoint['base-model'])
  return model
