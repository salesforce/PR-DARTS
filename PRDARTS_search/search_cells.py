##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from cell_operations import OPS
import pdb


class MixedOp(nn.Module):

  def __init__(self, space, C, stride, affine, track_running_stats):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in space:
      op = OPS[primitive](C, C, stride, affine, track_running_stats)
      self._ops.append(op)
      print(primitive)
    # pdb.set_trace()

  def forward_prdarts(self, x, weights, indexs):
    if len(indexs) == 0:
      return self._ops[0](x)
    output = self._ops[indexs[0].item()](x) * weights[indexs[0].item()]
    for i in range(1,len(indexs)):
      output += self._ops[indexs[i].item()](x) * weights[indexs[i].item()]
    return output



# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
class NASNetSearchCell(nn.Module):

  def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
    super(NASNetSearchCell, self).__init__()
    self.reduction = reduction
    self.op_names  = deepcopy(space)
    if reduction_prev: self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
    else             : self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
    self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
    self._steps = steps
    self._multiplier = multiplier

    # pdb.set_trace()
    self._ops = nn.ModuleList()
    self.edges     = nn.ModuleDict()
    for i in range(self._steps):
      for j in range(2+i):
        node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
        # print(node_str)
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(space, C, stride, affine, track_running_stats)
        self.edges[ node_str ] = op
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)


  def forward_prdarts(self, s0, s1, weightss):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    for i in range(self._steps):
      clist = []
      for j, h in enumerate(states):
        node_str = '{:}<-{:}'.format(i, j)
        op = self.edges[ node_str ]
        weights = weightss[ self.edge2index[node_str] ]
        index = torch.nonzero(weights)#.data.numpy()
        clist.append( op.forward_prdarts(h, weights, index) )
      if len(clist):
        states.append( sum(clist) )
    return torch.cat(states[-self._multiplier:], dim=1)

