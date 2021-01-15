from os import path as osp
import numpy as np
from typing import List, Text
import torch
import pdb

from config_utils import dict2config
# from genotypes import Structure as CellStructure
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
  if super_type == 'nasnet-super':
    return NASNetworkPRDARTS(config.C, config.N, config.steps, config.multiplier, \
                    config.stem_multiplier, config.num_classes, config.space, config.affine, config.track_running_stats)
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name) -> List[Text]:
  if xtype == 'cell':
    from cell_operations import SearchSpaceNames
    assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
    return SearchSpaceNames[name]
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))
