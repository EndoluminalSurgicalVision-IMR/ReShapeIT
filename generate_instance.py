# -*- coding: utf-8 -*-

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import configargparse
import re
import numpy as np

import torch
import modules, utils
from ISMM_net import ISMM
import sdf_meshing


def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


p = configargparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default='./recon', help='root for logging')
p.add_argument('--config', type=str, default='configs/generate/pancreas_20231027.yml', help='path for the config file')
p.add_argument('--subject_idx', type=parse_idx_range, help='index of subject to generate')
p.add_argument('--level', type=float, default=0, help='level of iso-surface for marching cube')
p.add_argument('--resolution', type=int, default=256, help='resolution')

if __name__ == '__main__':
    opt = p.parse_args()
    with open(os.path.join(opt.config), 'r') as stream:
        meta_params = yaml.safe_load(stream)
    model = ISMM(**meta_params)
    model.load_state_dict(torch.load(meta_params['checkpoint_path']))
    model.cuda()
    root_path = os.path.join(opt.logging_root, meta_params['experiment_name'])
    utils.cond_mkdir(root_path)
    for idx in opt.subject_idx:
        print('generate_instance:', idx)
        sdf_meshing.create_mesh_caselatent(model,
                                           os.path.join(root_path, 'test%04d' % idx + '_reso' + str(opt.resolution)),
                                           subject_idx=idx, N=opt.resolution, level=opt.level, get_color=True)
