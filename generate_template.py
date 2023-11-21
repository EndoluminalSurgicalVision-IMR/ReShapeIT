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

import logging
import plyfile
import skimage.measure
import time
import torch
from collections import OrderedDict


def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        ply_filename_out,
        offset=None,
        scale=None,
        level=0.0,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


def create_mesh(model, filename, embedding=None, N=128, max_batch=64 ** 3, offset=None, scale=None,
                level=0.0, get_color=True):
    start = time.time()
    ply_filename = filename

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()[None, ...]
        samples[head: min(head + max_batch, num_samples), 3] = (
            model.get_template_field(sample_subset).squeeze().detach().cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f s" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
        level
    )


p = configargparse.ArgumentParser()
p.add_argument('--logging_root', type=str, default='./recon/template', help='root for logging')
p.add_argument('--config', type=str, default='$YOUR_CONFIG_HERE$', help='generation configuration')
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
    create_mesh(model,
                filename=os.path.join(root_path,"template_" + meta_params['experiment_name'] + "_reso" + str(opt.resolution)),
                N=opt.resolution, level=opt.level)
