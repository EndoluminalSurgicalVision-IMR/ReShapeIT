# -*- coding: utf-8 -*-

'''
template interaction module
'''

import json
import os
import time
import re
import numpy as np
import trimesh
from natsort import natsorted
from pycpd import RigidRegistration
from scipy.io import savemat
from scipy.spatial import KDTree

import mesh_to_sdf.surface_point_cloud
from mesh_to_sdf.surface_point_cloud import BadMeshException
from mesh_to_sdf.utils import scale_to_unit_cube, scale_to_unit_sphere, get_raster_points, check_voxels

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=None, scan_count=100,
                            scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1

    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans(mesh, bounding_radius=bounding_radius, scan_count=scan_count,
                                                     scan_resolution=scan_resolution,
                                                     calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count,
                                                    calculate_normals=calculate_normals)
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))

class MySDFCalculator(SurfacePointCloud):
    def sample_points_on_surface(self, number_of_points=500000, use_scans=True, sign_method='normal',
                                 normal_sample_count=11, min_size=0, return_gradients=True):
        query_points = []
        surface_sample_count = number_of_points
        surface_points = self.get_random_surface_points(surface_sample_count, use_scans=use_scans)
        query_points.append(surface_points)
        query_points = np.asarray(query_points, dtype=np.float32).squeeze()

        if sign_method == 'normal':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count,
                                          return_gradients=return_gradients)
        elif sign_method == 'depth':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=True, return_gradients=return_gradients)
        else:
            raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
        if return_gradients:
            sdf, gradients = sdf

        if min_size > 0:
            model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
            if model_size < min_size:
                raise BadMeshException()

        if return_gradients:
            return query_points, sdf, gradients
        else:
            return query_points, sdf

    def sample_points_off_surface(self, number_of_points=500000, use_scans=True, sign_method='normal',
                                  normal_sample_count=11, min_size=0, return_gradients=False):
        query_points = []
        unit_sphere_sample_count = number_of_points
        unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
        query_points.append(unit_sphere_points)
        query_points = np.asarray(query_points, dtype=np.float32).squeeze()

        if sign_method == 'normal':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count,
                                          return_gradients=return_gradients)
        elif sign_method == 'depth':
            sdf = self.get_sdf_in_batches(query_points, use_depth_buffer=True, return_gradients=return_gradients)
        else:
            raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
        if return_gradients:
            sdf, gradients = sdf

        if min_size > 0:
            model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
            if model_size < min_size:
                raise BadMeshException()

        if return_gradients:
            return query_points, sdf, gradients
        else:
            return query_points, sdf


def template_interaction_module(template_in_path, root_dir, K=75):
    '''
    template_in_path: template model path (obj file)
    root_dir: initial results path
    K: Top K% Points selection
    '''
    filename_list = []
    for file in natsorted(os.listdir(os.path.join(root_dir, 'rawpredmeshes'))):
        filename_list.append(re.split('\.', file)[0])
    for filename in filename_list:
        pred_raw_mesh_in_path = os.path.join(root_dir, 'rawpredmeshes', filename + '.obj')
        sdf_points_root_dir = os.path.join(root_dir, 'sdfpointsPred', filename)
        mkdir(sdf_points_root_dir)
        surface_pts_n_normal_dir = os.path.join(sdf_points_root_dir, 'surface_pts_n_normal')
        mkdir(surface_pts_n_normal_dir)
        surface_pts_n_normal_out_path = os.path.join(surface_pts_n_normal_dir, filename + '.mat')
        free_space_pts_dir = os.path.join(sdf_points_root_dir, 'free_space_pts')
        mkdir(free_space_pts_dir)
        free_space_pts_out_path = os.path.join(free_space_pts_dir, filename + '.mat')
        template_mesh: trimesh.Trimesh = trimesh.load(template_in_path)
        pred_mesh: trimesh.Trimesh = trimesh.load(pred_raw_mesh_in_path)
        template_mesh_vertices = np.array(template_mesh.vertices)
        pred_mesh_vertices = np.array(pred_mesh.vertices)
        reg = RigidRegistration(X=template_mesh_vertices, Y=pred_mesh_vertices)
        reg.register()
        pred_mesh_register_vertices = reg.TY
        scale = reg.s
        R = reg.R.tolist()
        t = reg.t.tolist()
        pred_mesh_register: trimesh.Trimesh = trimesh.Trimesh(
            vertices=pred_mesh_register_vertices,
            faces=pred_mesh.faces)
        pred_mesh_register_vertices_array = np.array(pred_mesh_register_vertices)
        P1 = reg.P1
        indices = P1 >= np.percentile(P1, q=K)
        chosenpointsarray = pred_mesh_register_vertices_array[indices, ...]
        spc = get_surface_point_cloud(pred_mesh_register, 'scan', 1, 100, 400, 10000000, calculate_normals=True)
        my_sdf = MySDFCalculator(mesh=spc.mesh, points=spc.points, normals=spc.normals, scans=spc.scans)
        on_surface_points, on_surface_sdf, on_surface_gradients = my_sdf.sample_points_on_surface(
            number_of_points=800000)
        off_surface_points, off_surface_sdf = my_sdf.sample_points_off_surface(number_of_points=200000)
        off_surface_sdf = np.expand_dims(off_surface_sdf, axis=1)
        p = np.concatenate((on_surface_points, on_surface_gradients), axis=1)
        p_sdf = np.concatenate((off_surface_points, off_surface_sdf), axis=1)
        orign_surface_pts = p[:, 0:3]
        chosenpointsarray_kdtree = KDTree(data=chosenpointsarray)
        dd, ii = chosenpointsarray_kdtree.query(orign_surface_pts)
        dd_quantile = float(chosenpointsarray.shape[0] / 800000 * 100)
        dd_threshold = np.percentile(dd, q=dd_quantile)
        dd_indices = dd <= dd_threshold
        template_chosen_onsurface_pts = p[dd_indices, ...]
        num_chosen_surface_pts = template_chosen_onsurface_pts.shape[0]
        num_chosen_offsurface_pts = int(num_chosen_surface_pts / 4)
        offsurface_pts_indices = np.random.choice(p_sdf.shape[0], size=num_chosen_offsurface_pts, replace=False)
        template_chosen_offsurface_pts = p_sdf[offsurface_pts_indices, ...]
        savemat(surface_pts_n_normal_out_path, {"p": template_chosen_onsurface_pts})
        savemat(free_space_pts_out_path, {"p_sdf": template_chosen_offsurface_pts})
