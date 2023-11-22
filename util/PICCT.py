# -*- coding: utf-8 -*-

'''
PICCT module for data preparation
'''

import SimpleITK as sitk
from pycpd import RigidRegistration
import numpy as np
import trimesh


def rigid_align_meshes(a_mesh, m_mesh):
    anchor_vertices = np.asarray(a_mesh.vertices)
    moving_vertices = np.asarray(m_mesh.vertices)
    reg = RigidRegistration(X=anchor_vertices, Y=moving_vertices)
    reg.register()
    new_vertices = reg.TY
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=m_mesh.faces)
    scale = reg.s
    R = reg.R.tolist()
    t = reg.t.tolist()
    trans_dict = {
        's': scale,
        'R': R,
        't': t,
    }
    return new_mesh, trans_dict


def T_img_to_wor(im_sitk):
    spacing = im_sitk.GetSpacing()
    direction = im_sitk.GetDirection()
    origin = im_sitk.GetOrigin()
    img_to_wor = np.eye(4)
    img_to_wor[0, :3] = np.asarray(direction[:3]) * spacing[0]
    img_to_wor[1, :3] = np.asarray(direction[3:6]) * spacing[1]
    img_to_wor[2, :3] = np.asarray(direction[6:9]) * spacing[2]
    img_to_wor[0, 3] = origin[0]
    img_to_wor[1, 3] = origin[1]
    img_to_wor[2, 3] = origin[2]
    return img_to_wor


def T_wor_to_can(anc, mov):
    new_mesh, trans_dict = rigid_align_meshes(anc, mov)
    scale = trans_dict['s']
    R = np.array(trans_dict['R'])
    t = np.array(trans_dict['t'])
    wor_to_can = (scale, R, t)
    return wor_to_can
