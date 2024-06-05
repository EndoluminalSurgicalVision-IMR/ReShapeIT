# -*- coding: utf-8 -*-
import torch
from torch.autograd import grad
from torch import nn
import modules
from meta_modules import HyperNetwork
from loss import *
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



## ISMM with Valid Template
class ISMM(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1,
                 hyper_hidden_features=256, hidden_num=128, **kwargs):
        super(ISMM, self).__init__()
        self.latent_dim = latent_dim
        self.latent_dim_alpha = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        self.latent_codes_alpha = nn.Embedding(num_instances, self.latent_dim_alpha)
        nn.init.normal_(self.latent_codes_alpha.weight, mean=0, std=0.01)
        self.latent_code_template = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=True)
        nn.init.normal_(self.latent_code_template, mean=0, std=0.01)
        self.latent_code_template_alpha = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=True)
        nn.init.normal_(self.latent_code_template_alpha, mean=0, std=0.01)
        self.template_field = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num,
                                                   num_hidden_layers=3, in_features=3,
                                                   out_features=1)
        self.deform_net = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num,
                                               num_hidden_layers=3, in_features=3 + self.latent_dim_alpha,
                                               out_features=4)
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers,
                                      hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.deform_net)
        self.template_points_num = 100000
        self.register_buffer('template_surface_points',
                             torch.randn([self.template_points_num, 3], dtype=torch.float32) / 3)
        self.grid_res = 128
        self.voxel_size = 2.0 / float(self.grid_res)

    def get_latent_code(self, instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding

    def forward(self, model_input, gt, **kwargs):
        instance_idx = model_input['instance_idx']
        coords = model_input['coords']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        latent = self.latent_codes_alpha(instance_idx)
        model_output = self.deform_net(model_input, params=hypo_params, latent=latent)
        deformation = model_output['model_out'][:, :, :3]
        correction = model_output['model_out'][:, :, 3:]
        new_coords = coords + deformation
        x = model_output['model_in']
        u = deformation[:, :, 0]
        v = deformation[:, :, 1]
        w = deformation[:, :, 2]
        grad_outputs = torch.ones_like(u)
        laplacian_u = laplace(u, x)
        laplacian_v = laplace(v, x)
        laplacian_w = laplace(w, x)
        grad_deform = torch.stack([laplacian_u, laplacian_v, laplacian_w], dim=2)
        model_input_temp = {'coords': new_coords}
        model_output_temp = self.template_field(model_input_temp)
        sdf = model_output_temp['model_out']
        grad_temp = torch.autograd.grad(sdf, [new_coords], grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
        sdf_final = sdf + correction
        grad_sdf = torch.autograd.grad(sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0]

        batchsize = coords.shape[0]
        points_num = coords.shape[1]
        template_surface_points_num = points_num // 2
        surface_rand_idcs = np.random.choice(self.template_points_num, points_num // 2)
        free_rand_idcs = np.random.choice(self.template_points_num, points_num // 4)
        selected_surface_points = self.template_surface_points[surface_rand_idcs]
        selected_surface_points.requires_grad_()
        with torch.no_grad():
            off_surface_points = torch.rand([points_num // 4, 3], device=coords.device) * 2 - 1
            template_coords = torch.cat([selected_surface_points,
                                         off_surface_points,
                                         self.template_surface_points[free_rand_idcs] + torch.randn(
                                             [points_num // 4, 3], device=coords.device) * 0.0025,
                                         ], dim=0).unsqueeze(0)
        template_coords.requires_grad_()
        template_coords_model_in = {'coords': template_coords}
        self.latent_code_template = self.latent_code_template.unsqueeze(0)
        template_hypo_params = self.hyper_net(self.latent_code_template)
        self.latent_code_template_alpha = self.latent_code_template_alpha.unsqueeze(0)

        template_deform_modelout = self.deform_net(template_coords_model_in, params=template_hypo_params,
                                                   latent=self.latent_code_template_alpha)
        deformation_template = template_deform_modelout['model_out'][:, :, :3]
        correction_template = template_deform_modelout['model_out'][:, :, 3:]

        ## model_out
        model_out = {'model_in': model_output['model_in'], 'grad_temp': grad_temp, 'grad_deform': grad_deform,
                     'model_out': sdf_final, 'latent_vec': embedding,
                     'hypo_params': hypo_params, 'grad_sdf': grad_sdf, 'sdf_correct': correction,
                     'template_code': self.latent_code_template,
                     'deformation_template': deformation_template,
                     'correction_template': correction_template,
                     'latent_case': latent,
                     'latent_template': self.latent_code_template_alpha}
        if is_train:
            losses = ismm_train_loss(model_out, gt,
                                     loss_grad_deform=kwargs['loss_grad_deform'],
                                     loss_grad_temp=kwargs['loss_grad_temp'],
                                     loss_correct=kwargs['loss_correct'])
        else:
            losses = ismm_finetune_loss(model_out, gt)

        return losses

    def sample_template_points(self):
        with torch.no_grad():
            N = self.grid_res
            voxel_origin = [-1, -1, -1]
            voxel_size = 2.0 / (N - 1)
            overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor()).cuda()
            samples = torch.zeros(N ** 3, 3).cuda()
            samples[:, 2] = overall_index % N
            samples[:, 1] = (overall_index.long() / N) % N
            samples[:, 0] = ((overall_index.long() / N) / N) % N
            samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
            samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
            samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
            model_input_temp = {'coords': samples.unsqueeze(0)}
            template_sdf = self.template_field(model_input_temp)['model_out']
            sdf_values = template_sdf.reshape(N, N, N)
            sdf_values = sdf_values.unsqueeze(0)
            verts, faces = marching_cubes(sdf_values.transpose(1, 3).contiguous(), 0)
            verts = verts[0]
            faces = faces[0]
            verts *= voxel_size
            verts += torch.tensor(voxel_origin, dtype=verts.dtype, device=verts.device)
            alpha_gen = torch.tensor(np.random.dirichlet((1,) * 3, self.template_points_num)).cuda()
            face_idx = torch.randperm(self.template_points_num).cuda() % faces.shape[0]
            selected_verts = (alpha_gen[:, :, None] * verts[faces.long()[face_idx]]).sum(dim=1)
            self.template_surface_points[:] = selected_verts

    def get_template_field(self, coords):
        with torch.no_grad():
            model_in = {'coords': coords}
            template_embedding = self.latent_code_template
            template_embedding = template_embedding.unsqueeze(0)
            template_hypo_params = self.hyper_net(template_embedding)
            template_model_output = self.deform_net(model_in, params=template_hypo_params,
                                                    latent=self.latent_code_template_alpha.unsqueeze(0))
            template_deformation = template_model_output['model_out'][:, :, :3]
            template_correction = template_model_output['model_out'][:, :, 3:]
            new_coords = coords + template_deformation
            model_input_temp = {'coords': new_coords}
            model_output_temp = self.template_field(model_input_temp)
            return model_output_temp['model_out'] + template_correction

    def get_template_coords(self, coords, embedding, instance_idx):
        with torch.no_grad():
            latent = self.latent_codes_alpha(instance_idx)
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params, latent=latent)
            deformation = model_output['model_out'][:, :, :3]
            new_coords = coords + deformation
            return new_coords

    def get_templateself_coords(self, coords):
        with torch.no_grad():
            model_in = {'coords': coords}
            template_embedding = self.latent_code_template
            template_embedding = template_embedding.unsqueeze(0)
            template_hypo_params = self.hyper_net(template_embedding)
            template_model_output = self.deform_net(model_in, params=template_hypo_params,
                                                    latent=self.latent_code_template_alpha.unsqueeze(0))
            template_deformation = template_model_output['model_out'][:, :, :3]
            template_new_coords = coords + template_deformation
            return template_new_coords

