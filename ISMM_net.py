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


class ISMM(nn.Module):
    def __init__(self, num_instances, latent_dim=128, model_type='sine', hyper_hidden_layers=1,
                 hyper_hidden_features=256, hidden_num=128, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_dim_alpha = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)
        self.latent_codes_alpha = nn.Embedding(num_instances, self.latent_dim_alpha)
        nn.init.normal_(self.latent_codes_alpha.weight, mean=0, std=0.01)
        self.template_field = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num,
                                                   num_hidden_layers=3, in_features=3,
                                                   out_features=1)
        self.deform_net = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num,
                                               num_hidden_layers=3, in_features=3 + self.latent_dim_alpha,
                                               out_features=4)
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_layers=hyper_hidden_layers,
                                      hyper_hidden_features=hyper_hidden_features,
                                      hypo_module=self.deform_net)

        print(self)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self, instance_idx):
        embedding = self.latent_codes(instance_idx)

        return embedding

    def get_template_coords(self, coords, embedding, instance_idx):
        with torch.no_grad():
            latent = self.latent_codes_alpha(instance_idx)
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params, latent=latent)
            deformation = model_output['model_out'][:, :, :3]
            new_coords = coords + deformation

            return new_coords

    def get_template_field(self, coords):
        with torch.no_grad():
            model_in = {'coords': coords}
            model_output = self.template_field(model_in)

            return model_output['model_out']

    def get_template_coords_and_correction(self, coords, embedding):
        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            deformation = model_output['model_out'][:, :, :3]
            correction = model_output['model_out'][:, :, 3:]
            new_coords = coords + deformation
            return new_coords, correction

    def get_output_from_new_coords_and_correction(self, coords, correction):
        with torch.no_grad():
            model_input_temp = {'coords': coords}
            model_output_temp = self.template_field(model_input_temp)
            return model_output_temp['model_out'] + correction

    def inference(self, coords, embedding, instance_idx):
        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            latent = self.latent_codes_alpha(instance_idx)
            model_output = self.deform_net(model_in, params=hypo_params, latent=latent)
            deformation = model_output['model_out'][:, :, :3]
            correction = model_output['model_out'][:, :, 3:]
            new_coords = coords + deformation
            model_input_temp = {'coords': new_coords}
            model_output_temp = self.template_field(model_input_temp)
            return model_output_temp['model_out'] + correction

    def forward(self, model_input, gt, is_train=True, **kwargs):
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

        model_out = {'model_in': model_output['model_in'], 'grad_temp': grad_temp, 'grad_deform': grad_deform,
                     'model_out': sdf_final, 'latent_vec': embedding, 'latent_case': latent,
                     'hypo_params': hypo_params, 'grad_sdf': grad_sdf, 'sdf_correct': correction}
        if is_train:
            losses = imss_train_loss(model_out, gt, loss_grad_deform=kwargs['loss_grad_deform'],
                                     loss_grad_temp=kwargs['loss_grad_temp'], loss_correct=kwargs['loss_correct'])
        else:
            losses = imss_finetune_loss(model_out, gt)
        return losses
