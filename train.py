# -*- coding: utf-8 -*-

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import io
import numpy as np
import dataset, utils, training_loop, loss, modules, meta_modules

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from ISMM_net import ISMM

p = configargparse.ArgumentParser()
p.add_argument('--config', type=str, default='$YOUR_CONFIG_FILE$', help='training configuration.')
p.add_argument('--train_split', type=str, default='', help='training subject names.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--batch_size', type=int, default=32, help='training batch size.')
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for.')
p.add_argument('--epochs_til_checkpoint', type=int, default=5,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='',
               help='training data path.')
p.add_argument('--latent_dim', type=int, default=128, help='latent code dimension.')
p.add_argument('--hidden_num', type=int, default=128, help='hidden layer dimension of deform-net.')
p.add_argument('--loss_grad_deform', type=float, default=5, help='loss weight for deformation smoothness prior.')
p.add_argument('--loss_grad_temp', type=float, default=1e2, help='loss weight for normal consistency prior.')
p.add_argument('--loss_correct', type=float, default=1e2, help='loss weight for minimal correction prior.')
p.add_argument('--num_instances', type=int, default=5, help='numbers of instance in the training set.')
p.add_argument('--expand', type=float, default=-1, help='expansion of shape surface.')
p.add_argument('--max_points', type=int, default=200000, help='number of surface points for each epoch.')
p.add_argument('--on_surface_points', type=int, default=4000, help='number of surface points for each iteration.')


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_schedules=None,
          **kwargs):
    print('Training Info:')
    print('num_instances:\t\t', kwargs['num_instances'])
    print('batch_size:\t\t', kwargs['batch_size'])
    print('epochs:\t\t\t', epochs)
    print('learning rate:\t\t', lr)
    for key in kwargs:
        if 'loss' in key:
            print(key + ':\t', kwargs[key])
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    writer = SummaryWriter(summaries_dir)
    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                losses = model(model_input, gt, **kwargs)
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)
                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss
                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                if not total_steps % steps_til_summary:
                    torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, 'model_current.pth'))
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                        epoch, train_loss, time.time() - start_time))
                total_steps += 1
        torch.save(model.module.cpu().state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


if __name__ == '__main__':
    opt = p.parse_args()
    if opt.config == '':
        meta_params = vars(opt)
    else:
        with open(opt.config, 'r') as stream:
            meta_params = yaml.safe_load(stream)
    sdf_dataset = dataset.PointCloudMulti(root_dir=meta_params['point_cloud_path'],
                                          max_num_instances=meta_params['num_instances'], **meta_params)
    dataloader = DataLoader(sdf_dataset, shuffle=True, collate_fn=sdf_dataset.collate_fn,
                            batch_size=meta_params['batch_size'], pin_memory=False, num_workers=0, drop_last=True)
    print('Total subjects: ', sdf_dataset.num_instances)
    meta_params['num_instances'] = sdf_dataset.num_instances
    model = ISMM(**meta_params)
    model = nn.DataParallel(model)
    model.cuda()
    root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
    utils.cond_mkdir(root_path)
    with io.open(os.path.join(root_path, 'model.yml'), 'w', encoding='utf8') as outfile:
        yaml.dump(meta_params, outfile, default_flow_style=False, allow_unicode=True)
    train(model=model, train_dataloader=dataloader, model_dir=root_path, **meta_params)
