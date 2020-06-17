import pdb
from gu import *
from util import *
from Gaussianization_Flows.models.flow_model import Net
import seaborn.apionly as sns
import matplotlib.pyplot as plt
import matplotlib
import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torchvision
from torch.utils.data import DataLoader, Subset
import torch.distributions as tdist
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torch
import numpy as np
import os
import sys
import math
import logging
import argparse
import time
import copy
from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)


matplotlib.use('Agg')
plt.style.use('seaborn-paper')


# add device

parser = argparse.ArgumentParser()
# data
parser.add_argument('--gu_num', type=int, default=8,
                    help='Components of GU clusters.')
parser.add_argument('--dataset', default='GU', help='Which dataset to use.')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')

# model parameters
parser.add_argument(
    '--input_size',
    type=int,
    default=1,
    help='Input size in a model.')
parser.add_argument(
    '--layer',
    type=int,
    default=5,
    help="Total number of Gaussianization layers")
parser.add_argument(
    '--kde_num',
    type=int,
    default=1,
    help='Stacking multiple KDE layers before each rotation layer')
parser.add_argument(
    '--total_datapoints',
    type=int,
    default=100,
    help="Total number of data points for each KDE layer")
parser.add_argument(
    '--usehouseholder',
    action='store_true',
    help='Train rotation matrix using householder reflection')
parser.add_argument(
    '--multidim_kernel',
    action='store_true',
    help='Use multiple dimension bandwidth kernel')

parser.add_argument('--early_stopping', type=int, default=10)
# training params
parser.add_argument(
    '--batch_size',
    type=int,
    default=2048,
    help='Batch size in training.')
parser.add_argument('--niters', type=int, default=50000,
                    help='Total iteration numbers in training.')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0,
    help='Weight decay in Adam.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1 in Adam.')
parser.add_argument(
    '--beta2',
    type=float,
    default=0.999,
    help='Beta 2 in Adam.')
parser.add_argument(
    '--process_size',
    type=int,
    default=100,
    help="Process size")
parser.add_argument(
    '--adjust_step',
    type=int,
    default=10000,
    help="Decrease learning rate after a couple steps")

parser.add_argument(
    '--log_interval',
    type=int,
    default=1000,
    help='How often to show loss statistics and save models/samples.')

parser.add_argument(
    '--cuda',
    type=int,
    default=0,
    help='Number of CUDA to use if available.')
parser.add_argument(
    '--eval_size',
    type=int,
    default=100000,
    help='Sample size in evaluation.')


def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
    log_jacob = log_jacob.sum()
    loss = -(log_probs + log_jacob)

    if size_average:
        loss /= u.size(0)
    return loss


def train(model, dataloader, optimizer, args):

    start = time.time()
    running_loss = 0.

    DATA = dataloader.get_sample(args.total_datapoints)
    DATA = DATA.view(DATA.shape[0], -1)

    for i in range(1, args.niters + 1):
        model.train()
        if (i + 1) % args.adjust_step == 0:
            logger.info("Adjusting learning rate, divide lr by 2")
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2.

        x = dataloader.get_sample(args.batch_size)
        x = x.view(x.shape[0], -1).to(args.device)

        data, log_det, _ = model.forward(x, DATA, args.process_size)
        loss = flow_loss(data, log_det)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            model.eval()
            with torch.no_grad():
                # save model
                cur_state_path = os.path.join(model_path, str(i))
                torch.save(model, cur_state_path + '.pth')

                real = dataloader.get_sample(args.eval_size)
                prior = model.base_dist.sample((args.eval_size,))
                fake = model.sampling(
                    DATA,
                    prior,
                    process_size=args.eval_size,
                    sample_num=args.eval_size)
                w_distance_real = w_distance(real, fake)

                logger.info(
                    f'Iter {i} / {args.niters}, Time {round(time.time() - start, 4)},  '
                    f'w_distance_real: {w_distance_real}, loss {round(running_loss / args.log_interval, 5)}')

                real_sample = real.cpu().data.numpy().squeeze()
                fake_sample = fake.cpu().data.numpy().squeeze()

                # plot.
                plt.cla()
                fig = plt.figure(figsize=(FIG_W, FIG_H))
                ax = fig.add_subplot(111)
                ax.set_facecolor('whitesmoke')
                ax.grid(True, color='white', linewidth=2)

                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.get_xaxis().tick_bottom()

                # _sample = np.concatenate([real_sample, fake_sample])
                kde_num = 200
                min_real, max_real = min(real_sample), max(real_sample)
                kde_width_real = kde_num * \
                    (max_real - min_real) / args.eval_size
                min_fake, max_fake = min(fake_sample), max(fake_sample)
                kde_width_fake = kde_num * \
                    (max_fake - min_fake) / args.eval_size
                sns.kdeplot(
                    real_sample,
                    bw=kde_width_real,
                    label='Data',
                    color='green',
                    shade=True,
                    linewidth=6)
                sns.kdeplot(
                    fake_sample,
                    bw=kde_width_fake,
                    label='Model',
                    color='orange',
                    shade=True,
                    linewidth=6)

                ax.set_title(
                    f'True EM Distance: {w_distance_real}.',
                    fontsize=FONTSIZE)
                ax.legend(loc=2, fontsize=FONTSIZE)
                ax.set_ylabel('Estimated Density by KDE', fontsize=FONTSIZE)
                ax.tick_params(axis='x', labelsize=FONTSIZE * 0.7)
                ax.tick_params(
                    axis='y',
                    labelsize=FONTSIZE * 0.5,
                    direction='in')

                cur_img_path = os.path.join(image_path, str(i) + '.jpg')
                plt.tight_layout()

                plt.savefig(cur_img_path)
                plt.close()

                start = time.time()
                running_loss = 0

# --------------------
# Run
# --------------------


if __name__ == '__main__':
    args = parser.parse_args()

    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)

    search_type = 'manual'
    experiment = f'gu{args.gu_num}/gaussianization_flow/{args.niters}'

    model_path = os.path.join(rootPath, search_type, 'models', experiment)
    image_path = os.path.join(rootPath, search_type, 'images', experiment)
    makedirs(model_path, image_path)
    log_path = model_path + '/logs'
    logger = get_logger(log_path)

    # setup device
    # args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # load data
    if args.gu_num == 8:
        dataloader = GausUniffMixture(
            n_mixture=args.gu_num,
            mean_dist=10,
            sigma=2,
            unif_intsect=1.5,
            unif_ratio=1.,
            device=args.device,
            extend_dim=False)
    else:
        dataloader = GausUniffMixture(
            n_mixture=args.gu_num,
            mean_dist=5,
            sigma=0.1,
            unif_intsect=5,
            unif_ratio=3,
            device=args.device,
            extend_dim=False)
    args.input_size = 1

    # model
    model = Net(
        args.total_datapoints,
        args.layer,
        args.input_size,
        args.kde_num,
        multidim_kernel=args.multidim_kernel,
        usehouseholder=args.usehouseholder)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(
            args.beta1,
            args.beta2))

    logger.info('Start training...')
    train(model, dataloader, optimizer, args)
    logger.info('Finish All...')
