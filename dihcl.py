# Copyright 2020-present, Tianyi Zhou.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import os
import sys
import time
import shutil
import random
import copy
import math

import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image, ImageEnhance, ImageOps
from progress.bar import Bar as Bar
from sklearn.metrics import pairwise_distances
# from utils import CIFAR10PolicyAll, RandAugment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import models.cifar as models
# model_names = sorted(name for name in models.__dict__
#     if not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Curriculum Learning by Dynamic Instance Hardness (DIHCL)')
parser.add_argument('-d', '--dataset', type=str, default='CIFAR10')
    # choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'SVHN', 'STL10'])
parser.add_argument('-spath', '--save_path', default='result', type=str, metavar='PATH',
                    help='path to save results')
parser.add_argument('-dpath', '--data_path', default='../data', type=str, metavar='PATH',
                    help='path to dataset directory')
parser.add_argument('--trialID', default='00/', type=str, metavar='PATH',
                    help='path to specific trial')

# Optimization parameters
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='total epochs to run')
parser.add_argument('--schedule', type=int, nargs='+', default=[0, 5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300],
                        help='splitting points (epoch number) for multiple episodes of training')
parser.add_argument('--selfsupervise_cut_epoch', default=200, type=int,
                    help='epoch to stop self-supervision and centrality max')
parser.add_argument('--explore_cut_episode', default=5, type=int,
                    help='episode to stop update features, centrality, and increasing initial learning rate')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1024, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=2.0e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_min', default=5.0e-4, type=float,
                    metavar='LR', help='ending learning rate of each episode')
parser.add_argument('--lr_min_decay', default=0.8, type=float,
                    help='decay factor for ending learning rate after each episode')
parser.add_argument('--lr_max', default=1.0e-1, type=float,
                    metavar='LR', help='starting learning rate')
parser.add_argument('--lr_max_decay', default=0.9, type=float,
                    help='decay factor applied to starting learning rate after each episode')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='beta distribution parameter for Mix-Up')
parser.add_argument('--alpha_rate', default=1.1, type=float,
                    help='rate for mixup alpha')

# Neural Nets Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn')
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--repeat', type=int, default=3, help='number of blocks (WideResNet)')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--drop', '--dropout', default=0.3, type=float, metavar='Dropout', help='Dropout rate')

# GPU and CPU
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--random_state', default=32, type=int, help='random state/seed')

# DIHCL parameters
parser.add_argument('--k', default=1.0, type=float, 
                    help='(initial) ratio for subset selected')
parser.add_argument('--dk', default=0.1, type=float, 
                    help='increase rate of k')
parser.add_argument('--mk', default=0.3, type=float,
                    help='maximum/minimum of k')

parser.add_argument('--select_ratio', default=0.5, type=float,
                    help='(initial) ratio of further selection by centrality (when use_centrality=True)')
parser.add_argument('--select_ratio_rate', default=1.1, type=float,
                    help='multiplication factor applied to select_ratio for each episode')

parser.add_argument('--tmpt', default=20., type=float,
                    help='1/temperature used in updating EXP3 probability')
parser.add_argument('--tmpt_rate', default=0.8, type=float,
                    help='rate of temperature')

parser.add_argument('--mod', default=0.05, type=float,
                    help='(initial) weight of DIH (when use_centrality=True, since the weight for centrality is 1 - mod)')
parser.add_argument('--mod_rate', default=1.5, type=float,
                    help='multiplcation factor applied to mod every episode')

parser.add_argument('--ema_decay', default=0.9, type=float,
                    help='decay factor of exponential moving average')

parser.add_argument('--consistency', default=0., type=float,
                    help='weight of consistency regularization (provided by mean teacher) in the objective')
parser.add_argument('--consistency_rate', default=0.9, type=float,
                    help='multiplcation factor applied to the consistency weight every episode')

parser.add_argument('--contrastive', default=0., type=float,
                    help='weight of contrastive regularization (dictionary provided by mean teacher) in the objective')
parser.add_argument('--contrastive_rate', default=0.9, type=float,
                    help='multiplcation factor applied to the contrastive weight every episode')

# choices of different settings
parser.add_argument('--use_curriculum', action='store_true',
                    help='if using curriculum learning or not')
parser.add_argument('--use_mean_teacher', action='store_true',
                    help='maintain a time moving average of the model (ensemble over time) as a teacher for consistency regularization')
parser.add_argument('--bandits_alg', default='EXP3', type=str,
                    help='which bandits alg to use: EXP3, UCB, TS (Thampson sampling)')
parser.add_argument('--use_loss_as_feedback', action='store_true',
                    help='set True if use loss as instant feedback, set False if use the change of loss between epochs (only applies to EXP3 and UCB in DIHCL, TS always takes the change in prediction correctness as instant feedback)')
parser.add_argument('--use_random_subsample', action='store_true',
                    help='random subsample a subset before running centrality max to further select samples')
parser.add_argument('--use_centrality', action='store_true',
                    help='use centrality score in rating samples for more diversity')
parser.add_argument('--use_kernel_centrality', action='store_true',
                    help='compute centrality defined on a kernel matrix, if False, compute centrality based on the penultimate-layer features')
parser.add_argument('--use_noisylabel', action='store_true',
                    help='use training samples with noisy labels')
parser.add_argument('--label_noise_type', default='symmetric', type=str,
                    help='which type of label noise when --use_noisylabel: symmetric, pairflip')
parser.add_argument('--label_noise_rate', default=0.6, type=float,
                    help='noise rate on labels when --use_noisylabel')
parser.add_argument('--num_aug', default=0, type=int,
                    help='number of extra augmentations used for test/inference')
parser.add_argument('--save_dynamics', action='store_true',
                    help='save training dynamics (require large memory)')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_ids = list(map(int, args.gpu_id.split(',')))
device_ids = range(len(device_ids))
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

# name a folder by the trialID to store all the results
folder = args.trialID

def main():

    global best_acc
    if not os.path.isdir(args.save_path):
        mkdir_p(args.save_path)
    if not os.path.isdir(os.path.join(args.save_path, folder)):
        mkdir_p(os.path.join(args.save_path, folder))
    if not os.path.isdir(args.data_path):
        mkdir_p(args.data_path)

    # prepare datasets to train
    print('==> Preparing dataset %s' % args.dataset)

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        trans_mean = (0.4914, 0.4822, 0.4465)
        trans_std = (0.2470, 0.2435, 0.2616)
        cutout_size = 16
        input_shape = (1, 3, 32, 32)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            CIFAR10Policy(),
            # CIFAR10PolicyAll(),
            # RandAugment(),
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize(trans_mean, trans_std),
        ])

    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        trans_mean = (0.5071, 0.4865, 0.4409)
        trans_std = (0.2673, 0.2564, 0.2762)
        cutout_size = 16
        input_shape = (1, 3, 32, 32)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            CIFAR10Policy(),
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize(trans_mean, trans_std),
        ])

    elif args.dataset == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10
        trans_mean = (0.5, 0.5, 0.5)
        trans_std = (0.5, 0.5, 0.5)
        cutout_size = 20
        input_shape = (1, 3, 32, 32)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize(trans_mean, trans_std),
        ])

        args.lr_max *= 0.1
        args.lr_min *= 0.2

    elif args.dataset == 'stl10':
        dataloader = datasets.STL10
        num_classes = 10
        trans_mean = (0.5, 0.5, 0.5)
        trans_std = (0.5, 0.5, 0.5)
        cutout_size = 32
        input_shape = (1, 3, 96, 96)
        jitter_params = 0.4
        light_params = 0.1

        transform_train = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=jitter_params, contrast=jitter_params, saturation=jitter_params, hue=0), 
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            Lighting(light_params, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(trans_mean, trans_std),
        ])

        args.epochs *= 4
        args.selfsupervise_cut_epoch *= 4
        args.schedule = [sch*4 for sch in args.schedule]
        print(args.epochs, args.schedule)

    elif args.dataset == 'fmnist' or args.dataset == 'kmnist' or args.dataset == 'mnist':

        if args.dataset == 'mnist':
            dataloader = datasets.MNIST
            trans_mean = np.array([0.1307])
            trans_std = np.array([0.3081])
            cutout_size = 8
        elif args.dataset == 'fmnist':
            dataloader = datasets.FashionMNIST
            trans_mean = np.array([0.2860])
            trans_std = np.array([0.3530])
            cutout_size = 8
        elif args.dataset == 'kmnist':
            dataloader = datasets.KMNIST
            trans_mean = np.array([0.1904])
            trans_std = np.array([0.3475])
            cutout_size = 14
        num_classes = 10
        input_shape = (1, 1, 28, 28)
        args.lr_max *= 0.2
        args.lr_min *= 0.2

        transform_train = transforms.Compose([
            # transforms.Pad(4),
            # transforms.RandomCrop(28),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            cutout(cutout_size, 1.0, False),
            transforms.ToTensor(),
            transforms.Normalize(trans_mean, trans_std),
        ])

    transform_train2 = TransformTwice(transform_train, args.use_mean_teacher)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(trans_mean, trans_std),
    ])

    if args.dataset == 'stl10' or args.dataset == 'svhn':
        trainset = dataloader(root=args.data_path, split='train', download=True, transform=transform_train2)
        testset = dataloader(root=args.data_path, split='test', download=False, transform=transform_test)
        trainset0 = dataloader(root=args.data_path, split='train', download=True, transform=transform_test)
        if args.num_aug > 0:
            testset_aug = dataloader(root=args.data_path, split='test', download=False, transform=transform_train)
            trainset_aug = dataloader(root=args.data_path, split='train', download=False, transform=transform_train)
        n_train = len(trainset.labels)
    else:
        trainset = dataloader(root=args.data_path+'/'+args.dataset, train=True, download=True, transform=transform_train2)
        testset = dataloader(root=args.data_path+'/'+args.dataset, train=False, download=False, transform=transform_test)
        trainset0 = dataloader(root=args.data_path+'/'+args.dataset, train=True, download=True, transform=transform_test)
        if args.num_aug > 0:
            testset_aug = dataloader(root=args.data_path+'/'+args.dataset, train=False, download=False, transform=transform_train)
            trainset_aug = dataloader(root=args.data_path+'/'+args.dataset, train=True, download=False, transform=transform_train)
        n_train = len(trainset.targets)
        target_copy = trainset.targets

        if args.use_noisylabel:
            # noisy label
            true_target_copy = target_copy.copy()
            target_copy, actual_noise_rate = noisify(train_labels=np.asarray(target_copy), noise_type=args.label_noise_type, noise_rate=args.label_noise_rate, random_state=args.random_state, nb_classes=num_classes)
            target_copy = target_copy.tolist()
            trainset.targets = target_copy
            noise_or_not_label = np.transpose(target_copy)==np.transpose(true_target_copy)

        if use_cuda:
            target_copy = torch.cuda.LongTensor(target_copy)
        else:
            target_copy = torch.LongTensor(target_copy)

    trainloader_val = data.DataLoader(trainset0, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    if args.num_aug > 0:
        testloader_aug = data.DataLoader(testset_aug, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        trainloader_aug = data.DataLoader(trainset_aug, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    else:
        testloader_aug = None
        trainloader_aug = None

    # Initialize Neural Nets Model
    print("==> creating model '{}'".format(args.arch))
    if args.dataset == 'fmnist' or args.dataset == 'kmnist' or args.dataset == 'mnist':
        model = torch.nn.DataParallel(models.PreActResNet34_MNIST().cuda(), device_ids=device_ids)
        if args.use_mean_teacher:
            ema_model = torch.nn.DataParallel(models.PreActResNet34_MNIST().cuda(), device_ids=device_ids)
            for param in ema_model.parameters():
                param.detach_()
            init_as_ema(ema_model, model)
        else:
            ema_model = None
    else:
        model = torch.nn.DataParallel(wrn(input_shape = input_shape, num_classes=num_classes, depth=args.depth, 
                                    widen_factor=args.widen_factor, repeat = args.repeat, dropRate=args.drop, bias=True).cuda(), device_ids=device_ids)
        if args.use_mean_teacher:
            ema_model = torch.nn.DataParallel(wrn(input_shape = input_shape, num_classes=num_classes, depth=args.depth, 
                                    widen_factor=args.widen_factor, repeat = args.repeat, dropRate=args.drop, bias=True).cuda(), device_ids=device_ids)
            for param in ema_model.parameters():
                param.detach_()
            init_as_ema(ema_model, model)
        else:
            ema_model = None
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    # initialize log files
    title = args.dataset + args.arch
    logger = Logger(os.path.join(args.save_path, folder + 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    log_backup = open(os.path.join(args.save_path, folder + 'log_backup.txt'), 'w')
    log_result = ()

    # initialization of variables storing training results
    best_acc = 0.0
    schedule_idx = 0
    train_sample_num = 0
    batches_counter = 0
    rand_train_set_old = None
    batches_this_episode = np.ceil(float(n_train) / float(args.train_batch)) * (args.schedule[schedule_idx+1] - args.schedule[schedule_idx])
    epoch_time = np.zeros(args.epochs)

    # initialize DIH
    acc_flips = np.zeros(n_train)
    acc_diff_loss = np.zeros(n_train)
    acc_diff_loss_raw = np.zeros(n_train)
    acc_loss_raw = np.zeros(n_train)
    acc_loss = np.zeros(n_train)
    all_pace = np.zeros(n_train)
    selects = np.zeros(n_train)
    TS_A = 2.0*np.ones(n_train)
    TS_B = 2.0*np.ones(n_train)
    sigma_UCB = np.sqrt(2.0 * np.log(float(args.epochs)))

    if args.save_dynamics:
        select_dy_train = ()
        correct_dy_train = ()
        loss_dy_train = ()
        pred_dy_train = ()
        correct_dy_test = ()
        loss_dy_test = ()
        pred_dy_test = ()
    
    # training epochs
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # At the begining of each episode, before training begins, we need to update some training parameters
        if epoch == args.schedule[min(schedule_idx + 1, len(args.schedule) - 1)]:

            batches_counter = 0
            schedule_idx += 1
            if schedule_idx < args.explore_cut_episode:
                args.lr_max /= args.lr_max_decay
            else:
                args.lr_max *= args.lr_max_decay
                args.lr_min *= args.lr_min_decay

            args.consistency *= args.consistency_rate
            if args.consistency < 1.e-2 or epoch >= args.selfsupervise_cut_epoch:
                args.consistency = 0.
            args.contrastive *= args.contrastive_rate
            if args.contrastive < 1.e-3 or epoch >= args.selfsupervise_cut_epoch:
                args.contrastive = 0.

            if args.use_curriculum:

                # update k0 (random subsampling budget)
                args.k = max([args.k * (1.0-args.dk), args.mk])
                args.select_ratio = min(args.select_ratio_rate * args.select_ratio, 1.0)
                k0 = int(np.floor(args.k * n_train))
                if args.use_centrality:
                    subset_size = int(args.select_ratio * k0)
                    args.mod *= args.mod_rate
                    batches_per_epoch = np.ceil(float(subset_size) / float(args.train_batch))
                else:
                    batches_per_epoch = np.ceil(float(k0) / float(args.train_batch))
                args.mod *= args.mod_rate
                args.tmpt = max(1., args.tmpt_rate * args.tmpt)
                args.alpha = min(1., args.alpha_rate * args.alpha)

                # moving average for DIHCL
                acc_flips *= args.ema_decay
                acc_diff_loss *= args.ema_decay
                all_pace *= args.ema_decay
                acc_loss *= args.ema_decay
                acc_diff_loss_raw *= args.ema_decay
                acc_loss_raw *= args.ema_decay
                TS_A *= args.ema_decay
                TS_B *= args.ema_decay

                # update the size of selected subset and the number of training batches in the current episode
                batches_this_episode = batches_per_epoch * (args.schedule[schedule_idx+1] - args.schedule[schedule_idx])

                # one-pass inference of all training samples
                train_loss, train_acc, all_loss, all_correct, _, train_fea, train_pred, _ = test(trainloader_val, trainloader_aug, model if ema_model is None else ema_model, criterion, epoch, args.num_aug, use_cuda)

                # compute instant feedback
                all_loss_epoch = all_loss.cpu().numpy().astype(float)
                diff_loss = np.minimum(np.abs(all_loss_epoch - all_loss_old), 10.0)
                all_correct_epoch = all_correct.cpu().numpy().astype(float)
                diff_flips = np.abs(all_correct_epoch - all_correct_old)
                all_pred_epoch = train_pred.cpu().numpy()

                # save dynamics
                if args.save_dynamics:
                    loss_dy_train = loss_dy_train + (all_loss_epoch, )
                    correct_dy_train = correct_dy_train + (all_correct_epoch, )
                    pred_dy_train = pred_dy_train + (all_pred_epoch, )

                # store the current feedback for next iteration comparison
                all_loss_old = all_loss_epoch.copy()
                all_correct_old = all_correct_epoch.copy()
                all_pred_old = all_pred_epoch.copy()

                # update DIH
                acc_diff_loss_raw += diff_loss
                acc_loss_raw += all_loss_epoch
                if args.bandits_alg == 'EXP3':          
                    acc_diff_loss += diff_loss / all_pace
                    acc_loss += all_loss_epoch / args.lr
                    acc_flips += diff_flips / all_pace 
                elif args.bandits_alg == 'UCB':
                    acc_diff_loss = acc_diff_loss_raw / selects
                    acc_diff_loss += 1.0e-3 * np.mean(acc_diff_loss) * sigma_UCB * np.sqrt(1.0 / selects)
                    acc_loss = acc_loss_raw / selects
                    acc_loss += 1.0e-3 * np.mean(acc_loss) * sigma_UCB * np.sqrt(1.0 / selects)
                elif args.bandits_alg == 'TS':
                    TS_A += diff_flips
                    TS_B += 1.0 - diff_flips
                all_pace = np.zeros(n_train)

                # update the centrality of every training sample using pairwise similarity in the penultimate-layer feature space
                # it measures how representative of each sample for the data distribution and encourages diversity in data selection
                # it can be replaced by facility location function or other submodular functions if submodulax maximization is available
                if args.use_centrality and epoch <= args.schedule[args.explore_cut_episode]:
                    modular_estimate = 1.0
                    centrality_estimate = 1.0
                    if args.use_kernel_centrality:
                        if epoch > args.schedule[1]:
                            del train_sims, centrality
                        train_sims, centrality = compute_sims(np.clip(train_fea, 1.0e-10, 1.0e+10), sigma = 20., metric = 'cos')
                    else:
                        train_fea, centrality = compute_feamat(np.clip(train_fea, 1.0e-10, 1.0e+10))

            else:

                batches_per_epoch = np.ceil(float(n_train) / float(args.train_batch))
                batches_this_episode = batches_per_epoch * (args.schedule[schedule_idx+1] - args.schedule[schedule_idx])

            # initialize mean teacher model by the current model before training begins
            if args.use_mean_teacher:
                if epoch == args.schedule[1]:
                    print('Initialize mean teacher!')
                    init_as_ema(ema_model, model)
                else:
                    print('Re-initialize model by Ema!')
                    init_as_ema(model, ema_model, 1.e-2)

        # warm starting episode (epochs from args.schedule[0] to args.schedule[1])
        if epoch < args.schedule[1] or (not args.use_curriculum):

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            # one-pass training on the whole training set
            old_lr = args.lr
            indices = np.random.permutation(n_train)
            train_sample_num += n_train
            trainloader = data.DataLoader(Subset(trainset, indices), batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
            if epoch < args.schedule[1]:
                train_loss, train_acc, all_loss_per_sample, all_correct_per_sample, all_pred_per_sample, _, _, batches_counter, pace_delta = train(trainloader, model, ema_model, criterion, optimizer, batches_counter, batches_this_episode, use_cuda, args.ema_decay, 0., 0., args.alpha, lr_schedule = 'linear')
            else:
                train_loss, train_acc, all_loss_per_sample, all_correct_per_sample, all_pred_per_sample, _, _, batches_counter, pace_delta = train(trainloader, model, ema_model, criterion, optimizer, batches_counter, batches_this_episode, use_cuda, args.ema_decay, 0., 0., args.alpha, lr_schedule = 'cosine')

            # compute instant feedback
            all_loss_epoch = np.zeros(n_train)
            all_loss_epoch[indices] = all_loss_per_sample.cpu().numpy().astype(float)
            if epoch > 0:
                diff_loss = np.minimum(np.abs(all_loss_epoch - all_loss_old), 10.0)
            all_correct_epoch = np.zeros(n_train)
            all_correct_epoch[indices] = all_correct_per_sample.cpu().numpy().astype(float)
            if epoch > 0:
                diff_flips = np.abs(all_correct_epoch - all_correct_old)
            all_pred_epoch = np.zeros(n_train)
            all_pred_epoch[indices] = all_pred_per_sample.cpu().numpy()

            # save dynamics
            if args.save_dynamics:
                loss_dy_train = loss_dy_train + (all_loss_epoch, )
                correct_dy_train = correct_dy_train + (all_correct_epoch, )
                pred_dy_train = pred_dy_train + (all_pred_epoch, )

            # store the current feedback for next iteration comparison
            all_loss_old = all_loss_epoch.copy()
            all_correct_old = all_correct_epoch.copy()
            all_pred_old = all_pred_epoch.copy()

            # update DIH
            selects += 1.0
            if epoch > 0:
                acc_diff_loss_raw += diff_loss
                acc_loss_raw += all_loss_epoch
                if args.bandits_alg == 'EXP3':
                    acc_diff_loss += diff_loss / all_pace
                    acc_loss += all_loss_epoch / old_lr
                    acc_flips += diff_flips / all_pace
                elif args.bandits_alg == 'UCB':
                    acc_diff_loss = acc_diff_loss_raw / selects
                    acc_diff_loss += 1.0e-3 * np.mean(acc_diff_loss) * sigma_UCB * np.sqrt(1.0 / selects)
                    acc_loss = acc_loss_raw / selects
                    acc_loss += 1.0e-3 * np.mean(acc_loss) * sigma_UCB * np.sqrt(1.0 / selects)                      
                elif args.bandits_alg == 'TS':
                    TS_A += diff_flips
                    TS_B += 1.0 - diff_flips
            all_pace[indices] = 0.0
            all_pace += pace_delta

        # every episode except the warm starting episode (after epoch args.schedule[1]) selects a subset of data for training
        if args.use_curriculum and epoch >= args.schedule[1]:

            print('\nEpoch: [%d | %d] LR: %f, subset size k: %d' % (epoch + 1, args.epochs, state['lr'], k0))

            # compute sampling probability from current DIH
            acc_loss_mean = np.mean(acc_loss)
            acc_diff_loss_mean = np.mean(acc_diff_loss)
            if args.bandits_alg == 'TS':# TS in DIHCL
                utility = np.random.beta(TS_A, TS_B)
            else: # applies for EXP3 or UCB in DIHCL
                if args.use_loss_as_feedback:
                    utility = acc_loss / acc_loss_mean
                else:                        
                    utility = acc_diff_loss / acc_diff_loss_mean
            rand_prob = args.tmpt * utility
            rand_prob = np.exp(rand_prob - np.max(rand_prob))
            rand_prob /= np.sum(rand_prob)
            print('rand_prob min, max: ', rand_prob.min(), rand_prob.max())

            # sampling a subset according to rand_prob
            if args.use_centrality and epoch < args.selfsupervise_cut_epoch:
                mod = args.mod * (centrality_estimate / modular_estimate)
                if args.use_random_subsample:
                    rand_train_set = np.random.choice(n_train, k0, p = rand_prob, replace=False)
                    train_subset = np.argpartition(mod * rand_prob[rand_train_set] + centrality[rand_train_set], -subset_size)[-subset_size:]
                    rand_train_set = rand_train_set[train_subset]
                else:
                    p = mod * rand_prob + centrality
                    rand_train_set = np.random.choice(n_train, subset_size, p = p / p.sum(), replace=False)
                centrality_estimate, modular_estimate = centrality[rand_train_set].sum(), rand_prob[rand_train_set].sum()     
            else:
                rand_train_set = np.random.choice(n_train, k0, p = rand_prob, replace=False)

            # train model on the selected subset rand_train_set
            train_sample_num += len(rand_train_set)
            trainselectloader = data.DataLoader(trainset, batch_size=args.train_batch, sampler=SubsetSampler(rand_train_set), num_workers=args.workers)
            if rand_train_set_old is not None:
                print('new samples selected comparing to previous epoch:', len(np.setdiff1d(rand_train_set, rand_train_set_old, assume_unique=True)))
            rand_train_set_old = rand_train_set.copy()
            old_lr = args.lr
            train_loss1, train_acc, loss_rest, correct_rest, pred_rest, _, logit_rest, batches_counter, pace_delta2 = train(trainselectloader, model, ema_model, criterion, optimizer, batches_counter, batches_this_episode, use_cuda, args.ema_decay, args.consistency, args.contrastive, args.alpha, lr_schedule = 'cosine')
            if epoch > args.schedule[min(schedule_idx, len(args.schedule)-1)]:
                train_loss = train_loss1

            # compute instant feedback
            all_loss_epoch = all_loss_old.copy()
            loss_rest = loss_rest.cpu().numpy().astype(float)
            all_loss_epoch[rand_train_set] = loss_rest
            diff_loss = np.minimum(np.abs(all_loss_epoch[rand_train_set] - all_loss_old[rand_train_set]), 10.0)

            all_correct_epoch = all_correct_old.copy()
            all_correct_epoch[rand_train_set] = correct_rest.cpu().numpy().astype(float)
            diff_flips = np.abs(all_correct_epoch[rand_train_set] - all_correct_old[rand_train_set])

            all_pred_epoch = all_pred_old.copy()
            all_pred_epoch[rand_train_set] = pred_rest.cpu().numpy()

            # save dynamics
            if args.save_dynamics:
                loss_dy_train = loss_dy_train + (all_loss_epoch, )
                correct_dy_train = correct_dy_train + (all_correct_epoch, )
                pred_dy_train = pred_dy_train + (all_pred_epoch, )

            # store the current feedback for next iteration comparison
            all_loss_old = all_loss_epoch.copy()
            all_correct_old = all_correct_epoch.copy()
            all_pred_old = all_pred_epoch.copy()

            # update DIH
            selects[rand_train_set] += 1.0
            acc_diff_loss_raw[rand_train_set] += diff_loss
            acc_loss_raw[rand_train_set] += loss_rest
            if epoch > args.schedule[schedule_idx]:
                if args.bandits_alg == 'EXP3':
                    rc1 = rand_prob[rand_train_set] + 1.0e-8
                    rc = 1.e+4 * np.power(all_pace[rand_train_set], 1.e-2) * rc1
                    acc_diff_loss[rand_train_set] += diff_loss / rc
                    acc_loss[rand_train_set] += loss_rest / rc1
                    acc_flips[rand_train_set] +=  diff_flips / rc
                elif args.bandits_alg == 'UCB':
                    acc_diff_loss[rand_train_set] = acc_diff_loss_raw[rand_train_set] / selects[rand_train_set] 
                    acc_diff_loss[rand_train_set] += 1.0e-3 * acc_diff_loss_mean * sigma_UCB * np.sqrt(1.0 / selects[rand_train_set])
                    acc_loss[rand_train_set] = acc_loss_raw[rand_train_set] / selects[rand_train_set] 
                    acc_loss[rand_train_set] += 1.0e-3 * acc_loss_mean * sigma_UCB * np.sqrt(1.0 / selects[rand_train_set])
                elif args.bandits_alg == 'TS':
                    TS_A[rand_train_set] += diff_flips
                    TS_B[rand_train_set] += 1.0 - diff_flips
            all_pace[rand_train_set] = 0.0
            all_pace += pace_delta2     

        # inference on test set
        epoch_time[epoch] = time.time() - epoch_start_time
        test_loss, test_acc, all_loss_test, all_correct_test, _, test_fea, all_pred_test, _ = test(testloader, None, model, criterion, epoch, 0, use_cuda)
        if args.use_mean_teacher:
            ema_test_loss, ema_test_acc, _, _, _, _, _, _ = test(testloader, None, ema_model, criterion, epoch, 0, use_cuda)
        if args.save_dynamics:
            loss_dy_test = loss_dy_test + (all_loss_test.cpu().numpy().astype(float), )
            correct_dy_test = correct_dy_test + (all_correct_test.cpu().numpy().astype(float), )
            pred_dy_test = pred_dy_test + (all_pred_test.cpu().numpy(), )
            select_dy_train = select_dy_train + (selects.copy(), )
        torch.cuda.empty_cache()

        # save results to log files
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        if args.use_mean_teacher:
            result_iter = np.asarray([epoch, train_loss, test_loss, train_acc, test_acc, ema_test_loss, ema_test_acc, epoch_time[epoch], train_sample_num])
        else:
            result_iter = np.asarray([epoch, train_loss, test_loss, train_acc, test_acc, epoch_time[epoch], train_sample_num])
        log_result = log_result + (result_iter, )
        log_backup.write("\t".join(map(str, result_iter))+'\n')
        log_backup.flush()
        best_acc = np.max([best_acc, test_acc, ema_test_acc])

    # save result and log files
    log_backup.close()
    logger.close()
    np.savetxt(os.path.join(args.save_path, folder + 'log_result.txt'), np.vstack(log_result))
    if args.save_dynamics:
        np.savez(os.path.join(args.save_path, folder + args.dataset + '_dy.npz'), 
                loss_train = np.stack(loss_dy_train), loss_test = np.stack(loss_dy_test), 
                correct_train = np.stack(correct_dy_train), correct_test = np.stack(correct_dy_test), 
                pred_train = np.stack(pred_dy_train), pred_test = np.stack(pred_dy_test), 
                select_train = np.stack(select_dy_train), time_train = epoch_time)     
    print('Best acc:', best_acc)      

#--------------------------------DATA AUTOAUGMENTATION-------------------------------

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        Subranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        Subfunc = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = Subfunc[operation1]
        self.magnitude1 = Subranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = Subfunc[operation2]
        self.magnitude2 = Subranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

# -----------------------------DATASET and DATALOADER-----------------------------------

class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class SubsetSampler(sampler.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

#------------------------------Neural Network Structures (WideResNet)--------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)#, momentum = 0.001
        self.relu1 = nn.LeakyReLU(inplace=False)#, negative_slope = 0.02)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)#, momentum = 0.001
        self.relu2 = nn.LeakyReLU(inplace=False)#, negative_slope = 0.02)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class Flatten(nn.Module):
    def __init__(self, d):
        super(Flatten, self).__init__()
        self.d = d

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class WideResNet(nn.Module):
    def __init__(self, input_shape, depth, num_classes, widen_factor=1, dropRate=0.0, repeat=3, bias=True):
        super(WideResNet, self).__init__()
        nChannels = [16]
        if widen_factor > 20:
            for ii in range(repeat):
                nChannels.append(2**ii * widen_factor)
        else:
            for ii in range(repeat):
                nChannels.append(2**ii * 16 * widen_factor)
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(input_shape[1], nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.blocks = [NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)]
        for ii in range(repeat - 1):
            self.blocks.append(NetworkBlock(n, nChannels[ii+1], nChannels[ii+2], block, 2, dropRate))
        self.blocks = nn.ModuleList(self.blocks)
        self.bn1 = nn.BatchNorm2d(nChannels[-1])
        self.relu = nn.LeakyReLU(inplace=True)
        self.flatten = Flatten(nChannels[-1])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, num_classes, bias = bias)
        self.nChannels = nChannels[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if bias:
                    m.bias.data.zero_()

    def _forward_conv(self, x):
        out = self.conv1(x)
        for i, blk in enumerate(self.blocks):
            out = blk(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, output_size=1)
        return out        

    def forward(self, x):
        x = self._forward_conv(x)
        outfea = self.flatten(x)
        x = self.fc(outfea)
        return outfea, x

def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model

#------------------------------Train and Test-----------------------------------

def train(trainloader, model, ema_model, criterion, optimizer, step, total_steps, use_cuda, ema_decay, consistency = 20., contrastive = 1., alpha = 0., lr_schedule = 'cosine', dict_size = 10, mul = 1.):
    
    # switch to train mode
    model.train()

    logsoftmax = nn.LogSoftmax(dim=1).cuda()
    softmax = nn.Softmax(dim=1).cuda()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_correct = ()
    all_loss = ()
    if alpha > 0:
        all_loss_b = ()
    all_pred = ()
    all_prob = ()
    all_logit = ()
    pace = 0.0
    bcount = 0
    idx = 0
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, ((inputs, ema_inputs), targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if bcount == 0 and mul > 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            optimizer.zero_grad()
            if ema_model is not None:
                update_ema_variables(model, ema_model, ema_decay, step)
            bcount = mul

        if lr_schedule == 'linear':
            adjust_linear_learning_rate_step(optimizer, step, total_steps)
        elif lr_schedule == 'cosine':
            adjust_cosine_learning_rate_step(optimizer, step, total_steps, mul)
        elif lr_schedule == 'constant':
            adjust_constant_learning_rate_step(optimizer)
        pace += args.lr

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        bs = len(targets)

        if alpha > 0 and bs > 1:
            inputs, targets, targets_b, lam, mixup_index = mixup_data(inputs, targets, alpha = alpha)
            inputs, targets, targets_b = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(targets_b)
        else:
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outfea, outputs = model(inputs)
        if ema_model is not None:

            if use_cuda:
                ema_inputs = ema_inputs.cuda()
            with torch.no_grad():
                ema_fea, ema_outputs = ema_model(ema_inputs)
            all_logit = all_logit + (ema_outputs, )
            ema_prob = softmax(0.4 * (ema_outputs - ema_outputs.max(dim=1).values[:,None]))
            all_prob = all_prob + (ema_prob, )
            ema_fea = torch.autograd.Variable(ema_fea.detach().data)

            if consistency > 0:
                cons_loss = F.mse_loss(outfea, ema_fea, reduction='none').mean(dim=1).mean()

            if contrastive > 0:
                if batch_idx == 0:
                    ema_fea_dict = ema_fea / ema_fea.norm(p=2,dim=1).unsqueeze(dim=1)
                elif batch_idx < dict_size:
                    ema_fea_dict = torch.cat((ema_fea / ema_fea.norm(p=2,dim=1).unsqueeze(dim=1), ema_fea_dict))
                else:
                    ema_fea_dict = torch.cat((ema_fea / ema_fea.norm(p=2,dim=1).unsqueeze(dim=1), ema_fea_dict[:-bs]))
                cont_logprob = -logsoftmax(torch.mm(outfea / outfea.norm(p=2,dim=1).unsqueeze(dim=1), ema_fea_dict.t()) / 0.1)
                cont_loss = cont_logprob[range(bs), range(bs)].mean()
            
        # compute loss + regularizations
        loss = -logsoftmax(outputs).gather(1, targets.view(-1,1)).squeeze()
        if alpha > 0 and bs > 1:
            loss_b = -logsoftmax(outputs).gather(1, targets_b.view(-1,1)).squeeze()
            loss_mixup = lam * loss.data
            loss_mixup[mixup_index] += (1. - lam) * loss_b.data
            all_loss = all_loss + (loss_mixup, )
            loss = lam * loss.mean() + (1. - lam) * loss_b.mean()
            if ema_model is not None:
                if consistency > 0:
                    cons_loss = lam * cons_loss + (1. - lam) * F.mse_loss(outfea, ema_fea[mixup_index], reduction='none').mean(dim=1).mean()
                if contrastive > 0:
                    cont_loss = lam * cont_loss + (1. - lam) * cont_logprob[range(bs), mixup_index].mean()
        else:
            if bs > 1:
                all_loss = all_loss + (loss.data, )
            else:
                dim = len(loss.size())
                if dim == 0:
                    all_loss = all_loss + (loss.data.unsqueeze(0), )
                elif dim == 1:
                    all_loss = all_loss + (loss.data, )
                elif dim == 2:
                    all_loss = all_loss + (loss.data.squeeze(0), )
            loss = loss.mean()

        if ema_model is not None:
            if consistency > 0:
                loss = loss + consistency * cons_loss
            if contrastive > 0:
                loss = loss + contrastive * cont_loss

        # measure accuracy and record loss
        prec1, prec5, correct, pred = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), bs)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)
        all_correct = all_correct + (correct, )
        all_pred = all_pred + (pred, )

        if mul > 1:
            loss.backward()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema_variables(model, ema_model, ema_decay, step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        step += 1
        idx += bs
        bcount -= 1

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | lr: {lr: .7f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    lr = state['lr'], 
                    )
        bar.next()
    bar.finish()

    all_loss = torch.cat(all_loss, 0).squeeze()
    all_correct = torch.cat(all_correct, 0).squeeze()
    all_pred = torch.cat(all_pred, 0).squeeze()
    all_prob = torch.cat(all_prob, 0)
    all_logit = torch.cat(all_logit, 0)

    return (losses.avg, top1.avg, all_loss, all_correct, all_pred, all_prob, all_logit, step-1, pace)

def test(testloader, augloader, model, criterion, epoch, num_aug, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    softmax = nn.Softmax(dim=1).cuda()

    if augloader is not None and num_aug > 0:
        aug_iters = [iter(augloader) for i in range(num_aug)]

    all_loss = ()
    all_correct = ()
    all_logit = ()
    all_outfea = ()
    all_pred = ()
    all_prob = ()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        def get_output(dl_iter):

            inputs, targets = next(dl_iter)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            with torch.no_grad():
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[-1]
            return outputs

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        bs = inputs.size(0)

        with torch.no_grad():
            # compute output
            outfea, outputs = model(inputs)
            all_outfea = all_outfea + (outfea, )
            prob = softmax(0.4 * (outputs - outputs.max(dim=1).values[:,None]))
            loss = -(prob.log().data).gather(1, targets.data.view(-1,1))
            all_loss = all_loss + (loss, )
            all_prob = all_prob + (prob.data.gather(1, targets.data.view(-1,1)), )
            all_logit = all_logit + (outputs, )

            # TTA
            if augloader is not None and num_aug > 0:
                for aug_iter in aug_iters:
                    o = get_output(aug_iter)
                    outputs.add_(o)

            # measure accuracy and record loss
            prec1, prec5, correct, pred = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.mean().item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)
            all_correct = all_correct + (correct, )
            all_pred = all_pred + (pred, )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    all_loss = torch.cat(all_loss, 0).squeeze()
    all_correct = torch.cat(all_correct, 0).squeeze()
    all_pred = torch.cat(all_pred, 0).squeeze()
    all_logit = torch.cat(all_logit, 0)
    all_outfea = torch.cat(all_outfea, 0).data.cpu().numpy().astype(float)
    all_prob = torch.cat(all_prob, 0).squeeze()

    return (losses.avg, top1.avg, all_loss, all_correct, all_logit, all_outfea, all_pred, all_prob)

# ------------------------------------noisy-label-----------------------------------

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise

def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate

# --------------------------------------utility-------------------------------------

def swarp(val,alpha,beta):
    swarp_epsilon = 1e-30
    nonzero_indicators = (val > swarp_epsilon)
    if np.any(nonzero_indicators) > 0:
        val[nonzero_indicators] = 1.0 / (1 + (val[nonzero_indicators]**(1/np.log2(beta)) - 1 )**alpha)
    return val

def compute_sims(data, sigma=1.0, metric = 'l2', fl_alpha = 2., fl_beta = 0.75, fl_gamma = 2., fl_lambda = 0.9, fl_gsrow = False, fl_cent_op = True, fl_cent_mn = False, fl_cent_sm = False):

    # compute similarity matrix

    if metric == "ham":
        data = data > 0
        metric_name = "hamming"
    elif metric == "hammean":
        data = data > np.mean(data)
        metric_name = "hamming"
    elif metric == "corr":
        metric_name = "correlation"
    elif metric == "maha":
        metric_name = "mahalanobis"
    elif metric == "jac":
        data = data > 0
        metric_name = "jaccard"
    elif metric == "jacmean":
        data = data > np.mean(data)
        metric_name = "jaccard"
    elif metric == "l2":
        metric_name = "euclidean"
    elif metric == "cos":
        metric_name = "cosine"

    sims = pairwise_distances(data, metric=metric_name).astype(np.float32)

    for ii in range(data.shape[0]):
        sims[ii, ii] = 0

    if metric == "cos":
        sim = 1. - sims
    else:
        sims = np.exp(-sims / (sigma * np.median(sims, axis = 0)[:, np.newaxis]))

    sims += sims.T
    sims /= 2.0

    # First, do a max over columns of the matrix.
    if fl_gamma != 1.0 or fl_alpha != 1.0 or fl_beta != 0.5:
        # Then we need to make sure each column has max value of 1 before doing any gamma correction or swarping.
        t0 = time.time()
        if fl_gsrow:
            sims_max = sims.max(axis=1)
            # Next, normalize sims so that the max entry in each row is 1.
            sims = (sims.T / sims_max).T
        else:
            sims_max = sims.max(axis=0)
            # Next, normalize sims so that the max entry in each column is 1.
            sims /= sims_max
        # Next do potential swarping followed by potential gamma correction of the matrix.
        if fl_gamma != 1.0:
            if fl_alpha != 1.0 or fl_beta != 0.5:
                sims = swarp(sims,fl_alpha,fl_beta)**fl_gamma
            else:
                sims = sims**fl_gamma
        else:
            if fl_alpha != 1.0 or fl_beta != 0.5:
                sims = swarp(sims,fl_alpha,fl_beta)
        # factor back in the relative column maxes, so each column's relative importance is the same.
        if fl_gsrow:
            sims = (sims.T * sims_max).T
        else:
            sims *= sims_max
        print('computing gamma-swarp took ', time.time()-t0, ' seconds')


    fl_centrality = None
    if fl_lambda < 1.0:
        t0 = time.time()
        # create a modular function of centrality
        fl_centrality = np.zeros(sims.shape[0]);
        if fl_cent_op:
            fl_centrality = np.sum(sims.dot(sims.transpose()),axis=1)
            fl_centrality = fl_centrality / fl_centrality.sum()
        elif fl_cent_mn:
            fl_centrality = np.min(sims,axis=1)
            fl_centrality = fl_centrality / fl_centrality.sum()
        elif fl_cent_sm:
            fl_centrality = np.sum(sims,axis=1)
            fl_centrality = fl_centrality / fl_centrality.sum()
        else:
            print('failed to compute centrality')
        print('computing centrality took ', time.time()-t0, ' seconds')

    return sims, fl_centrality

def compute_feamat(feature_matrix, fe_alpha = 2., fe_beta = 0.75, fe_gamma = 2., fe_lambda = 0.9, fe_gsrow = False, fe_gsmin = True, fe_cent_op = True, fe_cent_mn = False, fe_cent_sm = False, fe_use_entropy_centrality = False):

    # preprocess feature matrix

    if fe_gamma != 1.0 or fe_alpha != 1.0 or fe_beta != 0.5:
        # Then we need to make sure each column has max value of 1 before doing any gamma correction or swarping.
        # First, do a max over columns of the matrix.
        t0 = time.time()
        if fe_gsrow:
            # Next, normalize feature_matrix so that the max entry in each row is 1.
            if fe_gsmin:
                feature_matrix_min = feature_matrix.min(axis=1)
                feature_matrix = feature_matrix.T - feature_matrix_min.T
                feature_matrix_max = feature_matrix.max(axis=1) + 1.0e-5
                feature_matrix = (feature_matrix / feature_matrix_max).T
            else:
                feature_matrix_max = feature_matrix.max(axis=1)
                feature_matrix = (feature_matrix.T / feature_matrix_max).T
        else:
            # Next, normalize feature_matrix so that the max entry in each column is 1.
            if fe_gsmin:
                feature_matrix_min = feature_matrix.min(axis=0)
                feature_matrix -= feature_matrix_min
                feature_matrix_max = feature_matrix.max(axis=0) + 1.0e-5
                feature_matrix /= feature_matrix_max
            else:
                feature_matrix_max = feature_matrix.max(axis=0)
                feature_matrix /= feature_matrix_max
        # Next do potential swarping followed by potential gamma correction of the matrix.
        if fe_gamma != 1.0:
            if fe_alpha != 1.0 or fe_beta != 0.5:
                feature_matrix = swarp(feature_matrix,fe_alpha,fe_beta)**fe_gamma
            else:
                feature_matrix = feature_matrix**fe_gamma
        else:
            if fe_alpha != 1.0 or fe_beta != 0.5:
                feature_matrix = swarp(feature_matrix,fe_alpha,fe_beta)
        if fe_gsrow:
            # factor back in the relative row maxes, so each rows's relative importance is the same.
            feature_matrix = (feature_matrix.T * feature_matrix_max).T
        else:
            # factor back in the relative column maxes, so each column's relative importance is the same.
            feature_matrix = feature_matrix * feature_matrix_max
        print('computing gamma-swarp took ', time.time()-t0, ' seconds')

    assert not np.any(feature_matrix < 0.), "feature_matrix contains negative elements!"

    fe_centrality = None
    if fe_lambda < 1.0:
        # create a modular function of centrality
        t0 = time.time()
        fe_centrality = np.zeros(feature_matrix.shape[0]);

        if fe_use_entropy_centrality:
            max_entropy = np.log2(feature_matrix.shape[1])
            for ii in range(feature_matrix.shape[0]):
                fe_centrality[ii] = max_entropy - entropy(feature_matrix[ii].T)
            fe_centrality = fe_centrality / np.sum(fe_centrality)
        else:
            if fe_cent_op:
                fe_centrality = np.sum(feature_matrix.dot(feature_matrix.transpose()),axis=1)
                fe_centrality = fe_centrality / fe_centrality.sum()
            elif fe_cent_mn:
                fe_centrality = np.min(feature_matrix,axis=1)
                fe_centrality = fe_centrality / fe_centrality.sum()
            elif fe_cent_sm:
                fe_centrality = np.sum(feature_matrix,axis=1)
                fe_centrality = fe_centrality / fe_centrality.sum()
            else:
                print('failed to compute centrality')
        print('computing centrality took ', time.time()-t0, ' seconds')

    return feature_matrix, fe_centrality

def save_checkpoint(state, is_best, checkpoint='checkpoint'):
    filepath = os.path.join(checkpoint, folder+'checkpoint.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, folder+'model_best.pth.tar'))

def adjust_cosine_learning_rate(optimizer, epoch, schedule_idx):
    global state
    step = epoch - args.schedule[schedule_idx]
    total_steps = args.schedule[schedule_idx+1] - args.schedule[schedule_idx]
    state['lr'] = args.lr_min + (args.lr_max - args.lr_min) * 0.5 * (1 + np.cos((float(step) / float(total_steps)) * np.pi))
    args.lr = state['lr']
    for opt in optimizer:
        for param_group in opt.param_groups:
            param_group['lr'] = state['lr']

def adjust_exp_learning_rate(optimizer, epoch, schedule_idx):
    global state
    step = epoch - args.schedule[schedule_idx]
    total_steps = args.schedule[schedule_idx+1] - args.schedule[schedule_idx]
    beta = np.log(args.lr_max / args.lr_min)
    state['lr'] = args.lr_max * np.exp(-beta * float(step) / float(total_steps))
    args.lr = state['lr']
    for opt in optimizer:
        for param_group in opt.param_groups:
            param_group['lr'] = state['lr']

def adjust_cosine_learning_rate_step(optimizer, step, total_steps, mul = 1.):
    global state
    state['lr'] = args.lr_min + (args.lr_max - args.lr_min) * 0.5 * (1 + np.cos((float(step) / float(total_steps)) * np.pi))
    args.lr = mul * state['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

def adjust_constant_learning_rate_step(optimizer):
    global state
    state['lr'] = args.lr_max
    args.lr = state['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

def adjust_linear_learning_rate_step(optimizer, step, total_steps):
    global state
    state['lr'] = args.lr_min + (args.lr_max - args.lr_min) * (float(step) / float(total_steps))
    args.lr = state['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

def adapt_learning_rate(optimizer, ratio):
    global state
    adapt_lr = state['lr'] * ratio
    for param_group in optimizer.param_groups:
        param_group['lr'] = adapt_lr

def restore_learning_rate(optimizer):
    global state
    for opt in optimizer:
        for param_group in opt.param_groups:
            param_group['lr'] = state['lr']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('correct shape:', correct.shape)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        return res[0]
    else:
        return (res[0], res[1], correct[0], pred[0])

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

# from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

def get_output_fn(output_fn='softmax'):
    output_fn = output_fn.lower()
    if output_fn == 'softmax':
        return nn.Softmax(dim=1)
    elif output_fn == 'log_softmax' or output_fn == 'logsoftmax':
        return nn.LogSoftmax(dim=1)
    elif output_fn == 'sigmoid':
        return nn.Sigmoid(dim=1)
    elif output_fn == 'log_sigmoid' or output_fn == 'logsigmoid':
        return nn.LogSigmoid(dim=1)
    else:
        assert False, 'Error: unknown output_fn specified'

def cutout(mask_size, p, cutout_inside, mask_color=0):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1. - 1. / (global_step + 1.), alpha)

    for param, ema_param in zip(list(model.state_dict().values()), list(ema_model.state_dict().values())):
        param_dtype = param.type()
        if not param_dtype.startswith(('torch.Long', 'torch.Int', 'torch.cuda.Long', 'torch.cuda.Int')):
            ema_param.mul_(alpha).add_(param * (1. - alpha))
        else:
            ema_param = ema_param.float() * alpha + param.float() * (1. - alpha)
            ema_param = ema_param.type(param_dtype)

def init_as_ema(model, ema_model, noise=0.0):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        param.data = ema_param.data.clone().to(param.data.get_device())
        if noise > 0:
            param.data.mul_(1. + noise * torch.randn_like(param.data).to(param.data.get_device()))

class TransformTwice:
    def __init__(self, transform, use_mean_teacher):
        self.transform = transform
        self.use_mean_teacher = use_mean_teacher

    def __call__(self, inp):
        out1 = self.transform(inp)
        if self.use_mean_teacher:
            out2 = self.transform(inp)
        else:
            out2 = None
        return out1, out2

# ------------------------------------------------------------------------

if __name__ == '__main__':
    main()
