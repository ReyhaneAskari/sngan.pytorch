# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models
# from functions import validate
from utils.utils import set_log_dir, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.fid_score_pytorch import calculate_fid
import torch.nn as nn

import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from itertools import chain
import datasets

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    # writer = writer_dict['writer']
    # global_steps = writer_dict['valid_global_steps']

    # eval mode

    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    gen_net = gen_net.eval()

    sample_list = []
    eval_iter = args.num_eval_imgs // args.eval_batch_size
    for i in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        samples = gen_net(z)
        sample_list.append(samples.data.cpu().numpy())

    new_sample_list = list(chain.from_iterable(sample_list))
    fake_image_np = np.concatenate([img[None] for img in new_sample_list], 0)

    real_image_np = []
    for i, (images, _) in enumerate(train_loader):
        real_image_np += [images.data.numpy()]
        batch_size = real_image_np[0].shape[0]
        if len(real_image_np) * batch_size >= fake_image_np.shape[0]:
            break
    real_image_np = np.concatenate(real_image_np, 0)[:fake_image_np.shape[0]]
    fid_score = calculate_fid(real_image_np, fake_image_np, batch_size=300)

    # eval_iter = args.num_eval_imgs // args.eval_batch_size
    # img_list = list()
    # for iter_idx in tqdm(range(eval_iter), desc='sample images'):
    #     z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

    #     # Generate a batch of images
    #     gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    #     for img_idx, img in enumerate(gen_imgs):
    #         file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
    #         imsave(file_name, img)
    #     img_list.extend(list(gen_imgs))

    # # get inception score
    # logger.info('=> calculate inception score')
    # mean, std = get_inception_score(img_list)

    # # get fid score
    # logger.info('=> calculate fid score')
    # fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    # os.system('rm -r {}'.format(fid_buffer_dir))

    # writer.add_image('sampled_images', img_grid, global_steps)
    # writer.add_scalar('Inception_score/mean', mean, global_steps)
    # writer.add_scalar('Inception_score/std', std, global_steps)
    # writer.add_scalar('FID_score', fid_score, global_steps)

    # writer_dict['valid_global_steps'] = global_steps + 1
    mean = 0

    return mean, fid_score


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models.' + args.model + '.Generator')(args=args).cuda()

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))

    # set writer
    logger.info(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f'=> loaded checkpoint {checkpoint_file}')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }
    inception_score, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
    logger.info(f'Inception score: {inception_score}, FID score: {fid_score}.')


if __name__ == '__main__':
    main()
