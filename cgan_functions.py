# Conditional GAN training  

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from utils import make_grid, save_image
from tqdm import tqdm
# import cv2 (Not used for NILM task)

logger = logging.getLogger(__name__)

def cur_stages(iter, args):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        # if search_iter < self.grow_step1:
        #     return 0
        # elif self.grow_step1 <= search_iter < self.grow_step2:
        #     return 1
        # else:
        #     return 2
        # for idx, grow_step in enumerate(args.grow_steps):
        #     if iter < grow_step:
        #         return idx
        # return len(args.grow_steps)
        idx = 0
        for i in range(len(args.grow_steps)):
            if iter >= args.grow_steps[i]:
                idx = i+1
        return idx


def gradient_penalty(y, x, args):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda(args.gpu, non_blocking=True)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    # Calculate gradient penalty
    dydx = dydx.reshape(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)    
    
def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    cls_criterion = nn.CrossEntropyLoss()
    lambda_cls = 1
    lambda_gp = 10
    
    # train mode
    gen_net.train()
    dis_net.train()
    
    pbar = tqdm(train_loader)
    for iter_idx, (real_power, real_time, real_img_labels) in enumerate(pbar):
        global_steps = writer_dict['train_global_steps']
        
        # Move to GPU
        real_power = real_power.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        real_time = real_time.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        real_img_labels = real_img_labels.type(torch.LongTensor).cuda(args.gpu, non_blocking=True)

        # Sample noise
        noise = torch.randn(real_power.shape[0], args.latent_dim, device=real_power.device)
        fake_img_labels = torch.randint(0, args.n_classes, (real_power.shape[0],)).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        dis_net.zero_grad()
        # 🚀 Discriminator sees Power + Time pairs
        # Discriminator forward-pass
        r_out_adv, r_out_cls = dis_net(real_power, real_time)
        fake_power = gen_net(noise, fake_img_labels, real_time)
        f_out_adv, f_out_cls = dis_net(fake_power, real_time)

        # Compute classification loss (only if model supports it)
        if r_out_cls is not None:
            d_cls_loss = cls_criterion(r_out_cls, real_img_labels)
        else:
            d_cls_loss = 0

        # Gradient penalty
        alpha = torch.rand(real_power.size(0), 1, 1, 1).cuda(args.gpu, non_blocking=True)
        # alpha = alpha.expand_as(real_power)
        x_hat = (alpha * real_power.data + (1 - alpha) * fake_power.data).requires_grad_(True)
        out_src, _ = dis_net(x_hat, real_time) # Gradient penalty uses real_time
        d_loss_gp = gradient_penalty(out_src, x_hat, args)
        
        # Total D loss
        # Total D loss
        d_loss_adv = -torch.mean(r_out_adv) + torch.mean(f_out_adv) + 10.0 * d_loss_gp
        d_loss = d_loss_adv + 1.0 * d_cls_loss

        # Optimization
        dis_optimizer.zero_grad()
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------
        
        gen_net.zero_grad()

        noise = torch.randn(args.batch_size, args.latent_dim).cuda(args.gpu, non_blocking=True).view(args.batch_size, args.latent_dim, 1, 1)
        fake_img_labels = torch.randint(0, args.n_classes, (args.batch_size,)).cuda(args.gpu, non_blocking=True)
        
        gen_power = gen_net(noise, fake_img_labels, real_time)
        g_out_adv, g_out_cls = dis_net(gen_power, real_time)
        
        if g_out_cls is not None:
            g_cls_loss = cls_criterion(g_out_cls, fake_img_labels)
        else:
            g_cls_loss = 0
            
        g_loss_adv = -torch.mean(g_out_adv)
        g_loss = g_loss_adv + 1.0 * g_cls_loss
        g_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        gen_optimizer.step()

        # adjust learning rate
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(global_steps)
            d_lr = dis_scheduler.step(global_steps)
            writer.add_scalar('LR/g_lr', g_lr, global_steps)
            writer.add_scalar('LR/d_lr', d_lr, global_steps)

        # moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.dis_batch_size * args.world_size * global_steps
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
        else:
            ema_beta = args.ema

        # moving average weight (Optimized for 4090 Speed)
        with torch.no_grad():
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(ema_beta).add_(p.data, alpha=1. - ema_beta)

        if args.rank == 0:
            pbar.set_description(
                f"Epoch {epoch}/{args.max_epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | EMA: {ema_beta:.4f}"
            )

        writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
        gen_step += 1

        del gen_power
        del real_power
        del fake_power
        del f_out_adv
        del r_out_adv
        del r_out_cls
        del g_out_cls
        del g_cls_loss
        del g_adv_loss
        del g_loss
        del d_cls_loss
        del d_adv_loss
        del d_loss

        writer_dict['train_global_steps'] = global_steps + 1 

        
def get_is(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('calculate Inception score...')
    mean, std = get_inception_score(img_list)

    return mean        
        
def save_samples(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):

    # eval mode
    gen_net.eval()
    with torch.no_grad():
        # generate images
        batch_size = fixed_z.size(0)
        sample_imgs = []
        for i in range(fixed_z.size(0)):
            sample_img = gen_net(fixed_z[i:(i+1)], epoch)
            sample_imgs.append(sample_img)
        sample_imgs = torch.cat(sample_imgs, dim=0)
        os.makedirs(f"./samples/{args.exp_name}", exist_ok=True)
        save_image(sample_imgs, f'./samples/{args.exp_name}/sampled_images_{epoch}.png', nrow=10, normalize=True, scale_each=True)
    return 0


def get_topk_arch_hidden(args, controller, gen_net, prev_archs, prev_hiddens):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(f'=> get top{args.topk} archs out of {args.num_candidate} candidate archs...')
    assert args.num_candidate >= args.topk
    controller.eval()
    cur_stage = controller.cur_stage
    archs, _, _, hiddens = controller.sample(args.num_candidate, with_hidden=True, prev_archs=prev_archs,
                                             prev_hiddens=prev_hiddens)
    hxs, cxs = hiddens
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f'arch: {archs[arch_idx]}')
        gen_net.set_arch(archs[arch_idx], cur_stage)
        is_score = get_is(args, gen_net, args.rl_num_eval_img)
        logger.info(f'get Inception score of {is_score}')
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]
    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f'top{args.topk} archs:')
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs, (topk_hxs, topk_cxs)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
#             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.to("cpu")) # Fixed: Removed redundant .cuda() and ensured .to("cpu")
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten