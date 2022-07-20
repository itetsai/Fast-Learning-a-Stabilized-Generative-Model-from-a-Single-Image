import os
import cv2
from numpy.lib.function_base import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import FastSinGAN.functions as functions
import FastSinGAN.models as models
import math

import csv

def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    real = functions.read_image(opt)
    real = functions.adjust_scales2image(real, opt)
    reals = functions.create_reals_pyramid(real, opt)
    print("Training on image pyramid: {}".format([r.shape for r in reals]))
    print("")

    generator = init_G(opt) #所有 stage 是共用一個 Generator
    fixed_noise = []
    fixed_noise_flip = []
    noise_amp = []
        
    for scale_num in range(opt.stop_scale+1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                print(OSError)
                pass
        functions.save_image('{}/real_scale.jpg'.format(opt.outf), reals[scale_num])

        d_curr = init_D(opt)
        if scale_num > 0:
            d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))   # 每一階 discriminator 的參數都是繼承舊的
            generator.init_next_stage()

        writer = SummaryWriter(log_dir=opt.outf)
        fixed_noise, fixed_noise_flip, noise_amp, generator, d_curr = train_single_scale(d_curr, generator, reals, fixed_noise, fixed_noise_flip, noise_amp, opt, scale_num, writer)

        torch.save(fixed_noise, '%s/fixed_noise.pth' % (opt.out_))
        torch.save(generator, '%s/G.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(noise_amp, '%s/noise_amp.pth' % (opt.out_))
        del d_curr
    writer.close()
    
    return

def train_single_scale(netD, netG, reals, fixed_noise, fixed_noise_flip, noise_amp, opt, depth, writer):
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    real_filp = torch.flip(real,[3])

    all_depth = len(reals)

    alpha = opt.alpha
    if opt.is_rec_loss==0:
        alpha = 0

    crop_alpha = opt.crop_loss_alpha
    ############################
    # define z_opt for training on reconstruction
    ###########################
    if depth == 0:
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            if opt.is_z_noise_real==1:
                z_opt = reals[0]
            else:
                print('all noise')
                z_opt = functions.generate_noise([3,
                                                reals_shapes[depth][2],
                                                reals_shapes[depth][3]],
                                                device=opt.device)
            z_opt_flip = real_filp
        elif opt.train_mode == "animation":
            z_opt = functions.generate_noise([opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                             device=opt.device).detach()
            z_opt_flip = functions.generate_noise([opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                             device=opt.device).detach()
    else:
        if opt.train_mode == "generation" or opt.train_mode == "animation":
            if netG.is_shrink==True:
                z_opt = functions.generate_noise([opt.nfc,
                                                reals_shapes[depth][2]+opt.num_layer*2+netG.noise_add*2,
                                                reals_shapes[depth][3]+opt.num_layer*2+netG.noise_add*2],
                                                device=opt.device)
                z_opt_flip = functions.generate_noise([opt.nfc,
                                                reals_shapes[depth][2]+opt.num_layer*2+netG.noise_add*2,
                                                reals_shapes[depth][3]+opt.num_layer*2+netG.noise_add*2],
                                                device=opt.device)
            else:
                z_opt = functions.generate_noise([opt.nfc,
                                reals_shapes[depth][2],
                                reals_shapes[depth][3]],
                                device=opt.device)
                z_opt_flip = functions.generate_noise([opt.nfc,
                                reals_shapes[depth][2],
                                reals_shapes[depth][3]],
                                device=opt.device)
        else:
            z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                              device=opt.device).detach()
            z_opt_flip = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                              device=opt.device).detach()
    fixed_noise.append(z_opt.detach())
    fixed_noise_flip.append(z_opt_flip.detach())

    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################
    # setup optimizers for D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG.body[-opt.train_depth:])]   #設定每一層 body 有不一樣的 learning rate

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)
    
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    if depth == 0:
        noise_amp.append(1)
    else:
        noise_amp.append(0)
        z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)

        criterion = nn.MSELoss()
        rec_loss = criterion(z_reconstruction, real)

        RMSE = torch.sqrt(rec_loss).detach()
        _noise_amp = opt.noise_amp_init * RMSE
        noise_amp[-1] = _noise_amp

    # start training
    _iter = tqdm(range(opt.niter))

    _errD_real=0
    _errD_fake=0
    _gradient_penalty=0
    _crop_loss=0
    _all_rec_loss=0
    _errD_total=0
    
    _errG=0
    _rec_loss=0
    _errG_total=0
    
    loss_list=[]
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale)+
            ' Dreal:'+str(_errD_real)+' Dfake:'+str(_errD_fake)+' gradient:'+str(_gradient_penalty)+
            ' crop_rec:'+str(_crop_loss)+' all_rec:'+str(_all_rec_loss)+' D_total:'+str(_errD_total)+
            ' G:'+str(_errG)+' rec:'+str(_rec_loss)+'  G_total:'+str(_errG_total))
        
        ############################
        # (0) sample noise for unconditional generation
        ###########################
        noise = functions.sample_random_noise(depth, reals_shapes, opt, netG)

        real_in=real

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()
            real_crop_image_list=[]
            crop_parameter_list=[]
            crop_times = opt.random_crop_n

            for i in range(crop_times):
                images_real_crop,crop_x,crop_y,crop_resolution = models.random_crop_images(real_in)
                real_crop_image_list.append(images_real_crop)
                crop_parameter_list.append([crop_x,crop_y,crop_resolution])

            output,crop_reconstruct_image_list,all_reconstruct_image = netD(real_in,crop_parameter_list)

            errD_real = -output.mean()

            crop_loss = nn.MSELoss()
            all_loss = nn.MSELoss()
                
 
            crop_rec_loss = 0
            if opt.is_crop_rec==1:
                crop_loss = nn.MSELoss()
                for i in range(len(crop_reconstruct_image_list)):
                    crop_rec_loss += crop_alpha * crop_loss(crop_reconstruct_image_list[i],real_crop_image_list[i])
                crop_rec_loss/=crop_times

            all_rec_loss = 0
            if opt.is_all_rec==1:
                all_loss = nn.MSELoss()
                all_rec_loss = crop_alpha * all_loss(all_reconstruct_image,real_in)


            # train with fake
            if j == opt.Dsteps - 1:
                fake = netG(noise, reals_shapes, noise_amp)
            else:
                with torch.no_grad():
                    fake = netG(noise, reals_shapes, noise_amp)
    
            output = netD(fake.detach())

            errD_fake = output.mean()

            gradient_penalty = functions.calc_gradient_penalty(netD, real_in, fake, opt.lambda_grad, opt.device)
            record_gradient_penalty = gradient_penalty.item()

            errD_total = errD_real + errD_fake + gradient_penalty + crop_rec_loss + all_rec_loss

            errD_total.backward()
            optimizerD.step()
            

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        #for param in netD.parameters(): #不知道這樣會不會比較省記憶體
        #    param.requires_grad = False
        
        output = netD(fake)
        errG = -output.mean()   #discriminator
        
        if alpha != 0:
            loss = nn.MSELoss()
            rec = netG(fixed_noise, reals_shapes, noise_amp)
            rec_loss = alpha * loss(rec, real)
        else:
            rec_loss = 0

        netG.zero_grad()
        errG_total = errG + rec_loss

        errG_total.backward()

        for _ in range(opt.Gsteps):     #目前觀察這是為了加速訓練   #越大越容易發生梯度爆炸
            optimizerG.step()
        
        ############################
        # (3) Log Results
        ###########################
        if iter % 250 == 0 or iter+1 == opt.niter:
            writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter+1)
            writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter+1)
            writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), record_gradient_penalty, iter+1)
            writer.add_scalar('Loss/train/G/gen', errG.item(), iter+1)
            writer.add_scalar('Loss/train/G/reconstruction', float(rec_loss), iter+1)

        if iter % 25 == 0 or iter+1 == opt.niter:
            if opt.is_crop_rec==1:
                functions.save_image('{}/real_crop_{}.jpg'.format(opt.outf, iter+1), images_real_crop.detach())
                functions.save_image('{}/reconstruction_crop_{}.jpg'.format(opt.outf, iter+1), crop_reconstruct_image_list[0].detach())
                real_crop_img=cv2.imread('{}/real_crop_{}.jpg'.format(opt.outf, iter+1))
                rec_crop_img=cv2.imread('{}/reconstruction_crop_{}.jpg'.format(opt.outf, iter+1))
                real_rec_crop_img = np.hstack([real_crop_img,rec_crop_img])
                cv2.imwrite('{}/crop_real_rec{}.jpg'.format(opt.outf, iter+1),real_rec_crop_img)

        if iter % 500 == 0 or iter+1 == opt.niter:
            functions.save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter+1), fake.detach())
            functions.save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter+1), rec.detach())
            generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)

        #----------------tqdm print-----------------------
        _errD_real=round(float(errD_real),3)
        _errD_fake=round(float(errD_fake),3)
        _gradient_penalty=round(float(gradient_penalty),3)
        _crop_loss=round(float(crop_rec_loss),3)
        _all_rec_loss=round(float(all_rec_loss),3)
        _errD_total=round(float(errD_total),3)
        
        _errG=round(float(errG),3)
        _rec_loss=round(float(rec_loss),3)
        _errG_total=round(float(errG_total),3)

        #---------------------------------------
        
        loss_list.append([iter,_errD_real,_errD_fake,_gradient_penalty,_crop_loss,_all_rec_loss,_errD_total,_errG,_rec_loss,_errG_total])

        schedulerD.step()
        schedulerG.step()
        # break
      
    # 開啟輸出的 CSV 檔案
    with open(str(functions.generate_dir2save(opt))+'/'+str(depth)+'/loss.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        csv_writer = csv.writer(csvfile)

        # 寫入一列資料
        csv_writer.writerow(['iter','errD_real','errD_fake','gradient_penalty','crop_loss','all_rec_loss','errD_total','errG','rec_loss','errG_total'])
        
        for i in range(len(loss_list)):
            csv_writer.writerow(loss_list[i])

    functions.save_networks(netG, netD, z_opt, opt)
    return fixed_noise, fixed_noise_flip, noise_amp, netG, netD

def generate_samples(netG, opt, depth, noise_amp, writer, reals, iter, n=25):
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(depth, reals_shapes, opt,netG)
            sample = netG(noise, reals_shapes, noise_amp)
            all_images.append(sample)
            functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save, idx), sample.detach())

        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[depth].squeeze()
        grid = make_grid(all_images, nrow=min(5, n), normalize=True)
        writer.add_image('gen_images_{}'.format(depth), grid, iter)

def init_G(opt):
    # generator initialization:
    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    # print(netG)

    return netG

def init_D(opt):
    #discriminator initialization:
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)

    return netD
