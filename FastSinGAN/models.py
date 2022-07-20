from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from torch.nn.modules import activation
from torch.nn.modules.container import Sequential
from torch.nn.modules.padding import ZeroPad2d
from FastSinGAN.imresize import imresize, imresize_to_shape

import torch.nn.functional as F

import random

class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
    def forward(self,x):
        x = x * F.sigmoid(x)
        return x

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class dilated_involution3(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 opt):
        super(dilated_involution3, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 8
        self.group_channels = 1
        groupnorm_n = opt.inv_groupnorm
        self.groups = self.channels // self.group_channels

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels // reduction_ratio,kernel_size=1,bias=False),
            nn.InstanceNorm2d(num_features=channels // reduction_ratio,affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2=nn.Conv2d(in_channels=channels // reduction_ratio,out_channels=kernel_size**2 * self.groups,kernel_size=1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        
        self.zeropad = nn.ZeroPad2d( -(kernel_size-1)//2-(dilation-1)+padding)
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, x):
        weight = self.zeropad(self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x))))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out

class SimpleDecoder(nn.Module):
    def __init__(self,in_channels,opt):
        super(SimpleDecoder, self).__init__()

        self.blockN = opt.simple_decoder_stage
        self.body = torch.nn.ModuleList([])
        
        self.conv_size=3

        for i in range(self.blockN):
            block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=self.conv_size),
                nn.GroupNorm(num_groups=opt.simple_groupnorm, num_channels=in_channels),
                nn.GLU(dim=1)
                )
            in_channels=in_channels//2
            self.body.append(block)
        self.end = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=3, kernel_size=1),
            nn.Tanh()
            )
    def forward(self, inputs,target_size): 
        x = upsample(inputs,size=[target_size[0]+self.blockN*(self.conv_size-1),target_size[1]+self.blockN*(self.conv_size-1)])   #每一層block之後都會-2 所以要 upsample blockN*2
        for i in range(len(self.body)):
            x = self.body[i](x)
        x = self.end(x)
        return x

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def crop_images(images,x,y,crop_resolution):
    return images[:, :, y:y + crop_resolution, x:x + crop_resolution]
    
def center_crop_images(images):
    """
    Crops the center of the images
    Args:
        images: shape: (B, H, W, 3), H should be equal to W
        crop_resolution: target resolution for the crop
    Returns:
        cropped images which has the shape: (B, crop_resolution, crop_resolution, 3)
    """

    #取比較小的
    crop_resolution = images.shape[2] if images.shape[2]<images.shape[3] else images.shape[3]   #tf.cast(crop_resolution, tf.float32)
    crop_resolution = crop_resolution/2
    
    half_of_crop_resolution = crop_resolution / 2

    image_height = images.shape[2]  #tf.cast(tf.shape(images)[1], tf.float32)
    image_center = image_height / 2


    from_ = int(image_center - half_of_crop_resolution)
    to_ = int(image_center + half_of_crop_resolution)

    return images[:, :, from_:to_, from_:to_]

def random_crop_images(images):
    #取比較小的
    img_width = images.shape[3]
    img_height  = images.shape[2]
    crop_resolution = img_height if img_height < img_width else img_width   #取短的
    crop_resolution = int(crop_resolution / 2)  #目前crop範圍為短邊的 1/2

    crop_x = random.randint(0,img_width-crop_resolution-1)    #-1是為了防止取到邊邊
    crop_y = random.randint(0,img_height-crop_resolution-1)

    crop_img = crop_images(images,crop_x,crop_y,crop_resolution)

    return crop_img,crop_x,crop_y,crop_resolution

def weights_init(m):    #xavier
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        if classname=='FilterNorm':
            return
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True),
                   "mish": Mish(),
                   "swish": Swish()
                   }
    return activations[opt.activation]

def upsample(x, size):
    x_up =  torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock,self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        
        self.add_module(opt.activation, get_activation(opt))

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        N = opt.Dis_nfc #int(opt.nfc)   #測試之後設到256結果會變差

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt)
        self.body1 = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
        self.body2 = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
        self.body3 = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size)       
        
        if self.opt.is_crop_rec==1:
            self.crop_simpleDecoder = SimpleDecoder(N,opt)

        if self.opt.is_all_rec==1:
            self.all_simpleDecoder = SimpleDecoder(N,opt)

    def forward(self,x,crop_image_list=None):  #if x is real image
        head = self.head(x)

        body1 = self.body1(head)
        body2 = self.body2(body1)
        body3 = self.body3(body2)
        
        out = self.tail(body3)
        
        if crop_image_list==None:   #if x is fake image
            return out
        else:   #if x is fake image
            crop_reconstruct_list = None
            if self.opt.is_crop_rec==1:
                crop_reconstruct_list=[]
                for i in range(len(crop_image_list)):   #crop_image_list = [x][crop_x,crop_y,crop_resolution]
                    crop_upsample = upsample(body2,size=[x.shape[2],x.shape[3]])
                    crop_x = crop_image_list[i][0]
                    crop_y = crop_image_list[i][1]
                    crop_resulution = crop_image_list[i][2]
                    body_crop = crop_images(crop_upsample,crop_x,crop_y,crop_resulution)    #裁切 feature map

                    crop_reconstruct = self.crop_simpleDecoder(body_crop,[body_crop.shape[2],body_crop.shape[3]])
                    crop_reconstruct_list.append(crop_reconstruct)

            all_reconstruct = None
            if self.opt.is_all_rec==1:
                all_reconstruct = self.all_simpleDecoder(body3,[x.shape[2],x.shape[3]])
                                   
            return out,crop_reconstruct_list,all_reconstruct
      
class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()

        self.stage = 0

        self.noise_add = 0 #看要再多放大多少

        self.opt = opt
        N = int(opt.nfc)

        self._pad = nn.ZeroPad2d(1)

        self._pad_block = nn.ZeroPad2d(opt.num_layer-1) if opt.train_mode == "generation"\
                                                           or opt.train_mode == "animation" \
                                                        else nn.ZeroPad2d(opt.num_layer)

        self.head = nn.Sequential()
        self.head.add_module('head_conv',ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True))
        
        self.body = torch.nn.ModuleList([])

        self.is_shrink = True   #如果使用的 body 層的 conv 或 involution 是會變小的的話 這項要設為 True

        self.body_invol_template = nn.Sequential()

        self.body_invol_template.add_module('involution0',dilated_involution3(channels=N, kernel_size=3, stride=1, dilation=3,padding=2, opt=opt))
        self.body_invol_template.add_module(opt.activation+'0', get_activation(opt))
        self.body_invol_template.add_module('conv0',ConvBlock(N, N, 1, 0, opt, generator=True))
        self.body_invol_template.add_module('involution1',dilated_involution3(channels=N, kernel_size=3, stride=1, dilation=3,padding=2, opt=opt))
        self.body_invol_template.add_module(opt.activation+'1', get_activation(opt))
        self.body_invol_template.add_module('conv1',ConvBlock(N, N, 1, 0, opt, generator=True))
        self.body_invol_template.add_module('involution2',dilated_involution3(channels=N, kernel_size=3, stride=1, dilation=3,padding=2, opt=opt))
        self.body_invol_template.add_module(opt.activation+'2', get_activation(opt))
        self.body_invol_template.add_module('conv2',ConvBlock(N, N, 1, 0, opt, generator=True))

        self.body.append(copy.deepcopy(self.body_invol_template))

        self.tail = nn.Sequential(
            nn.Conv2d(N, opt.nc_im, kernel_size=opt.ker_size, padding=opt.padd_size),
            nn.Tanh())

    def init_next_stage(self):
        self.stage+=1
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self._pad(noise[0])
        x = self.head(x)

        # we do some upsampling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
            if self.is_shrink==True:
                x = upsample(x, size=[x.shape[2] + 2 + self.noise_add*2, x.shape[3] + 2 + self.noise_add*2])  #這裡+2是為了 tail 時的 conv 會-2
            else:
                x = upsample(x, size=[x.shape[2] + 2 - 6, x.shape[3] + 2 - 6])  #-6是為了第一層的 body (因為它不在迴圈內)

        x = self._pad_block(x)

        x_prev_out = self.body[0](x)
        
        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])    #原圖 upsample
                if self.is_shrink==True:
                    x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer*2+ self.noise_add*2,   #原圖 upsample 到更大,因為過conv後會變小
                                                              real_shapes[idx][3] + self.opt.num_layer*2+ self.noise_add*2])
                else:
                    x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2],real_shapes[idx][3]])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1+noise[idx]*noise_amp[idx]))
            x_prev_out = x_prev + x_prev_out_1
        out = self.tail(self._pad(x_prev_out))
        
        return out