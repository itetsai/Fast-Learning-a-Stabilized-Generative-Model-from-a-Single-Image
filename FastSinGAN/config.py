import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_save_path', type=str,default='TrainedModels')
    parser.add_argument('--Dis_nfc', type=int,default=128)
    
    parser.add_argument('--inv_reduction_ratio', type=int,default=8)
    parser.add_argument('--inv_group_channels', type=int,default=1)
    parser.add_argument('--inv_groupnorm', type=int,default=4)

    parser.add_argument('--is_crop_rec', type=int,default=1) #是否使用crop_image_reconstruction loss    #reconstruction crop image
    parser.add_argument('--crop_loss_alpha', type=int,default=5)
    parser.add_argument('--random_crop_n', type=int,default=1)  #一次要取多少個 reconstruct crop
    parser.add_argument('--simple_decoder_stage', type=int,default=2)
    parser.add_argument('--simple_groupnorm', type=int,default=1)

    parser.add_argument('--is_all_rec', type=int,default=1) #是否使用all_image_reconstruction loss  #reconstruction all image

    parser.add_argument('--is_rec_loss', type=int,default=1)

    parser.add_argument('--is_z_noise_real', type=int,default=1)  #第一層z noise 是否使用原圖

    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    #stage hyper parameters:
    parser.add_argument('--nfc', type=int,help='number of filters per conv layer', default=64)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers per stage',default=3)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)
        
    #pyramid parameters:
    parser.add_argument('--nc_im',type=int,help='image # channels',default=3)
    parser.add_argument('--noise_amp',type=float,help='additive noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=300)
    parser.add_argument('--train_depth', type=int, help='how many layers are trained if growing', default=3)
    parser.add_argument('--start_scale', type=int, help='at which stage to start training', default=0)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penalty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
    parser.add_argument('--activation', default='lrelu', help="activation function {lrelu, prelu, elu, selu}")
    parser.add_argument('--lrelu_alpha', type=float, help='alpha for leaky relu', default=0.05)
    parser.add_argument('--batch_norm', action='store_true', help='use batch norm in generator', default=0)

    return parser
