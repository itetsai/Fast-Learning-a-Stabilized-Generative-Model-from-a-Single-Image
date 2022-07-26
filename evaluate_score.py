import os
import pathlib

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', type=str,default='Images/Places/31.jpg')
    parser.add_argument('--model_path', type=str,default='TrainedModels/31/2022_07_20_16_47_17_generation_train_depth_3_lr_scale_0.1_act_lrelu_0.05')
    parser.add_argument('-c', '--gpu', default='0', type=str, help='GPU to use (leave blank for CPU only)')
        
    parser.add_argument('--version_name',type=str,default='test')
    
    args = parser.parse_args()
    
    from evaluation_func.sifid_score import calculate_sifid_one_dir_image,calculate_sifid_one2one
    from evaluation_func.author_diversity import calculate_diversity

    #sifid
    sample_path=args.model_path+'/Evaluation/random_samples'
    sifid_mean=calculate_sifid_one_dir_image(
        args.img_path,
        sample_path,
        1,
        False,  #args.gpu!='',
        64,
        'jpg'
        )

    f = open(args.model_path+'/SIFID','w')
    f.write('-----------sifid_mean-------------\n')
    f.write(str(sifid_mean))
    f.close()

    print('-----------sifid_mean-------------')
    print(sifid_mean)

    #diversity
    real_img=cv2.imread(args.img_path)
    _sample_path = pathlib.Path(sample_path)
    sample_files_name = list(_sample_path.glob('*.%s' %'jpg'))

    sample_img_list=[]
    for j in range(len(sample_files_name)):
        sample_img_list.append(cv2.imread(sample_path+'/'+sample_files_name[j].name))

    diversity=calculate_diversity(real_img,sample_img_list)

    f = open(args.model_path+'/diversity','w')
    f.write('-----------diversity_mean-------------\n')
    f.write(str(diversity))
    f.close()

    print('-----------diversity-------------')
    print(diversity)
#