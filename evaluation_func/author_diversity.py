import numpy as np
import cv2
import pathlib
import os

def calculate_diversity(original, generated):
    """
    Calculate image diversity between a real reference image and a list of other images.
    :param original: real (normalized) image, shape HxWxC
    :param generated: list of generated (normalized) images, [HxWxC, HxWxC, ...]
    :return: calculated diversity
    """
    # calculate statistics of original image
    original_pixels_std = np.mean(np.std(np.ndarray.flatten(np.asarray(original))))

    # calculate statistics of generated images
    generated = np.stack(generated, axis=2)
    generated = np.reshape(generated, [generated.shape[0], generated.shape[1], -1])
    generated_pixels_std = np.mean(np.std(generated, axis=2))

    # calculate diversity
    diversity = generated_pixels_std / original_pixels_std
    return diversity

if __name__ == '__main__':

    real_path = 'ConSinGAN_Real'
    fake_path = 'RE_ConSinGAN/self_supervise_10'

    diversity_list=[]
    for i in range(50):
        #----------------------ConSinGAN-----------------------
        realImg_path=real_path+'/'+str(i)+'.jpg'
        print( '\n' + 'real:'+realImg_path)
        real_img=cv2.imread(realImg_path)

        sample_path=fake_path+'/'+str(i)
        #print(sample_path)
        dir_name = os.listdir(sample_path)
        dir_name = dir_name[0]
        sample_path=sample_path+'/'+dir_name+'/Evaluation/random_samples'
        #------------------------------------------------------
        

        _sample_path = pathlib.Path(sample_path)

        sample_files_name = list(_sample_path.glob('*.%s' %'jpg'))

        sample_img_list=[]
        for j in range(len(sample_files_name)):
            sample_img_list.append(cv2.imread(sample_path+'/'+sample_files_name[j].name))
        print( '\n' + sample_path)

        diversity=calculate_diversity(real_img,sample_img_list)
        print( '\n' + str(diversity))

        diversity_list.append(diversity)

    print( '\n' + '--------------all----------------')
    print( '\n' + str(np.array(diversity_list).mean()))