import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv
import os

import cv2

def Loss2Picture(file_dir):
    filepath=file_dir+'/loss.csv'
    loss_table=[]
    with open(filepath, newline='') as csvfile:
        loss_table=list(csv.reader(csvfile))
        csvfile.close()

    item=loss_table[0]
    #print(item)         #['iter', 'errD_real', 'errD_fake', 'gradient_penalty', 'errD_total', 'errG', 'rec_loss', 'errG_total', 'crop_loss', all_rec_loss]
    del loss_table[0]

    loss_table=np.array(loss_table,dtype=np.float)
    name_list=[]
    for i in range(1,len(item)):    #第一項是 index
        value_list=loss_table[:,i]

        value_list=value_list.transpose()
        df = pd.DataFrame(value_list, columns = [item[i]])
        fig = df.plot(figsize = (40,6))  #建立圖表物件，並複製給fig

        plt.legend(loc = 'upper right')
        plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
        plt.title(item[i],fontsize=50)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=30)

        plt.xlabel('epoch',fontsize=15)
        plt.ylabel('loss',fontsize=20)

        plt.savefig(file_dir+'/'+item[i]+'.jpg')
        plt.close()

    errD_total_img=cv2.imread(file_dir+'/errD_total.jpg')
    errD_real_img=cv2.imread(file_dir+'/errD_real.jpg')
    errD_fake_img=cv2.imread(file_dir+'/errD_fake.jpg')
    gradient_penalty_img=cv2.imread(file_dir+'/gradient_penalty.jpg')
    crop_loss_img=cv2.imread(file_dir+'/crop_loss.jpg')
    all_rec_loss_img=cv2.imread(file_dir+'/all_rec_loss.jpg')
    dis_img = cv2.vconcat([errD_total_img, errD_real_img, errD_fake_img, gradient_penalty_img,crop_loss_img,all_rec_loss_img])
    cv2.imwrite(file_dir+"/Discriminator_Loss.jpg", dis_img)

    errG_total_img=cv2.imread(file_dir+'/errG_total.jpg')
    errG_img=cv2.imread(file_dir+'/errG.jpg')
    rec_loss_img=cv2.imread(file_dir+'/rec_loss.jpg')
    gen_img = cv2.vconcat([errG_total_img, errG_img, rec_loss_img])
    cv2.imwrite(file_dir+"/Generator_Loss.jpg", gen_img)

def DrawModelLoss(model_path):
    dir_path=model_path
    for i in range(50):
        sample_path=dir_path+'/'+str(i)
        dir_name = os.listdir(sample_path)
        dir_name = dir_name[0]

        stages=0
        stage_dir=os.listdir(sample_path+'/'+dir_name)
        while True:
            if str(stages) in stage_dir:
                stages+=1
            else:
                break
        print('stages:'+str(stages))
        for j in range(stages):
            loss_csv_path=sample_path+'/'+dir_name+'/'+str(j)
            print(loss_csv_path)
            Loss2Picture(loss_csv_path)

if __name__ == '__main__':
    path='ConSinGAN_IV_sukiya/involution_pad_nobatchnormal_conv0-3_invol4-5'
    DrawModelLoss(path)


