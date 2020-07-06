# coding=utf-8
import os
import shutil
import numpy as np
import nibabel as nb
import SimpleITK as sitk
import time

def transformix(input, tp_file):
    cmd = 'transformix'
    save_dir = '/mnt/disk3/xpl/segreg/elastix/tmp/bspline'
    command = cmd+' -in '+input+' -out '+save_dir+' -tp '+tp_file
    p = os.popen(command)
    print(p.read())
    p.close()


def dice_reg(name, reg_trans):
    root = '/mnt/disk3/xpl/segreg/'
    nii1 = nb.load(root+'liver_ground/'+name+'-probe.nii')
    nii2 = nb.load(root+reg_trans+name+'-bspline-ground.nii')

    data1 = np.array(nii1.get_data())
    data2 = np.array(nii2.get_data())

    inter = data1*data2
    dice = 2*np.sum(inter)/(np.sum(data1)+np.sum(data2))
    return dice


train0 = ['JDJ', 'LGY', 'LZH', 'SJM', 'SM', 'SP', 'ZSF', 'ZSJ', 'ZZM',
         'FXX', 'HZL', 'XHJ', 'ZWJ']
test0 = ['CAL', 'JNH', 'LML', 'LTM', 'SHB', 'WZC', 'WXB', 'YCP', 'ZRQ',  
        'CDH', 'LYL', 'LYY', 'QJP', 'SWF', 'WLL', 'WY', 'YWY']
names = [ 'CAL', 'JNH', 'LML', 'LTM', 'SHB', 'WZC', 'WXB', 'YCP', 'ZRQ',  
          'CDH', 'LYL', 'LYY', 'QJP', 'SWF', 'WLL', 'WY', 'YWY', 
          'JDJ', 'LGY', 'LZH', 'SJM', 'SM', 'SP', 'ZSF', 'ZSJ', 'ZZM',
          'FXX', 'HZL', 'XHJ', 'ZWJ']
test = ['SP', 'SJM', 'YCP', 'WXB', 'JDJ', 'ZZM', 'SM', 'SHB', 'LTM',
        'YWY', 'XHJ', 'SWF', 'LYL', 'WLL', 'HZL']
train = ['ZSJ', 'LML', 'WZC', 'ZSF', 'ZRQ', 'LGY', 'LZH', 'JNH', 'CAL',
         'CDH', 'QJP', 'FXX', 'ZWJ', 'LYY', 'WY',]
test0 = ['JDJ', 'LTM', 'SJM', 'SHB', 'SM', 'SP', 'WXB', 'YCP', 'ZZM']
test918 = ['HZL', 'LYL', 'SWF', 'WLL', 'XHJ', 'YWY']
full0 = ['CAL', 'JNH', 'LML', 'LGY', 'LZH', 'WZC', 'ZSF', 'ZSJ', 'ZRQ']+test0
full918 = ['CDH', 'FXX', 'LYY', 'QJP', 'WY', 'ZWJ']+test918

root = '/mnt/disk3/xpl/segreg/'
dices = []
trans_opt = 1
options = ['rigid', 'bspline']
opt = options[1]

reg_file = '/reg/'
reg_trans = '/reg_ground/'

names = test
t0 = time.time()
for name in names:
    t_in = root+'/liver_ground/'+name+'-mri.nii'
    tp_file = root+reg_file+name+'-b'+opt+'.txt'
    if trans_opt:
        transformix(t_in, tp_file)
        shutil.copy(root+'/elastix/tmp/bspline/result.nii', root+reg_trans+name+'-'+opt+'-ground.nii')
    file1 = root+'/liver_ground/'+name+'-probe.nii'
    file2 = root+reg_trans+name+'-'+opt+'-ground.nii'
    nii1 = nb.load(file1)
    nii2 = nb.load(file2)
    data1 = np.array(nii1.get_data()).astype(np.uint8)
    data2 = np.array(nii2.get_data()).astype(np.uint8)
    im1 = sitk.GetImageFromArray(data1)
    im2 = sitk.GetImageFromArray(data2)

    #h = sitk.HausdorffDistanceImageFilter()
    #h.Execute(im1, im2)
    #h_dis = h.GetHausdorffDistance()
    #h_ave = h.GetAverageHausdorffDistance()

    d = sitk.LabelOverlapMeasuresImageFilter()
    d.Execute(im1, im2)
    dice = d.GetDiceCoefficient()
    # dice =  dice_reg(name)
    dices.append(dice)
    print(name, dice)
    # print(name, h_dis)
dices = np.array(dices)
t1 = time.time()
print(','.join([str(i) for i in dices]))
print('reg dice avg: %.4f, %.4f' % (np.average(dices), np.std(dices)), '\nTime:', (t1-t0))