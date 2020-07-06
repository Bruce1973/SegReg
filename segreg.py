# coding=utf-8
import PIL.Image as Image
from os import listdir
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os
import random
import time
import tensorflow as tf
import shutil
import math
import nibabel as nb
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train0 = ['ZZM', 'FXX', 'HZL', 'XHJ', 'ZWJ']
test0 = ['CAL', 'JNH', 'LML', 'LTM', 'SHB', 'WZC', 'WXB', 'YCP', 'ZRQ' 
        'CDH', 'LYL', 'LYY', 'QJP', 'SWF', 'WLL', 'WY', 'YWY']
test = ['SP', 'SJM', 'YCP', 'WXB', 'JDJ', 'ZZM', 'SM', 'SHB', 'LTM',
        'YWY', 'XHJ', 'SWF', 'LYL', 'WLL', 'HZL']
train = ['ZSJ', 'LML', 'WZC', 'ZSF', 'ZRQ', 'LGY', 'LZH', 'JNH', 'CAL',
         'CDH', 'QJP', 'FXX', 'ZWJ', 'LYY', 'WY']

full = ['CAL', 'LML', 'LTM', 'JNH', 'YWY', 'SHB', 'WZC',
         'WXB', 'YCP', 'ZRQ', 'CDH', 'LYL', 'LYY', 'QJP',
         'SWF', 'WLL', 'WY',  'JDJ', 'LGY', 'LZH', 'SJM',
         'SM', 'SP', 'ZSF', 'ZSJ', 'ZZM', 'FXX', 'HZL', 'XHJ', 'ZWJ']
# add = ['CAL', 'LML', 'LTM', 'JNH', 'YWY']
names = test

seg_opt = 1
rigid_opt = 1


def nii2png(file_name, save_dir, mode):
    t0 = time.time()
    print('******NIFTI to PNG******')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    nii = nb.load(file_name)
    affine = nii.affine
    # print(affine)
    resx = nii.header['pixdim'][1]
    data = nii.get_data()
    data = np.array(data)
    if mode == 'mri':
        data = np.clip(data, 0, 1000)
        png_size = 256
    else:
        data = np.clip(data, -100, 200)
        png_size = 512
    #png_size = 256  #
    min = data.min()
    max = data.max()
    data = ((data-min)/(max-min))*255
    shape = data.shape
    ref = np.round(affine / resx)
    # print(ref)
    if ref[0, 0] == 0:
        data = np.transpose(data, (1, 0, 2))
        if ref[0, 1] == 1:
            data = np.flip(data, 0)
        if ref[1, 0] == 1:
            data = np.flip(data, 1)
    if ref[1, 0] == 0:
        if ref[0, 0] == 1:
            data = np.flip(data, 0)
        if ref[1, 1] == 1:
            data = np.flip(data, 1)
    data = np.transpose(data, (1, 0, 2))
    for i in range(shape[2]):
        array = data[:, :, i]
        img = Image.fromarray(array).convert('L').resize((png_size, png_size))
        img.save(os.path.join(save_dir, str(i+1)+'.png'))
    t1 = time.time()
    print('nii to png time:', t1-t0)
    return shape[2]


def liver_seg(name, mode, low, high, batch=5):
    print('******Liver segmentation******')
    slice_num = high - low
    if mode == 'mri':
        tag = mode
    else:
        tag = 'ct'
    model_dir = '/mnt/disk3/xpl/segreg/model/'+tag
    tmp_dir = '/mnt/disk3/xpl/segreg/tmp'
    png_path = os.path.join(tmp_dir, 'slice_png')
    save_path = os.path.join(tmp_dir, 'pred_png')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    graph = tf.Graph()
    with open(model_dir+'/checkpoint', 'r') as f:
        path = f.readline().split('\"')
        model_name = path[1]
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(model_name+'.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        input = graph.get_tensor_by_name('input_image:0')
        pred = graph.get_tensor_by_name('prediction:0')

        iter = math.ceil(slice_num / batch)
        size = 512
        if mode == 'mri':
            size = 256
        #size = 256 # multi-seg
        t0 = time.time()
        for i in range(iter):
            images = []
            start = i * batch + low
            end = min(start + batch, high)
            for num in range(start, end):
                image = Image.open(os.path.join(png_path, str(num) + '.png'))
                shape = image.size
                image = image.resize([size, size])
                image = image.getdata()
                images.append(image)
            images = np.array(images).reshape([-1, size, size, 1])
            print('Input images:(%d, %d)' % (start, end - 1))
            pred_mask = sess.run(pred, feed_dict={input: images})
            pred_mask = (pred_mask * 255).astype(np.uint8)
            if len(pred_mask.shape) == 4:
                pred_mask = np.squeeze(pred_mask, 3)
            # print('Output Shape:', pred_mask.shape)
            for j in range(end - start):
                image = Image.fromarray(pred_mask[j])
                image = image.resize(shape)
                image.save(os.path.join(save_path, str(start + j) + '.png'))
                # print('pred_%d.png saved!' % (start + j))
    t1 = time.time()
    print('liver seg time:', t1-t0)


def png2nii(name, mode):
    t0 = time.time()
    print('******PNG to NIFTI******')
    image = '/mnt/disk3/xpl/segreg/source_img/'+name+'-'+mode+'.nii'
    path = '/mnt/disk3/xpl/segreg/tmp/pred_png/'
    save = '/mnt/disk3/xpl/segreg/seg/'+name+'-'+mode+'.nii'
    nii = nb.load(image)
    affine = nii.affine
    resx = nii.header['pixdim'][1]
    data = np.array(nii.get_data())
    shape = data.shape
    new_data = np.zeros(shape)
    name_list = listdir(path)
    img_num = len(name_list)
    name_list = [i[0:-4] for i in name_list]
    name_list = [int(i) for i in name_list]
    min_index = min(name_list)
    # print(min_index, min_index+img_num-1)
    for i in range(min_index, min_index+img_num):
        mask_img = Image.open(path+str(i)+'.png')
        mask_img = mask_img.resize([shape[0], shape[1]])
        mask_img = np.array(mask_img)
        new_data[:, :, i-1] = mask_img.transpose()
    ref = np.round(affine / resx)
    # print(ref)
    data = new_data
    if ref[0, 0] == 0:
        if ref[0, 1] == 1:
            data = np.flip(data, 0)
        if ref[1, 0] == 1:
            data = np.flip(data, 1)
        data = np.transpose(data, (1, 0, 2))
    if ref[1, 0] == 0:
        if ref[0, 0] == 1:
            data = np.flip(data, 0)
        if ref[1, 1] == 1:
            data = np.flip(data, 1)
    new_nii = nb.Nifti1Image(np.array(data//255).astype(np.uint8), affine)
    nb.save(new_nii, save)
    t1 = time.time()
    print('png2nii time:', t1-t0)


def mask_to_liver(name, mode):
    t0 = time.time()
    print('******Mask to liver******')
    root = "/mnt/disk3/xpl/segreg/"
    tag = name+'-'+mode+'.nii'
    source_img = root+'source_img/'+tag
    mask_img = root+'seg/'+tag
    save = root+'liver/'+tag

    nii = nb.load(source_img)
    data = nii.get_data()
    # shape = data.shape
    # print(shape)
    # min_value = data.min()
    # print(min_value)
    mask = nb.load(mask_img)
    mask_data = mask.get_data()

    new_data = np.array(data*mask_data).astype(np.int16)
    new_nii = nb.Nifti1Image(new_data, nii.affine)
    nb.save(new_nii, save)
    t1 = time.time()
    print('mask to liver time:', t1-t0)


def elastix_reg(f_img, m_img, reg_mode, name=None):
    print('******Liver registration******')
    tmp_path = '/mnt/disk3/xpl/segreg/elastix/tmp/'
    cmd = 'elastix'
    if reg_mode == 'rigid':
        out_dir = tmp_path+'rigid/liver'
        param_file = tmp_path+'binary.txt'
    else:
        out_dir = tmp_path+'bspline/liver'
        param_file = tmp_path+'my_bspline.txt'+' -t0 '+'/mnt/disk3/xpl/segreg/reg/'+name+'-rigid.txt'
        # -fMask  /mnt/disk3/xpl/segreg/liver_ground/'+name+'-probe.nii'
    command = cmd + ' -f ' + f_img + ' -m ' + m_img + ' -out ' + out_dir + ' -p ' + param_file
    print(command)
    p = os.popen(command)
    print(p.read())
    p.close()

def transformix(input, trans_file, mode):
    print('******Transformation******')
    # elastix_path = '/mnt/disk3/xpl/segreg/elastix/bin/'
    cmd = 'transformix'
    out_dir = '/mnt/disk3/xpl/segreg/elastix/tmp/'+mode
    command = cmd + ' -in '+input+' -out '+out_dir+' -tp '+trans_file
    print(command)
    p = os.popen(command)
    print(p.read())
    p.close()


def del_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    else:
        os.mkdir(dir)


def segreg(name, seg_opt, rigid_opt):
    t0 = time.time()
    root = '/mnt/disk3/xpl/segreg/'
    modes = ['probe', 'mri']
    if seg_opt:
        for mode in modes:
            print('***********%s**********' % mode)
            source_img = root+'source_img/'+name+'-'+mode+'.nii'
            slice_dir = root+'tmp/slice_png/'
            pred_dir = root+'tmp/pred_png/'
            del_dir(slice_dir)
            del_dir(pred_dir)
            slice_num = nii2png(source_img, slice_dir, mode)
            liver_seg(name, mode, 1, slice_num+1, 5)
            png2nii(name, mode)
            mask_to_liver(name, mode)
    f_tag = modes[0]
    m_tag = modes[1]
    f_rigid = root+'seg/'+name+'-'+f_tag+'.nii'
    m_rigid = root+'seg/'+name+'-'+m_tag+'.nii'
    f_bspline = root+'liver/'+name+'-'+f_tag+'.nii'
    m_bspline = root+'liver/'+name+'-'+m_tag+'.nii'
    f_source = root+'source_img/'+name+'-'+f_tag+'.nii'
    m_source = root+'source_img/'+name+'-'+m_tag+'.nii'
    reg_tmp = '/mnt/disk3/xpl/segreg/elastix/tmp/'
    save_dir = root+'reg/'+name
    # f_full = root+'source_img/'+name+'-probe.nii'
    # m_full = root+'source_img/'+name+'-mri.nii'
    # f_rigid_ref = root+'liver_ground/'+name+'-probe.nii'
    # m_rigid_ref = root+'liver_ground/'+name+'-mri.nii'
    # f_bspline_ref =  root+'liver_gray/'+name+'-probe.nii'
    # m_bspline_ref = root+'liver_gray/'+name+'-mri.nii'
    t10 = time.time()
    if rigid_opt: 
        
        elastix_reg(f_rigid, m_rigid, 'rigid')
        t1 = time.time()
        shutil.copy(reg_tmp+'rigid/liver/TransformParameters.0.txt', save_dir+'-rigid.txt')

        transformix(m_source, save_dir+'-rigid.txt', 'rigid')
        shutil.copy(reg_tmp+'rigid/result.nii', save_dir+'-rigid-full.nii')

        transformix(root+'liver/'+name+'-'+m_tag+'.nii', save_dir+'-rigid.txt', 'rigid')
        shutil.copy(reg_tmp+'rigid/result.nii', save_dir+'-rigid-liver.nii')

        transformix(m_rigid, save_dir+'-rigid.txt', 'rigid')
        shutil.copy(reg_tmp+'rigid/result.nii', save_dir+'-rigid-mask.nii')
    t20 = time.time()
    elastix_reg(f_bspline, m_bspline, 'bspline', name)
    t21 = time.time()
    shutil.copy(reg_tmp+'bspline/liver/TransformParameters.0.txt', save_dir+'-bspline.txt')
    transformix(m_source, save_dir+'-bspline.txt', 'bspline')
    shutil.copy(reg_tmp+'bspline/result.nii', save_dir+'-bspline-full.nii')
    with open(save_dir+'-bspline.txt', 'r') as r:
        lines = r.readlines()
        lines[-8]='(FinalBSplineInterpolationOrder 1)\n'
        with open(save_dir+'-bbspline.txt', 'w') as w:
            w.writelines(lines)
    with open(save_dir+'-rigid.txt', 'r') as r:
        lines = r.readlines()
        lines[-8]='(FinalBSplineInterpolationOrder 1)\n'
        with open(save_dir+'-brigid.txt', 'w') as w:
            w.writelines(lines)

    t2 = time.time()
    print('segreg time:', t2-t0)
    return t20-t10, t21-t20

rt = []
ft = []
t0 = time.time() 
for name in names:
    print('****************%s****************' % name)
    rigid_time, ffd_time=segreg(name, seg_opt, rigid_opt)
    rt.append(rigid_time)
    ft.append(ffd_time)
    # segreg(name)
t1 = time.time()
print('Total time:', t1-t0)
print(rt)
print(ft)
#print(np.average(rt), np.std(rt))
#print(np.average(ft), np.std(ft))
