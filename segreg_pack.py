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
import sys, getopt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seg_opt = 1
rigid_opt = 1


def nii2png(file_name, save_dir, mode):
    t0 = time.time()
    print('******NIFTI to PNG******')
    create_dir(save_dir)
    nii = nb.load(file_name)
    affine = nii.affine
    resx = nii.header['pixdim'][1]
    data = nii.get_data()
    data = np.array(data)
    if mode == 'mri':
        data = np.clip(data, 0, 1000)
        png_size = 256
    else:
        data = np.clip(data, -100, 200)
        png_size = 512
    min = data.min()
    max = data.max()
    data = ((data-min)/(max-min))*255
    shape = data.shape
    ref = np.round(affine / resx)

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


def liver_seg(root, mode, low, high, batch=5):
    print('******Liver segmentation******')
    slice_num = high - low
    if mode == 'mri':
        tag = mode
    else:
        tag = 'ct'
    model_dir = root+'model/'+tag
    tmp_dir = root+'tmp'
    png_path = os.path.join(tmp_dir, 'slice_png')
    save_path = os.path.join(tmp_dir, 'pred_png')
    create_dir(save_path)
    graph = tf.Graph()
    ckpt_name = model_dir + '/checkpoint'
    with open(ckpt_name, 'r') as f:
        model_num = f.readline().split('\"')[1]
        print(model_num)
        model_num=model_num.split('/')[-1]
        print(model_num)
    with open(ckpt_name, 'w') as f:
        f.write('model_checkpoint_path: \"' + model_dir + '/' + model_num +  '\"')
    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(model_dir + '/' + model_num + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        input = graph.get_tensor_by_name('input_image:0')
        pred = graph.get_tensor_by_name('prediction:0')

        iter = math.ceil(slice_num / batch)
        size = 512
        if mode == 'mri':
            size = 256
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
            for j in range(end - start):
                image = Image.fromarray(pred_mask[j])
                image = image.resize(shape)
                image.save(os.path.join(save_path, str(start + j) + '.png'))
    t1 = time.time()
    print('liver seg time:', t1-t0)


def png2nii(root, name, mode):
    t0 = time.time()
    print('******PNG to NIFTI******')
    image = root+'source_img/'+name+'-'+mode+'.nii'
    path = root+'tmp/pred_png/'
    create_dir(root+'seg')
    save = root+'seg/'+name+'-'+mode+'.nii'
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
    for i in range(min_index, min_index+img_num):
        mask_img = Image.open(path+str(i)+'.png')
        mask_img = mask_img.resize([shape[0], shape[1]])
        mask_img = np.array(mask_img)
        new_data[:, :, i-1] = mask_img.transpose()
    ref = np.round(affine / resx)
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


def mask_to_liver(root, name, mode):
    t0 = time.time()
    print('******Mask to liver******')

    tag = name+'-'+mode+'.nii'
    source_img = root+'source_img/'+tag
    mask_img = root+'seg/'+tag
    create_dir(root+'liver')
    save = root+'liver/'+tag

    nii = nb.load(source_img)
    data = nii.get_data()
    mask = nb.load(mask_img)
    mask_data = mask.get_data()

    new_data = np.array(data*mask_data).astype(np.int16)
    new_nii = nb.Nifti1Image(new_data, nii.affine)
    nb.save(new_nii, save)
    t1 = time.time()
    print('mask to liver time:', t1-t0)


def elastix_reg(root, f_img, m_img, reg_mode, name=None):
    print('******Liver registration******')
    tmp_path = root+'elastix/tmp/'
    cmd = root+'elastix/bin/elastix'
    if reg_mode == 'rigid':
        out_dir = tmp_path+'rigid/liver'
        param_file = tmp_path+'binary.txt'
    else:
        out_dir = tmp_path+'bspline/liver'
        param_file = tmp_path+'my_bspline.txt'+' -t0 '+root+'reg/'+name+'-rigid.txt'
    command = cmd + ' -f ' + f_img + ' -m ' + m_img + ' -out ' + out_dir + ' -p ' + param_file
    print(command)
    p = os.popen(command)
    print(p.read())
    p.close()


def transformix(root, input, trans_file, mode):
    print('******Transformation******')
    cmd = root+'elastix/bin/transformix'
    out_dir = root+'elastix/tmp/'+mode
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


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def segreg(root, name, seg_opt, rigid_opt):
    t0 = time.time()
    modes = ['probe', 'mri']
    if seg_opt:
        for mode in modes:
            print('***********%s**********' % mode)
            source_img = root+'source_img/'+name+'-'+mode+'.nii'
            create_dir(root+'tmp')
            slice_dir = root+'tmp/slice_png/'
            pred_dir = root+'tmp/pred_png/'
            del_dir(slice_dir)
            del_dir(pred_dir)
            slice_num = nii2png(source_img, slice_dir, mode)
            liver_seg(root, mode, 1, slice_num+1, 5)
            png2nii(root, name, mode)
            mask_to_liver(root, name, mode)
    f_tag = modes[0]
    m_tag = modes[1]
    f_rigid = root+'seg/'+name+'-'+f_tag+'.nii'
    m_rigid = root+'seg/'+name+'-'+m_tag+'.nii'
    f_bspline = root+'liver/'+name+'-'+f_tag+'.nii'
    m_bspline = root+'liver/'+name+'-'+m_tag+'.nii'
    f_source = root+'source_img/'+name+'-'+f_tag+'.nii'
    m_source = root+'source_img/'+name+'-'+m_tag+'.nii'
    reg_tmp = root+'elastix/tmp/'
    create_dir(root+'reg')
    save_dir = root+'reg/'+name

    if rigid_opt:
        elastix_reg(root, f_rigid, m_rigid, 'rigid')
        shutil.copy(reg_tmp+'rigid/liver/TransformParameters.0.txt', save_dir+'-rigid.txt')

        transformix(root, m_source, save_dir+'-rigid.txt', 'rigid')
        shutil.copy(reg_tmp+'rigid/result.nii', save_dir+'-rigid-full.nii')

        transformix(root, root+'liver/'+name+'-'+m_tag+'.nii', save_dir+'-rigid.txt', 'rigid')
        shutil.copy(reg_tmp+'rigid/result.nii', save_dir+'-rigid-liver.nii')

        transformix(root, m_rigid, save_dir+'-rigid.txt', 'rigid')
        shutil.copy(reg_tmp+'rigid/result.nii', save_dir+'-rigid-mask.nii')
    elastix_reg(root, f_bspline, m_bspline, 'bspline', name)

    shutil.copy(reg_tmp+'bspline/liver/TransformParameters.0.txt', save_dir+'-bspline.txt')
    transformix(root, m_source, save_dir+'-bspline.txt', 'bspline')
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

    t1 = time.time()
    print('segreg time:', t1-t0)


def help_info():
    print('-h, --help: details about the SEGREG tool')
    print('-f, --fixed: choose the filename of the fixed image(CT)')
    print('-m, --moving: choose the filename of the moving image(MRI)')
    print('-o --output: set up the results\' filename')


t0 = time.time()
root = os.getcwd()+'/'
fix_name = str()
moving_name = str()
output = str()
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hf:m:o:', ['help', 'fixed=', 'moving=', 'output='])
    for opt, file in opts:
        if opt in ('-f', '--fixed'):
            fix_name = file
        elif opt in ('-m', '--moving'):
            moving_name = file
        elif opt in ('-o', '--output'):
            output = file
        elif opt in ('-h', '--help'):
            help_info()
            sys.exit(1)
except getopt.GetoptError:
    print('ERRORS! please input the following arguments:')
    help_info()
    sys.exit(0)
if not len(fix_name) or not len(moving_name):
    print('ERRORS! please input the following arguments:')
    help_info()
    sys.exit(0)

shutil.copy(fix_name, root+'source_img/src-probe.nii')
shutil.copy(moving_name, root+'source_img/src-mri.nii')
segreg(root, 'src', seg_opt, rigid_opt)
t1 = time.time()
print('Total time:', t1-t0)