#!/usr/bin/env python

from __future__ import print_function
import os
import os.path as osp

import numpy as np
import cv2

from SSH.test import detect
from utils.get_config import cfg_from_file, cfg, cfg_print
from utils.timer import Timer

import caffe

VISUALIZE_RLT = False
# CLASSES = ('__background__', 'face')

# prototxt = 'models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
prototxt = '/disk2/zhaoyafei/SSH-FD//SSH/models/test_ssh.prototxt'
caffemodel = '/disk2/zhaoyafei/SSH-FD/data/SSH_models/SSH.caffemodel'
# caffemodel = '/disk2/zhaoyafei/SSH-FD/output/ssh/wider_train/SSH_iter_21000.caffemodel'

config_file = './SSH/configs/default_config.yml'
# config_file = './SSH/configs/wider.yml'
# config_file = './SSH/configs/wider_pyramid.yml'

img_root_dir = '/disk2/data/FACE/widerface/WIDER_val/images'
img_list_file = '/disk2/data/FACE/widerface/list_img_widerface_val.txt'
save_dir = 'fd_rlt_widerface_val_default_cfg'

# img_root_dir = '/disk2/data/FACE/widerface/WIDER_test/images'
# img_list_file = '/disk2/data/FACE/widerface/list_img_widerface_test.txt'
# save_dir = 'fd_rlt_widerface_test'

gpu_id = 0

# thresh = 0.5
# NMS_THRESH = 0.3

# if thresh < 0.3:
#     thresh = 0.3

# if NMS_THRESH > thresh:
#     NMS_THRESH = thresh


def main():
    print('config file: ', config_file)
    print('load prototxt: ', prototxt)
    print('load caffemodel: ', caffemodel)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, 'rlt.txt'), 'w')

    if not osp.isfile(prototxt):
        raise IOError('Can not find file: {}'.format(prototxt))
    if not osp.isfile(caffemodel):
        raise IOError('Can not find file: {}'.format(caffemodel))
    if not osp.isfile(config_file):
        raise IOError('Can not find file: {}'.format(config_file))

    # cfg.GPU_ID = gpu_id
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # Load the external config
    cfg_from_file(config_file)
    # Print config file
    print("---> config options: \n")
    cfg_print(cfg)

    # Loading the network
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = 'SSH'

    msg = '\n\nLoaded network {:s}\n'.format(caffemodel)
    print(msg)
    fp_rlt.write(msg)

    pyramid_flag = True if len(cfg.TEST.SCALES) > 1 else False

    timer = Timer()

    fp = open(img_list_file, "r")
    for line in fp:
        line = line.strip()
        if not line:
            continue

        line_splits = line.split('/')

        sub_dir = osp.join(save_dir, line_splits[0])
        if not osp.exists(sub_dir):
            os.mkdir(sub_dir)

        base_fn = osp.splitext(line_splits[1])[0]

        fn_det = osp.join(sub_dir, base_fn + '.txt')
        fp_det = open(fn_det, 'w')
        fp_det.write(line + '\n')

        im_file = osp.join(img_root_dir, line)

        msg = '===> ' + im_file
        print(msg)
        fp_rlt.write(msg + '\n')

        # im = cv2.imread(im_file)

        # msg = 'size (W x H): {:d} x {:d} '.format(im.shape[1], im.shape[0])
        # print(msg)
        # fp_rlt.write(msg + '\n')

        # Detect all object classes and regress object bounds
        timer.tic()

        # Perform detection
        dets, _ = detect(
            net, im_file, visualization_folder=save_dir,
            visualize=VISUALIZE_RLT, pyramid=pyramid_flag)

        timer.toc()

        msg = ('Detection took {:.3f}s\n').format(
            timer.diff)

        print(msg)
        fp_rlt.write(msg)

        n_dets = len(dets)
        fp_det.write('{:d}\n'.format(n_dets))

        for i in range(n_dets):
            msg = ('{:f} {:f} {:f} {:f} {:f}\n').format(
                dets[i][0], dets[i][1],
                dets[i][2] - dets[i][0], dets[i][3] - dets[i][1],
                dets[i][4]
            )
            fp_det.write(msg)

        fp_det.close()

        # msg = ('cls = {:s}: {:d} detections after NMS').format(
        #     cls_name, dets.shape[0])
        # print(msg)

        msg = ('===> Processed {:d} images took {:.3f}s, '
               'Avg time: {:.3f}s\n').format(timer.calls, timer.total_time,
                                             timer.total_time / timer.calls)

        print(msg)
        fp_rlt.write(msg)

        fp_rlt.write(msg)
        fp_rlt.flush()

    fp_rlt.close()


if __name__ == '__main__':
    main()
