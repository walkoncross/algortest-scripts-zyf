#!/usr/bin/env python

import os
import os.path as osp

import numpy as np
import cv2

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe

CLASSES = ('__background__', 'face')

# prototxt = 'models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
prototxt = '/disk2/zhaoyafei/caffemodels/face_models/vgg16-face-rcnn/deploy.prototxt'
caffemodel = '/disk2/zhaoyafei/caffemodels/face_models/vgg16-face-rcnn/weight.caffemodel'

img_root_dir = '/disk2/data/FACE/fddb/originalPics'
img_list_file = '/disk2/data/FACE/fddb/FDDB-fold-01-10_2845.txt'
save_dir = 'pyfrcn_vgg16_fddb_rlt'

thresh = 0.5
NMS_THRESH = 0.3

gpu_id = 0

if thresh < 0.3:
    thresh = 0.3

if NMS_THRESH > thresh:
    NMS_THRESH = thresh


def main():
    print 'load prototxt: ', prototxt
    print 'load caffemodel: ', caffemodel

    img_cnt = 0
    time_ttl = 0.0

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, 'rlt.txt'), 'w')
    fp_fddb = open(osp.join(save_dir, save_fn), 'w')
	
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    if not osp.isfile(prototxt):
        raise IOError('Can not find file: {}'.format(prototxt))
    if not osp.isfile(caffemodel):
        raise IOError('Can not find file: {}'.format(caffemodel))

    # cfg.GPU_ID = gpu_id
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    fp_rlt.write('\n\nLoaded network {:s}\n\n'.format(caffemodel))
    msg = 'thresh={:.3f}, NMS_THRESH={:.3f}'.format(thresh, NMS_THRESH)
    print(msg)
    fp_rlt.write(msg)

    timer = Timer()

    fp = open(img_list_file, "r")
    for line in fp:
        line = line.strip()
        if not line:
            continue

        if line.startswith('200'):
            base_fn = line
        else:
            base_fn = line[line.find('200'):]

        if base_fn.endswith('.jpg'):
            base_fn = osp.splitext(base_fn)[0]
        fp_fddb.write(base_fn + '\n')

        if not line.endswith('.jpg'):
            line += '.jpg'
        im_file = osp.join(img_root_dir, line)

        msg = '===> ' + im_file
        print(msg)
        fp_rlt.write(msg + '\n')

        img = cv2.imread(im_file)

        msg = 'size (W x H): {:d} x {:d} '.format(img.shape[1], img.shape[0])
        print(msg)
        fp_rlt.write(msg + '\n')

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, img)
        timer.toc()

        msg = ('Detection took {:.3f}s for {:d} object proposals\n').format(
            timer.diff, boxes.shape[0])

        print(msg)
        fp_rlt.write(msg)

        cls_ind = 1  # because we skipped background
        # cls_name = CLASSES[cls_ind]
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        # msg = ('cls = {:s}: {:d} detections before NMS').format(
        #     cls_name, dets.shape[0])
        # print(msg)
        # fp_rlt.write(msg)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        if len(keep) > 0:
            keep = np.where(dets[:, -1] > thresh)[0].tolist()
            dets = dets[keep, :]

        n_dets = len(keep)
        fp_fddb.write('{:d}\n'.format(n_dets))

        for i in range(n_dets):
            msg = ('{:f} {:f} {:f} {:f} {:f}\n').format(
                dets[i][0], dets[i][1],
                dets[i][2] - dets[i][0], dets[i][3] - dets[i][1],
                dets[i][4]
            )
            fp_fddb.write(msg)

        fp_fddb.close()

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

        fp_fddb.flush()
    
    fp.close()
    fp_rlt.close()
    fp_fddb.close()


if __name__ == '__main__':
    main()
