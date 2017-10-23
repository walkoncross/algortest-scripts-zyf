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

# img_root_dir = '/disk2/data/FACE/widerface/WIDER_val/images'
# img_list_file = '/disk2/data/FACE/widerface/list_img_widerface_val.txt'
# save_dir = 'fd_rlt_widerface_val'

img_root_dir = '/disk2/data/FACE/widerface/WIDER_test/images'
img_list_file = '/disk2/data/FACE/widerface/list_img_widerface_test.txt'
save_dir = 'fd_rlt_widerface_test'

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

        im = cv2.imread(im_file)

        msg = 'size (W x H): {:d} x {:d} '.format(im.shape[1], im.shape[0])
        print(msg)
        fp_rlt.write(msg + '\n')

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()

        msg = ('Detection took {:.3f}s for {:d} object proposals\n').format(
            timer.total_time, boxes.shape[0])

        print(msg)
        fp_rlt.write(msg)

        time_ttl += timer.total_time
        img_cnt += 1

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
               'Avg time: {:.3f}s\n').format(img_cnt, time_ttl, time_ttl / img_cnt)

        print(msg)
        fp_rlt.write(msg)

        fp_rlt.write(msg)
        fp_rlt.flush()

    fp_rlt.close()


if __name__ == '__main__':
    main()
