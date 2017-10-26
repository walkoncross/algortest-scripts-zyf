#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import time

import cv2

# you need the mtecnn_detector wrapper from:
# https://github.com/walkoncross/mtcnn-caffe-good
from mtcnn_detector import MtcnnDetector


caffe_model_path = '/disk2/zhaoyafei/mtcnn-caffe-good/model'

img_root_dir = '/disk2/data/FACE/fddb/originalPics'
img_list_file = '/disk2/data/FACE/fddb/FDDB-fold-01-10_2845.txt'
save_dir = 'mtcnn_fddb_rlt'

save_fn = './fd_result_fddb_format.txt'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def main():

    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, 'rlt.txt'), 'w')
    fp_fddb = open(osp.join(save_dir, save_fn), 'w')

    t1 = time.clock()
    detector = MtcnnDetector(caffe_model_path)
    t2 = time.clock()
    print("initFaceDetector() costs %f seconds" % (t2 - t1))

    fp_rlt.write('\n\nNetwork path: {:s}\n\n'.format(caffe_model_path))
    msg = 'minsize={:d}, scale_factor={:.3f}, threshold=[{}, {}, {}]\n'.format(
        minsize, scale_factor,
        threshold[0], threshold[1], threshold[2]
    )
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

        fp_fddb.write(base_fn + '\n')

        im_file = osp.join(img_root_dir, line)

        msg = '===> ' + im_file
        print(msg)
        fp_rlt.write(msg + '\n')

        img = cv2.imread(im_file)

        msg = 'size (W x H): {:d} x {:d} '.format(img.shape[1], img.shape[0])
        print(msg)
        fp_rlt.write(msg + '\n')

        # Detect all object classes and regress object bounds
        timer.tic()
        bboxes, points = detector.detect_face(img, minsize,
                                              threshold, scale_factor)
        timer.toc()

        msg = ('Detection took {:.3f}s\n').format(
            timer.diff)

        print(msg)
        fp_rlt.write(msg)

        n_dets = len(bboxes)
        fp_fddb.write('{:d}\n'.format(n_dets))

        for i in range(n_dets):
            msg = ('{:f} {:f} {:f} {:f} {:f}\n').format(
                bboxes[i][0], bboxes[i][1],
                bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1],
                bboxes[i][4]
            )
            fp_fddb.write(msg)

        fp_fddb.close()

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
