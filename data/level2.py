#!/usr/bin/env python2.7
# coding: utf-8


import os
from os.path import join, exists
import time
from collections import defaultdict
import cv2
import numpy as np
import h5py
from common import logger, createDir, getDataFromTxt, getPatch, processImage
from common import shuffle_in_unison_scary
from utils import randomShift, randomShiftWithArgument

# 第二阶段 -- 对每一个关键点做两种padding的小框，对框进行微调
# 左眼、右眼、鼻子、左嘴角和右嘴角分别取0.16和0.18的padding
types = [(0, 'LE1', 0.16),
         (0, 'LE2', 0.18),
         (1, 'RE1', 0.16),
         (1, 'RE2', 0.18),
         (2, 'N1', 0.16),
         (2, 'N2', 0.18),
         (3, 'LM1', 0.16),
         (3, 'LM2', 0.18),
         (4, 'RM1', 0.16),
         (4, 'RM2', 0.18),]
for t in types:
    d = '/home/zt/face_key_point_detection/deep_landmark/mydataset/mytrain/2_%s' % t[1]
    createDir(d)

def generate(ftxt, mode, argument=False):
    """
        Generate Training Data for LEVEL-2
        mode = train or test
    """
    data = getDataFromTxt(ftxt)
    
    
    trainData = defaultdict(lambda: dict(patches=[], landmarks=[]))
    for (imgPath, bbox, landmarkGt) in data:
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)

        # 对人脸框进行随机微调，减小过拟合程度
        landmarkPs = randomShiftWithArgument(landmarkGt, 0.05)
        if not argument:
            landmarkPs = [landmarkPs[0]]

        # 得到关键点生成的padding的边界框
        for landmarkP in landmarkPs:
            for idx, name, padding in types:
                # patch -- 关键点扩展之后的图片；patch_BBox -- 关键点扩展之后的框框
                patch, patch_bbox = getPatch(img, bbox, landmarkP[idx], padding)
                # 将大小不一的图像resize到固定的相同大小15*15
                patch = cv2.resize(patch, (15, 15))
                # 转换为caffe的输入格式，caffe输入格式为（batch，channel，h，w）
                patch = patch.reshape((1, 15, 15))
                # 将图片放入patches中
                trainData[name]['patches'].append(patch)
                # 将微调之后的图片关键点位置进行相应的变换，将绝对位置转化为相对位置
                _ = patch_bbox.project(bbox.reproject(landmarkGt[idx]))
                trainData[name]['landmarks'].append(_)

    for idx, name, padding in types:
        logger('writing training data of %s'%name)
        patches = np.asarray(trainData[name]['patches'])
        landmarks = np.asarray(trainData[name]['landmarks'])
        # 对取出的关键点图片首先进行预处理
        patches = processImage(patches)
        # 再进行shuffle操作
        shuffle_in_unison_scary(patches, landmarks)

        # 我们有10种patch，每种patch都一一对应其label，生成hdf5文件
        with h5py.File('/home/zt/face_key_point_detection/deep_landmark/mydataset/mytrain/2_%s/%s.h5'%(name, mode), 'w') as h5:
            h5['data'] = patches.astype(np.float32)
            h5['landmark'] = landmarks.astype(np.float32)
        with open('/home/zt/face_key_point_detection/deep_landmark/mydataset/mytrain/2_%s/%s.txt'%(name, mode), 'w') as fd:
            fd.write('/home/zt/face_key_point_detection/deep_landmark/mydataset/mytrain/2_%s/%s.h5'%(name, mode))


if __name__ == '__main__':
    np.random.seed(int(time.time()))
    # trainImageList.txt
    generate('/home/zt/face_key_point_detection/deep_landmark/cnn-face-data/trainImageList.txt', 'train', argument=True)
    # testImageList.txt
    generate('/home/zt/face_key_point_detection/deep_landmark/cnn-face-data/testImageList.txt', 'test')
    # Done
