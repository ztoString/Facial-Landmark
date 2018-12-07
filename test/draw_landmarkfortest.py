#coding:utf-8

import os
from os.path import join
import cv2
import sys
caffe_root = '/home/zt/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np


class CNN(object):
    """
        Generalized CNN for simple run forward with given Model
    """

    def __init__(self, net, model):
        self.net = net
        self.model = model
        # net -- deploy文件；model -- 训练好的模型；caffe.TEST -- 进行测试
        self.cnn = caffe.Net(net, model, caffe.TEST) # failed if not exists

    def forward(self, data, layer='fc2'):
        print data.shape
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(len(x)/2)])
        result = t(result)
        return result


class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = int(bbox[0])
        self.right = int(bbox[1])
        self.top = int(bbox[2])
        self.bottom = int(bbox[3])
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        print len(landmark)
        if not len(landmark) == 5:
            landmark = landmark[0]
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

    def cropImage(self, img):
        """
            crop img with left,right,top,bottom
            **Make Sure is not out of box**
        """
        return img[self.top:self.bottom+1, self.left:self.right+1]


class Landmarker(object):
    """
        class Landmarker wrapper functions for predicting facial landmarks
    """

    def __init__(self):
        """
            Initialize Landmarker with files under VERSION
        """
        #model_path = join(PROJECT_ROOT, VERSION)
        deploy_path = "/home/zt/face_key_point_detection/deep_landmark/myprototxt"
        model_path = "/home/zt/face_key_point_detection/deep_landmark/mymodel"
        # model中的子文件夹
        CNN_TYPES = ['LE1', 'RE1', 'N1', 'LM1', 'RM1', 'LE2', 'RE2', 'N2', 'LM2', 'RM2']
        # level1、level2、level3分别对应之前的三个阶段
        level1 = [(join(deploy_path, '1_F_deploy.prototxt'), join(model_path, '1_F/_iter_100000.caffemodel'))]
        level2 = [(join(deploy_path, '2_%s_deploy.prototxt'%name), join(model_path, '2_%s/_iter_100000.caffemodel'%(name))) \
                    for name in CNN_TYPES]
        level3 = [(join(deploy_path, '3_%s_deploy.prototxt'%name), join(model_path, '3_%s/_iter_100000.caffemodel'%(name))) \
                    for name in CNN_TYPES]
        # 初始化网络模型
        self.level1 = [CNN(p, m) for p, m in level1]
        # 通过for循环依次初始化10个网络模型
        self.level2 = [CNN(p, m) for p, m in level2]
        self.level3 = [CNN(p, m) for p, m in level3]

    def detectLandmark(self, image, bbox, mode='three'):
        """
            Predict landmarks for face with bbox in image
            fast mode will only apply level-1 and level-2
        """
        #if not isinstance(bbox, BBox) or image is None:
            #return None, False
            
        # 对人脸框进行裁剪
        face = bbox.cropImage(image)
        # 将裁剪后人脸框图片resize到相同大小39*39
        face = cv2.resize(face, (39, 39))
        # 转换为caffe的输入格式，caffe输入格式为（batch，channel，h，w）
        face = face.reshape((1, 1, 39, 39))
        # 图像预处理 -- (图像-mean)/std
        face = self._processImage(face)
        
        # 网络进行前向传播
        # level-1, 对全局进行预测，得到5对坐标值
        landmark = self.level1[0].forward(face)
        # level-2，对level-1得到的每个关键点生成padding
        landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
        #if mode == 'fast':
        #    return landmark,True
        # level-3，对level-2得到的每个关键点再次生成padding，再次进行微调
        landmark = self._level(image, bbox, landmark, self.level3, [0.11, 0.12])
        
        return landmark
    def _level(self, img, bbox, landmark, cnns, padding):
        """
            LEVEL-?
        """
        # 对5对关键点取patch
        for i in range(5):
            # 首先拿到第i对关键点
            x, y = landmark[i]
            
            # 上下左右各取h、w的0.16倍生成padding，获得padding图像patch和边界框patch_bbox
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[0])
            # 对patch图像做resize和reshape操作
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            # 图像预处理
            patch = self._processImage(patch)
            # 对第i个关键点坐标前向传播
            d1 = cnns[i].forward(patch) # size = 1x2
            
            # 上下左右各取h、w的0.18倍生成padding，获得padding图像patch和边界框patch_bbox
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[1])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d2 = cnns[i+5].forward(patch)

            # 之前得到的d1和d2为关键点相对于patch这个小图片的相对位置，
            # 现在我们要得到关键点相对于人脸框的位置
            d1 = bbox.project(patch_bbox.reproject(d1[0]))
            d2 = bbox.project(patch_bbox.reproject(d2[0]))
            # 对第i个关键点坐标进行更新操作，取两次padding的均值
            landmark[i] = (d1 + d2) / 2
        return landmark

    def _getPatch(self, img, bbox, point, padding):
        """
            Get a patch iamge around the given point in bbox with padding
            point: relative_point in [0, 1] in bbox
        """
        
        point_x = bbox.x + point[0] * bbox.w
        point_y = bbox.y + point[1] * bbox.h
        patch_left = point_x - bbox.w * padding
        patch_right = point_x + bbox.w * padding
        patch_top = point_y - bbox.h * padding
        patch_bottom = point_y + bbox.h * padding
        patch = img[int(patch_top): int(patch_bottom)+1, int(patch_left): int(patch_right)+1]
        patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        return patch, patch_bbox
        """
        point_x = bbox[0] + point[0] * bbox[2]
        point_y = bbox[1] + point[1] * bbox[3]
        patch_left = point_x - bbox[2] * padding
        patch_right = point_x + bbox[2] * padding
        patch_top = point_y - bbox[3] * padding
        patch_bottom = point_y + bbox[3] * padding
        patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
        #patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        patch_bbox = [patch_left,patch_top,patch_right-patch_left,patch_bottom-patch_top]
        return patch, patch_bbox
        """
        

    def _processImage(self, imgs):
        """
            process images before feeding to CNNs
            imgs: N x 1 x W x H
        """
        imgs = imgs.astype(np.float32)
        for i, img in enumerate(imgs):
            m = img.mean()
            s = img.std()
            imgs[i] = (img - m) / s
        return imgs
    
def drawLandmark(img,  landmark):
    
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 5, (0,255,0), -1)
    return img

if __name__ == '__main__':
    result_path = '/home/zt/face_key_point_detection/deep_landmark/result-folder/'
    test_folder = '/home/zt/face_key_point_detection/deep_landmark/test-folder/'
    test_images = os.listdir(test_folder)
    for image in test_images:
        
        #/home/matt/Documents/caffe/data/faceReco/baike100-c/Angelababy
        # 读入图像
        img = cv2.imread(test_folder+image)
        #img = cv2.imread('/home/admin01/workspace/deep_landmark/cnn-face-data/lfw_5590/Ai_Sugiyama_0001.jpg')
        #img = cv2.imread('/home/matt/Documents/caffe/data/faceLaMa/img_celeba/000002.jpg')
        # 将图像转化为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将图像resize到256*256大小
        cv2.resize(gray,(256,256))
        # 先确定人脸框的大概位置（人脸框越准确，关键点一定越准确）
        # 可以首先通过人脸检测得到人脸框，再进行关键点预测
        bbox = BBox([70 ,190 ,70,200])
        # 利用cv2将人脸框在图像中画出来
        cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
        
        
        
        # 网络初始化
        get_landmark = Landmarker()
        # 经过３次level操作，得到最终的landmark
        final_landmark= get_landmark.detectLandmark(gray, bbox)
        # 对得到的final_landmark进行坐标转换，将相对坐标转换为绝对坐标
        final_landmark = bbox.reprojectLandmark(final_landmark)
        # 画图
        img = drawLandmark(img,  final_landmark)
        
        #　可以分别保存不同阶段生成的结果，进行对比
        #cv2.imwrite(result_path+'level1-'+image+'.jpg', img)
        #cv2.imwrite(result_path+'level1-'+image+'-level2-.jpg', img)
        cv2.imwrite(result_path+'level1-'+image+'-level2-'+'-level3.jpg', img)
        
        