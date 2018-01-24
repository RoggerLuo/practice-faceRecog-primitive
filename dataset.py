#-*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.cross_validation import train_test_split
import pickle
from keras import backend as K
from keras.utils import np_utils
import random

import sys
sys.path.append("..")
from Directory import Directory
from Photo import Photo, IMAGE_SIZE

class Dataset:

    def __init__(self):

        self.sample_number = 1000

        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        # self.path_name    = path_name

        # 当前库采用的维度顺序
        self.input_shape = None
        self.all_samples = []

    def pkl_load(self):
        with open('./keras_face_data.pkl', 'rb') as f:
            self.all_samples = pickle.load(f)
    def getRandomInd(self):
        return np.random.randint(0, len(self.all_samples) - 1)
    def get1pair_by_path(self, path, i):
        xs = [x for x in self.all_samples if x['dir_path'] == path]
        if i < 10:
            print('---------length of get1pair_by_path %d---------' % len(xs))
        randind = np.random.randint(0, len(xs) - 1)
        randind2 = np.random.randint(0, len(xs) - 1)
        # concateSample = np.concatenate(
        #     (xs[randind]['encoding'], xs[randind2]['encoding']))
        image = xs[randind]['encoding'] - xs[randind2]['encoding']
        return image, 1

    def get1pair(self, i):
        ind1 = self.getRandomInd()
        ind2 = self.getRandomInd()
        sample1 = self.all_samples[ind1]
        sample2 = self.all_samples[ind2]
        image1 = sample1['encoding']
        image2 = sample2['encoding']
        image = image1 - image2
        # concateSample = np.concatenate((image1, image2))
        # assert len(concateSample) == 256
        label = 1
        if sample1['dir_path'] != sample2['dir_path']:
            label = 0
        if i < 10:
            print(sample1['dir_path'])
            print(sample2['dir_path'])
            print(label)
        return image, label, sample1['dir_path']
    def preprocess(self):
        self.pkl_load()
        images = []
        labels = []
        for i in range(self.sample_number):
            try:
                image,label,dir_path = self.get1pair(i)
                images.append(image)
                labels.append(label)

                image2,label2 = self.get1pair_by_path(dir_path, i)
                images.append(image2)
                labels.append(label2)

            except Exception:
                print('读取失败%d' %i)
        print("--------一共张%d照片---------" % len(images))
        print("--------一共张%d照片---------" % len(images))

        return np.array(images), np.array(labels)


    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=2):

        # 加载数据集到内存
        images, labels = self.preprocess()  # load_dataset(self.path_name)
        # assert len(images[0]) == 128
        # assert len(labels[0]) == 1

        train_images, valid_images, train_labels, valid_labels = train_test_split(
            images, labels, test_size=0.3, random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(
            images, labels, test_size=0.5, random_state=random.randint(0, 100))

        # 输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels


        self.input_shape = (128,)
