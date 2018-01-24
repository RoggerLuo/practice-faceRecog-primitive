#-*- coding: utf-8 -*-
import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from dataset import Dataset
import cv2
import sys
sys.path.append("..")
from Photo import IMAGE_SIZE, Photo


class Model:

    def __init__(self):
        self.model = None

    # 建立模型
    def build_model(self, dataset, nb_classes=2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        self.model.add(Dense(512,input_shape=dataset.input_shape))  # 14 Dense层,又被称作全连接层 #512
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.25))  # 16 Dropout层

        # roger
        # self.model.add(Dense(2048))  # 14 Dense层,又被称作全连接层
        # self.model.add(Activation('relu'))  # 15 激活函数层
        # self.model.add(Dropout(0.2))  # 16 Dropout层

        self.model.add(Dense(1024))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.2))  # 16 Dropout层

        # self.model.add(Dense(256))  # 14 Dense层,又被称作全连接层
        # self.model.add(Activation('relu'))  # 15 激活函数层
        # self.model.add(Dropout(0.2))  # 16 Dropout层

        self.model.add(Dense(64))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.2))  # 16 Dropout层

        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果

        # 输出模型概况
        self.model.summary()
    # 训练模型

    def train(self, dataset, batch_size=10, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        self.model.fit(dataset.train_images,
                       dataset.train_labels,
                       batch_size=batch_size,
                       nb_epoch=nb_epoch,
                       validation_data=(
                           dataset.valid_images, dataset.valid_labels),
                       shuffle=True)

    MODEL_PATH = './me.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(
            dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def predict(self,path):
        image = Photo(path).resize().image
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        rs = self.model.predict((image), 1)
        print(rs)

        # image = cv2.imread('./49.jpg')
        # image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
        # image = image.reshape([20,20,1])

        # layer = self.model.get_layer(index=14)
        # print(layer.get_weights())



if __name__ == '__main__':

    dataset = Dataset()
    dataset.load()
    print(dataset.input_shape)
    model = Model()
    model.build_model(dataset)


    # for i in range(10):
    #     #测试训练函数的代码
    # dataset = Dataset()
    # dataset.load()
    model.train(dataset,10,30)
    #     print('---------第%d次生成训练数据---------' %(i+2) )

    # dataset = Dataset()
    # dataset.load()
    # model.train(dataset,10,30)

    # dataset = Dataset()
    # dataset.load()
    # model.train(dataset,20,20)

    model.save_model(file_path = './model/me.face.model.h5')
    model.evaluate(dataset)
    
