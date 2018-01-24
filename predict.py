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
import face_recognition

import sys
sys.path.append("..")
from Photo import IMAGE_SIZE, Photo


class Model:

    def __init__(self):
        self.model = None

    MODEL_PATH = './model/me.face.model.h5'

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def predict(self,path1,path2):
        image1 = face_recognition.load_image_file(path1)
        print(image1.shape)
        print(image1.shape)
        print(image1.shape)

        encoding1 = face_recognition.face_encodings(image1)[0]

        image2 = face_recognition.load_image_file(path2)
        encoding2 = face_recognition.face_encodings(image2)[0]

        concateSample = encoding1 - encoding2 #np.concatenate((encoding1, encoding2))
        concateSample = concateSample.reshape((1,128))
        rs = self.model.predict(concateSample,1)
        print(rs)

if __name__ == '__main__':
    model = Model()
    model.load_model(file_path='./model/me.face.model.h5')
    model.predict('./2.jpeg','./3.jpg')

