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
from Directory import Directory

from Photo import Photo, IMAGE_SIZE


def get_repo(path):
    all_samples = Directory(path).find_images()
    converted_samples = []
    for sample in all_samples:
        print('正在读取图片%s' %sample['image_path'])
        image = face_recognition.load_image_file(sample['image_path'])
        try:
            encodings = face_recognition.face_encodings(image)
            if len(encodings) >= 1 :
                encoding = encodings[0]
                converted_samples.append({'encoding':encoding,'name':sample['image_name'],'dir_path':sample['image_dir_path']})
            else:
                print('   no face found here')
        except Exception:
            print('读取失败,在"%s"' % sample['image_dir_path'])

    return converted_samples



class Model:

    def __init__(self):
        self.model = None

    # MODEL_PATH = './model/me.face.model.h5'
    def load_model(self, file_path):
        self.model = load_model(file_path)

    def predict(self,encoding1,encoding2):
        # encoding2 = face_recognition.face_encodings(image2)[0]
        concateSample = encoding1 - encoding2 #np.concatenate((encoding1, encoding2))
        concateSample = concateSample.reshape((1,128))
        rs = self.model.predict(concateSample,1)
        return rs[0][1]


def compare_face(encoding1,repo,model):

    
    scores = [ model.predict(encoding1,item['encoding']) for item in repo]
    ind = scores.index(max(scores))
    # print(repo[ind]['name'])
    return repo[ind]['name']

