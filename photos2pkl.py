#-*- coding: utf-8 -*-
# import random
# import numpy as np
# # from sklearn.cross_validation import train_test_split
# # from conv2arr import load_dataset, resize_image, IMAGE_SIZE
# import random

import face_recognition
import pickle

import sys
sys.path.append("..")
from Directory import Directory
from Photo import Photo, IMAGE_SIZE


all_samples = Directory('./photos').find_images()
converted_samples = []
for sample in all_samples:
    print('正在读取图片%s' %sample['image_path'])
    image = face_recognition.load_image_file(sample['image_path'])
    try:
        encodings = face_recognition.face_encodings(image)
        if len(encodings) >= 1 :
            encoding = encodings[0]
            converted_samples.append({'encoding':encoding,'dir_path':sample['image_dir_path']})
        else:
            print('   no face found here')
    except Exception:
        print('读取失败,在"%s"' % sample['image_dir_path'])


print('一共读取了%d张图片' %len(converted_samples))
with open('./keras_face_data.pkl', 'wb') as f:
    pickle.dump(converted_samples, f, True)
print('-----')
print('保存成功')



