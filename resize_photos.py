#-*- coding: utf-8 -*-
import numpy as np
from Directory import Directory
from Photo import Photo, IMAGE_SIZE
import os


all_samples = Directory('./photos').find_images()
for sample in all_samples:
    p = Photo(sample['image_path']).resize(300,300)
    print('image_dir_path:::',sample['image_dir_path'])
    dst = os.path.join('./resized', sample['image_dir_path'])
    
    image_name = sample['image_name']
    print('image name in resize.py:',image_name)
    print('dst in resize.py:',dst)

    p.write(image_name,dst)
    
