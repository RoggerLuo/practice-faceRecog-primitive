import face_recognition
import cv2
# import os
import sys
from subprocess import call 
# from load_samples import load_samples 
from compare_face import compare_face ,get_repo,Model
import time



model = Model()
model.load_model(file_path='./model/me.face.model.h5')

# import gc
# gc.collect()

# from compare_face import get_repo 
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
#使用人脸识别分类器
classfier = cv2.CascadeClassifier("/Users/RogersMac/opcv/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml")

repeatHelper = []

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

counter = 0
last_match = []
color = (0, 255, 0)


# encodingsArr,chineseNameArr = load_samples('./photos/')

repo = get_repo('./testRepo')
last_time = 0
frame = []
while video_capture.isOpened():
    # Grab a single frame of video
    ok, frame = video_capture.read()


    # dis = (time.time() - last_time)
    # print(dis)
    # if dis > 0.5:
        # last_time = time.time()
        # print(dis)
    # else:
        # continue


    if not ok:            
        break  

    #将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 

    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        # for faceRect in faceRects:  #单独框出每一张人脸
    last_time = time.time()

    if len(faceRects) >= 1:
        faceRect = faceRects[0]
        x, y, w, h = faceRect                        

        #将当前帧保存为图片
        image = frame[y - 60: y + h + 60, x - 60: x + w + 60]

        # image = frame[y: y + h, x: x + w]
    
        # image = frame[y - 60: y + h + 60, x - 60: x + w + 60]
        if image.shape[0] != 0:
            rate = 64/image.shape[0]
            small_frame = cv2.resize(image, (0, 0), fx=rate, fy=rate)
            print(small_frame.shape)

            if process_this_frame:

                face_encodings = face_recognition.face_encodings(small_frame)
                print(len(face_encodings))
                if len(face_encodings) >= 1:
                    face_encoding = face_encodings[0]
                    

                    rs = compare_face(face_encoding,repo,model)
                    
                    print(rs)

                    print(time.time() - last_time)

            process_this_frame = not process_this_frame
        cv2.rectangle(frame, (x - 60, y - 60), (x + w + 60, y + h + 60), color, 2)

    # Resize frame of video to 1/4 size for faster face recognition processing
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

