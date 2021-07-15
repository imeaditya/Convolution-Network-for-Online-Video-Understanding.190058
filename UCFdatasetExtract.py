from google.colab import drive
drive.mount('/content/drive')

#Extracting videos of UCF Dataset from its .rar file
!unrar x "/content/drive/MyDrive/UCF101.rar" "/content/drive/MyDrive/ucf101_extracted/"

import os
from random import shuffle
import glob
import numpy as np
import cv2 
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas
import pickle
directory = '/content/drive/MyDrive/UCF-101'
# Get the labels from the directory 
# labels = os.listdir(directory)
labels = [x[1] for x in os.walk(directory)][0]   
NUM_LABELS = len(labels)
# Sort the labels to be consistent
# build dictionary for indexes
label_indexes = {labels[i]: i for i in range(0, len(labels))}  
sorted(labels)
# print(label_indexes)
# get the file paths
data_files= []#to store paths and id
i =-1 #label id
for file in os.listdir(directory):#grabbing the frames and the id
    i+=1 #the id everytime change title folder
    filename = os.fsdecode(directory+ '/' + file)
    for file1 in os.listdir(filename):
      
        if file1.endswith('.avi'):
            # Playing video from file:
            name1, ext = os.path.splitext (file1)
            cap = cv2.VideoCapture(filename+'/' +file1)
            try:
                if not os.path.exists('/content/drive/MyDrive/Ucf-101onevideoframes/' + file + '/'+ name1):
                    os.makedirs('/content/drive/MyDrive/Ucf-101onevideoframes/' + file+ '/'+ name1)
            except OSError:
                print ('Error: Creating directory of data')
            currentFrame = 0 #every new id start at 0
            while(True):
            # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == False:
                    break
                # Saves image of the current frame in jpg file
                name = '/content/drive/MyDrive/Ucf-101onevideoframes/' + file+ '/'+ name1 + '/' + file + str(currentFrame) + '.jpg'
                cv2.imwrite(name, frame)
                data_files.append((i,name))
                # To stop duplicate images
                currentFrame += 1
            # When everything done, release the capture
            cap.release()
            break
