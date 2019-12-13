# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 07:51:02 2019

@author: Chris Naughton
"""
import cv2
import numpy as np
import os
import json

def blackCrop(oriimg):
    #Convert to greyscale
    gray = cv2.cvtColor(oriimg,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    #Find the contours to isolate the black border
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    #Crop the image and return it
    crop = oriimg[y:y+h,x:x+w]
    dim = (x,y,w,h)
    
    return crop, dim

dt = 3000

Test = ['ARM02.mp4','FMJ01.mp4','DU02.mp4','MoS01.mp4','ASM01.mp4',
        'GBH01.mp4']

#Create a dictionary of directors that we have movies from
dirDict = {}
i = 0
folders = next(os.walk('.'))[1]
for name in folders:
    #Check that there are mp4 files in the folder
    files = os.listdir(name)
    if len(files) == 0 or files[0][-3:] != 'mp4':
        continue
    
    #If there are mp4 files, save the file name as a director
    dirDict[name] = i
    i += 1
    
    
#Learn Transformations
cropDict = {}
x_train = np.array([]).reshape(0,5)
y_train = np.array([])

x_test = np.array([]).reshape(0,5)
y_test = np.array([])


#For each director
for key,val in dirDict.items():
    director = val
    
    #For each file, learn the cropping
    for file in os.listdir(key):
        vidcap = cv2.VideoCapture(key + '/' + file)
        #Capture at 50 seconds to not get black screen & appease Justice League
        vidcap.set(cv2.CAP_PROP_POS_MSEC,50000)
        success,image = vidcap.read()
        
        #First, crop out the black border
        size1 = np.shape(image)
        image2,dim = blackCrop(image)
        size2 = np.shape(image2)
        
        cropDict[file] = dim
        

    #Now, extract features from the movie files
    for file in os.listdir(key):
        print('Reading ' + file)
        
        if file in Test:
            testBool = True
        else:
            testBool = False
        
        vidcap = cv2.VideoCapture(key+ '/' + file)
        success,image = vidcap.read()
        t = 0
        
        while success:
            #Crop Image
            x,y,w,h = cropDict[file]
            image = image[y:y+h,x:x+w]
            
            rval = np.average(image[:,:,0])
            gval = np.average(image[:,:,1])
            bval = np.average(image[:,:,2])
            
            #Extract saturation values, S = (max(RGB)-min(RGB))/max(RGB)
            #Saturation is within [0,1], so make sure to use floats!
            imfl = image.astype(float)
            diff = np.max(imfl,2)-np.min(imfl,2)
            #Make sure to avoid dividing by zero!
            ind = diff!=0
            diff[ind] = diff[ind]/np.max(imfl,2)[ind]
            sval = np.average(diff)
            svar = np.var(diff)
            
            data = np.array([rval,gval,bval,sval,svar])
            
            if testBool:
                x_test = np.append(x_test,data.reshape(1,5),axis=0)
                y_test = np.append(y_test,np.array([director]),axis=0)
            else:
                x_train = np.append(x_train,data.reshape(1,5),axis=0)
                y_train = np.append(y_train,np.array([director]),axis=0)
            
            t += dt
            vidcap.set(cv2.CAP_PROP_POS_MSEC,t)
            success,image = vidcap.read()

#Save the files
np.savetxt('trainData.csv',x_train,delimiter=",")
np.savetxt('testData.csv',x_test,delimiter=",")
np.savetxt('trainTruth.csv',y_train,delimiter=",")
np.savetxt('testTruth.csv',y_test,delimiter=",")
            
            
            
            
            
            
            
            
    
    
