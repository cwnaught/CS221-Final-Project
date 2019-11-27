# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:01:21 2019

@author: Chris Naughton
"""
import cv2
import numpy as np
import os
import json

"""
///////////////////Adjustable Parameters\\\\\\\\\\\\\\\\\\\
"""
# Number of frames to blend
Num = 5

# Spacing between each frame in milliseconds
dt = 500

# Height of frames in pixels
hPixels = 144

# Width of frames in pixels
wPixels = 256
"""
\\\\\\\\\\\\\\\\\\----------------------///////////////////
"""

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

#Create a list of movie files that we have already analyzed
if not os.path.isdir('data'):
    os.mkdir('data')

fileList = []
if os.path.isdir('./Data/color'):
    files = os.listdir('Data/color')
    for file in files:
        fileList.append(file[:-7])
else:
    os.mkdir('data/color')
    os.mkdir('data/blackwhite')
    
#Learn Transformations
cropDict = {}
truthDict = {}

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
        truthDict[file[:-4]] = director
        #cv2.imwrite('Processing/' + file[:-4] + '.png',image)
        

    #Now, save the movies as binary files of the concatenated arrays
    for file in os.listdir(key):
        print('Reading ' + file)
        
        #Check if we already have data on it
        if file[:-4] in fileList:
            print('Skipping ' + file)
            continue
        
        vidcap = cv2.VideoCapture(key+ '/' + file)
        success,image = vidcap.read()
        t = 0
        breakBool = False
        
        #Initialize an empty array to concatenate each blended frame to
        movie_c = np.array([], np.int64).reshape((3*hPixels*wPixels,0))
        movie_bw = np.array([], np.int64).reshape((hPixels*wPixels,0))
        
        while success:
            #Crop Image
            x,y,w,h = cropDict[file]
            crop = image[y:y+h,x:x+w]
            
            #Resize to 144 x 256
            newimg = cv2.resize(crop,(wPixels,hPixels))
            
            newimg = newimg.astype(np.float32)
            aggregate = newimg / Num
            
            for i in range(Num-1):
                t += dt
                vidcap.set(cv2.CAP_PROP_POS_MSEC,t)
                success,image = vidcap.read()   
                
                if not success:
                    breakBool = True
                    break
                
                x,y,w,h = cropDict[file]
                crop = image[y:y+h,x:x+w]
                
                #Resize to input dimensions
                image = cv2.resize(crop,(wPixels,hPixels))
                
                image = image.astype(np.float32)
                aggregate += image / Num
            
            if breakBool:
                break
            
            gray = cv2.cvtColor(aggregate,cv2.COLOR_BGR2GRAY)
            movie_c = np.concatenate((movie_c,aggregate.reshape((3*hPixels*wPixels,1))),axis=1)
            movie_bw = np.concatenate((movie_bw,gray.reshape((hPixels*wPixels,1))),axis=1)
            
            t += dt
            vidcap.set(cv2.CAP_PROP_POS_MSEC,t)
            success,image = vidcap.read()
            
        np.save('Data/color/' + file[:-4] + '_cl',movie_c)
        np.save('Data/blackwhite/' + file[:-4] + '_bw',movie_bw)
        print('Finished ' + file) 
    
with open('data/truthVector.json','w') as fp:
    json.dump(truthDict,fp)
    
with open('data/dirVector.json','w') as fp:
    json.dump(dirDict,fp)
    



