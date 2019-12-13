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
dt2 = 2500
# Height of frames in pixels
hPixels = 108

# Width of frames in pixels
wPixels = 192
"""
\\\\\\\\\\\\\\\\\\----------------------///////////////////
"""

Valid = ['TF01.mp4','TS01.mp4','KB01.mp4','300_02.mp4','WM01.mp4',
         'NCfOM01.mp4','MK01.mp4','GBH03.mp4']

Test = ['ARM02.mp4','FMJ01.mp4','DU02.mp4','MoS01.mp4','ASM01.mp4',
        'GBH01.mp4']

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
    
#Learn Transformations
cropDict = {}
truthDict = {}

#For each director
for key,val in dirDict.items():
    director = val
    z = 1
    
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
        
        vidcap = cv2.VideoCapture(key+ '/' + file)
        success,image = vidcap.read()
        t = 0
        t2 = 0
        breakBool = False
        
        while success:
            #Crop Image
            x,y,w,h = cropDict[file]
            crop = image[y:y+h,x:x+w]
            
            #Resize to 144 x 256
            newimg = cv2.resize(crop,(wPixels,hPixels))
            
            newimg = newimg.astype(np.float32)
            aggregate = newimg / Num
            
            t2 = t+dt
            
            for i in range(Num-1):
                vidcap.set(cv2.CAP_PROP_POS_MSEC,t2)
                t2 += dt
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
            
            #Choose the appropriate save location based on the filename
            if file in Test:
                saveFolder = 'transfer data/movie_test/'
            elif file in Valid:
                saveFolder = 'transfer data/movie_valid/'
            else:
                saveFolder = 'transfer data/movie_train/'
                
            #If the appropriate save location doesn't exist, make it!
            if not os.path.isdir(saveFolder+key):
                os.mkdir(saveFolder+key)
            
            #Save the picture to the appropriate folder
            cv2.imwrite(saveFolder+key+'/'+key+'_'+str(z)+'.png',aggregate)
            
            z += 1
            t += dt2
            vidcap.set(cv2.CAP_PROP_POS_MSEC,t)
            success,image = vidcap.read()
        
        print('Finished ' + file) 
    
with open('transfer data/truthVector.json','w') as fp:
    json.dump(truthDict,fp)
    
with open('transfer data/dirVector.json','w') as fp:
    json.dump(dirDict,fp)
    
    




