# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:38:59 2019

@author: Chris Naughton
"""
import cv2
import numpy as np
import os

#Dictionary of directors that we have data on
dirDict = {'Coen Brothers':0,'Wes Anderson':1,'Zack Snyder':2}

trainData = np.zeros(5)
testData = np.zeros(5)
trainTruth = np.zeros(1)
testTruth = np.zeros(1)
"""
Truth Vector is an nx1 vector containing a number in {0,1,2} 
referencing a director
0 = Coen Brothers
1 = Wes Anderson
2 = Zac Snyder
"""

#Number of movies to add to training set
numTrainMovies = 4
movieNum = 1
testBool = False

#For each director
for key,val in dirDict.items():
    director = val

    #For each movie file associated with the director
    for file in os.listdir(key):
        #If we're looking at a movie for testing 
        if movieNum == numTrainMovies+1:
            movieNum = 1
            testBool = True
        
        vidcap = cv2.VideoCapture(key + '/' + file)
        success,image = vidcap.read()
        count = 0

        while success:
            #For every 72nd frame (3rd second in real time)
            if count % 72 == 0 and np.sum(image) > 0:
                #cv2.imwrite("frame%d.jpg" % count,image)
                #Find average pixel color values
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
                
                if testBool:
                    testData = np.vstack((testData,np.array([rval,gval,bval,sval,svar])))
                    testTruth = np.vstack((testTruth,np.array([director])))
                
                else:   
                    #Add the training data
                    trainData = np.vstack((trainData,np.array([rval,gval,bval,sval,svar])))
                    trainTruth = np.vstack((trainTruth,np.array([director])))
                
            success,image = vidcap.read()
            count += 1
        
        testBool = False
        movieNum += 1
            
        #Close the video
        print('Finished Reading ' + file)
        vidcap.release()
        
#Save the files
np.savetxt('trainData.csv',trainData[1:,:],delimiter=",")
np.savetxt('testData.csv',testData[1:,:],delimiter=",")
np.savetxt('trainTruth.csv',trainTruth[1:,:],delimiter=",")
np.savetxt('testTruth.csv',testTruth[1:,:],delimiter=",")