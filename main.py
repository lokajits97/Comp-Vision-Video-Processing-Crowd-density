#!/usr/bin/python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function


import numpy as np
import cv2
import time

maxAllowed=2000
minAllowed=20



help_message = \
'''
USAGE: optical_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''
count = 0
#np.set_printoptions(threshold='nan')


def draw_flow(img, flow, step=16):
    
    #from the beginning to position 2 (excluded channel info at position 3)
   
      	
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    (h, w) = flow.shape[:2]
    (fx, fy) = (flow[:, :, 0], flow[:, :, 1])
       
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
     
    #exit()
    '''
    for i in range(len(v)):
    	for j in range(len(v[i])): 		
    		if v[i][j]<1:
    			v[i][j]=0
    		else:
    			v[i][j]=0XFF
    
    '''
    
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 0xFF   
    hsv[..., 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow('hsv', bgr)
    return bgr

count=0

if __name__ == '__main__':
    import sys
   # print (help_message)
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    cam = cv2.VideoCapture(fn)
    (ret, prev) = cam.read()

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = prev.copy()

    while cam.isOpened():
        (ret, img) = cam.read()
        vis = img.copy()    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	        
        """
        Computes a dense optical flow using the Gunnar Farneback’s algorithm.
        cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) → flow	
        """
        flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,5,5,3,5,1.1,cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray, flow))
      
        gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray1,0, 0xFF,cv2.THRESH_BINARY)[1]                       
        #thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=1)
        cv2.imshow('thresh',thresh)
        gray2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
        # loop over the contours
        
        """
        if len(cnts)>0:   
            maxAllowed = maxAvg = max(maxAvg, sum([cv2.contourArea(cnt) for cnt in cnts])/len(cnts))
        """
        modifiedCnts = [cnt for cnt in cnts if minAllowed < cv2.contourArea(cnt) < maxAllowed]
        
        	
        cv2.drawContours(vis, modifiedCnts, -1, (0,255,0), 1)        
        textNoOfPeople = "People: " +str(len(modifiedCnts))
        cv2.putText(vis, textNoOfPeople, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)    
        
        cv2.imwrite("frame%d.jpg" % count, vis)
        count=count+1
        print ('Read a new frame')
        cv2.imshow('Image', vis)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        
    cv2.destroyAllWindows()
