#!/usr/bin/env python3
import numpy as np
import rospy
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from numpy import savetxt

def bev(images):
    margin_h = 283
    margin_w = 225
    width = 640
    height =420
    img_size = (width, height)
    
    ## These points would be normally found by extrinsic callibration (here they were chosen by just looking at the image)      
    src = np.float32([[258,275], [421,276],[84,420], [594,420]]) #last  for 680*420
    
    dst = np.float32([[margin_w,margin_h],[width-margin_w,margin_h],[margin_w,height],[width-margin_w,height]]) #final
    # Show calibration points in image
    image=np.copy(images)
    for point in src: cv2.circle(image,(point[0],point[1]), 5, (0,0,255), -1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        
    # Find transformation
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    savetxt('BEV.csv', M, delimiter=',')
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    #Calculae pixel per millimeter
    pixel_per_meter_x =  (width-2*margin_w)  /0.780   #Horizontal distance between src points in the real world (780 mm)
    pixel_per_meter_y =  (height-margin_h)/1.4   #Vertical distance between src points in the real world (1400 mm)
    print('y=',pixel_per_meter_y)
    print('x=',pixel_per_meter_x)
    print(M)
    while True:
        cv2.imshow('bev',image)
        cv2.imshow('bev2',warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return warped


def callback(data):
    bridge1 = CvBridge()
    
    while True:
	
        cv_image = bridge1.imgmsg_to_cv2(data, desired_encoding='passthrough')
        im_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
   	    #resize img to 680*420
        img= cv2.resize(im_rgb,(680, 420))
        #cv2.imshow('img',img)
        warped = bev(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    

def start():
    rospy.init_node('warped',anonymous=True)
    rospy.Subscriber('/cameraC/image_raw', Image,callback) 

    rospy.spin()

if __name__ == '__main__':

     try:
        start()

     except rospy.ROSInterruptException:
        pass

