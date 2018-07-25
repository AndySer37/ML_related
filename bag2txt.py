#import rosbag
#from cv_bridge import CvBridge, CvBridgeError
#import cv2
#bag = rosbag.Bag('trainset_2018-05-29-11-52-18.bag','r')
#n = 0
#bridge = CvBridge()
#for topic, msg, t in bag.read_messages(topics=['/kara/camera_node/image/compressed']):
#    print("Received an image!")
#    try:
#        # Convert your ROS Image message to OpenCV2
#        cvs_image = bridge.imgmsg_to_cv2(msg,"bgr8")
#    except CvBridgeError, e:
#        print(e)
#    else:
#        # Save your OpenCV2 image as a jpeg 
#        cv2.imwrite(str(n)+'.jpg', cv2_img)    
#        n+=1
#bag.close()

#!/usr/bin/python

# Extract images from a bag file.

#PKG = 'beginner_tutorials'
import roslib   #roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np

# Reading bag filename from command line or roslaunch parameter.
#import os
#import sys

class ImageCreator():


    def __init__(self):
        self.i = 0
        self.t = 0
        self.n = 0
        self.train_arr = np.zeros(15)
        self.test_arr = np.zeros(15)
        self.omega = 0
        self.bridge = CvBridge()
        f1 = open('little/train.txt', 'w')
        f2 = open('little/test.txt', 'w') 
        with rosbag.Bag('little_little_2018-06-03-16-49-19.bag', 'r') as bag: 
            for topic,msg,t in bag.read_messages():
                #print topic
                if topic == "/kara/joy_mapper_node/car_cmd":
                    #print topic, msg.header.stamp
                    self.omega = int(round(((msg.omega*(1/6.0))+1)*(14/2)))
                    self.t = 1
                elif topic == "/kara/camera_node/image/compressed":
                    if self.t == 1:
                        try:
                            #print topic, msg.header.stamp
                            #cv_image = self.bridge.imgmsg_to_cv2(msg)
                            np_arr = np.fromstring(msg.data, np.uint8)
                            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        except CvBridgeError as e:
                            print (e)
                        #timestr = "%.6f" %  msg.header.stamp.to_sec()
                        image_name = str(self.n)+ ".jpg" 
                        cv2.imwrite("little/"+image_name, cv_image)
                        
                        if self.i == 9:
                            self.test_arr[self.omega] += 1
                            if self.test_arr[self.omega] > 300:
                                continue
                            f2.write("little/"+image_name+" "+str(self.omega)+"\n")
                            self.i = 0
                        else:   
                            self.train_arr[self.omega] += 1
                            if self.train_arr[self.omega] > 300:
                                continue                           
                            f1.write("little/"+image_name+" "+str(self.omega)+"\n")
                            self.i += 1
                        print ("image crop:",self.n)
                        self.n += 1
                        self.t = 0
        f1.close()
        f2.close()  

if __name__ == '__main__':

    #rospy.init_node(PKG)

    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass
