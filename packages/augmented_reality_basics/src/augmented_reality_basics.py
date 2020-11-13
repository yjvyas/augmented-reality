#!/usr/bin/env python3
import numpy as np
import os
import rospy
import sys
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import CompressedImage, CameraInfo
from math import pi
import yaml
import cv2
import math
from cv_bridge import CvBridge

class Augmenter(DTROS):

    def __init__(self, node_name, map_file="", veh_name=""):
        """Augmenter Node.
        This implements basic Augmented Reality functionality.
        """

        # Initialize the DTROS parent class
        super(Augmenter, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = veh_name
        rospy.loginfo(self.veh_name)

        self.map = self.readYamlFile(map_file)
        self.map_name = os.path.basename(map_file).split('.')[0]
        calibration_fname = '/data/config/calibrations/camera_extrinsic/default.yaml'
        if 'CALIBRATION_FILE' in os.environ:
            calibration_fname = os.environ['CALIBRATION_FILE']
        self.homography = self.readYamlFile(calibration_fname)
        H = np.reshape(np.array(self.homography['homography']), [3, 3])
        self.H = np.linalg.inv(H)
        rospy.loginfo('Map received is:')
        rospy.loginfo(self.map)

        self.sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.callback)
        self.camera_info = rospy.Subscriber(f'/{self.veh_name}/camera_node/camera_info', CameraInfo, self.set_camera_info)
        self.pub = rospy.Publisher(f'/{self.veh_name}/augmented_reality/{self.map_name}/image/compressed', CompressedImage, queue_size=10)
        self.colours = {'red':[0, 0, 255], 'green': [0, 255, 0], 'blue':[255, 0, 0], 'yellow':[0,255,255]}

        self.cvbr = CvBridge()

        # camera calibration info
        self.P = None
        self.K = None
        self.D = None
        self.log("Initialized augmented_reality_basics_node...")

    def process_image(self, img):
        """Undistors raw images."""
        if self.P is not None:
            img = cv2.undistort(img, self.K, self.D)

            return img
        else:
            return None

    def ground2pixel(self, point, H):
        """Transforms points into ground coordinates. point is the point xyz as a list.
        H is the homography matrix."""
        point = np.array(point[0:2] + [1])

        pixel = np.dot(H, point)
        pixel = (1/pixel[-1])*pixel
        pixel = pixel[:2].reshape([1,2])
        pixel = tuple(np.rint(pixel).flatten().astype(int).tolist())
        return pixel


    def render_segments(self, img):
        """Renders segments on the image, given points. Img has to be of cv2 type.
        Segments are provided from the map."""
        for segment in self.map['segments']:
            if self.map['points'][segment['points'][0]][0] == 'axle':
                P_start = self.H
            elif self.map['points'][segment['points'][0]][0] == 'camera':
                P_start = np.identity(4)
            if self.map['points'][segment['points'][1]][0] == 'axle':
                P_end = self.H
            elif self.map['points'][segment['points'][1]][0] == 'camera':
                P_end = np.identity(4)
            start_pixel = self.ground2pixel(self.map['points'][segment['points'][0]][1], P_start)
            end_pixel = self.ground2pixel(self.map['points'][segment['points'][1]][1], P_end)
            
            img = cv2.line(img, start_pixel, end_pixel, self.colours[segment['color']])
        return img

    def callback(self, msg):
        """Callback for augmenter class."""
        image = self.cvbr.compressed_imgmsg_to_cv2(msg)
        image = self.process_image(image) # undistort
        if image is not None:
            image = self.render_segments(image)
            self.pub.publish(self.cvbr.cv2_to_compressed_imgmsg(image))
        

    def set_camera_info(self, msg):
        """Receives camera info from the appropriate topic and sets it."""
        self.K = np.reshape(msg.K, [3,3])
        self.P = np.reshape(msg.P, [3, 4])
        self.D = msg.D

    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                        %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


if __name__ == '__main__':
    myargv = rospy.myargv(argv=sys.argv)
    node = Augmenter(node_name='augmented_reality_basics_node', map_file=myargv[1], veh_name=myargv[2])
    # Keep it spinning to keep the node alive
    # node.run()
    
    rospy.spin()
    rospy.loginfo("augmented_reality_basics_node is up and running...")