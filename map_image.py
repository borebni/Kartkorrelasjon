#!/usr/bin/env python

#Importing necessary libraries
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from PIL import Image
import os
import datetime
#Defining variable used for stopping the callback function
#callback_img = False#############33
#Creating new directory for internal map
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#Insert pathname below
save_folder = "pathname{}".format(timestamp)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#Fucntion to convert the occupancy grid (internal map) to a image
def occupancy_grid_callback(msg):  
    global callback_img
    global timestamp
    global save_folder

    #callback_img = True
    #Converting the occupancy grid to array
    occupancy_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    #Converting the array to an image
    occupancy_image = (255 * (1 - (occupancy_data / 100))).astype(np.uint8)
    occupancy_image = Image.fromarray(occupancy_image) 

    #Saving the internal map
    image_filename = os.path.join(save_folder, "map_screenshot_{}.png".format(timestamp))
    occupancy_image.save(image_filename)
    rospy.loginfo("Saved map screenshot as %s", image_filename)
    
    
    #if callback_img
    #Shutting down the ROS node
    rospy.signal_shutdown("The functions has been executed")
    #sys.exit(0)




if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('map_screenshot_node')

    # Subscribe to the /map topic
    rospy.Subscriber('/map', OccupancyGrid, occupancy_grid_callback)

    # Spin the ROS node to keep it running
    rospy.spin()
    #rospy.sleep(1.)
