# Source: https://github.com/prayat-pu/Computer_vision/blob/master/tranform_image/rotation_img.py

import numpy as np
#function for converting degrees to radius
def change_angle_to_radius_unit(angle):
    angle_radius = angle * (np.pi/180)
    return angle_radius

def rotate(src_img,angle_of_rotation,pivot_point,shape_img):

    #creating rotation matrix
    rotation_mat = np.transpose(np.array([[np.cos(angle_of_rotation),-np.sin(angle_of_rotation)], [np.sin(angle_of_rotation),np.cos(angle_of_rotation)]]))
   
    w,h = shape_img
    
    pivot_point_x =  pivot_point[0]
    pivot_point_y = pivot_point[1]
    
    new_img = np.zeros(src_img.shape,dtype='u1') 
    #rotating image
    for height in range(h): #h = number of row
        for width in range(w): #w = number of col
            
            xy_mat = np.array([[width-pivot_point_x],[height-pivot_point_y]])
            rotate_mat = np.dot(rotation_mat,xy_mat)

            new_x = pivot_point_x + int(rotate_mat[0])
            new_y = pivot_point_y + int(rotate_mat[1])


            if (0<=new_x<=w-1) and (0<=new_y<=h-1): 
                new_img[new_y,new_x] = src_img[height,width]

    return new_img
