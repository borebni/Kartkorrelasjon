#Importing necessary libraries
import os 
import glob
import cv2 as cv
import numpy as np  
import datetime
import time
from matplotlib import pyplot as plt
#importerer funksjoner fra rotate.py
from rotate import change_angle_to_radius_unit, rotate

#Finding and reading map image. Insert path to file 
imgfile = 'pathname/filename' 

#Reading image file while converting to greyscale
img = cv.imread(imgfile, cv.IMREAD_GRAYSCALE)

#Selecting scalingfactor
scaling_factor_x = 1
scaling_factor_y = 1

#Scaling the image according to scaling factor
img_scaled = cv.resize(img, None, fx= scaling_factor_x, fy= scaling_factor_y, interpolation= cv.INTER_LINEAR)

#Defining image variable for visualization
img_show = cv.imread(imgfile)
assert img is not None, "no map"

#Finding and reading the latest internal map made by HECTOR_SLAM. Insert path
base_dir = "pathname"
test_dir = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
test_dir.sort(reverse=True)

if test_dir:
    last_test = test_dir[0]

    templ_files = [f for f in os.listdir(last_test) if f.endswith(".png")]
    if templ_files:
        templfile = os.path.join(last_test, templ_files[0])

#Reading template file while converting to greyscale
templ = cv.imread(templfile, cv.IMREAD_GRAYSCALE)

#Mirroring the template.
templ_mirror = np.fliplr(templ)

templ_show = cv.imread(templfile)
templ_show_mirror = np.fliplr(templ_show_cropped)
assert templ is not None, "no local map"

#Finding the dimensions of the template 
w, h = templ_mirror.shape[::-1]  

#Defining center of template as the pivot point for rotating
pivot = (w//2, h //2)

#Defining method used for Template Matching
match_method = cv.TM_CCOEFF
	
#Making array for results
max_val_loc_array = []

#rotating template image using function from rotate.py
for angle in range(0, 360, 1):
  radius = change_angle_to_radius_unit(angle)
  new_templ = rotate(templ_mirror, radius, pivot, (w, h))
  result = cv.matchTemplate(img_scaled, new_templ, match_method)
  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
  max_val_loc_array.append((max_val, max_loc, min_loc, angle))

#Sorting results in ascending order
results_sorted = sorted(max_val_loc_array, key=lambda elem: elem[0], reverse=True)

#Finding corners of the best matching area
top_left = results_sorted[0][1] 
bottom_right = (top_left[0] + w, top_left[1] + h)

#Finding the center of the area; the plattforms starting position
rover_position = (top_left[0]+((bottom_right[0]-top_left[0])//2), top_left[1]+((bottom_right[1]-top_left[1])//2))

#Marking best matching area on map image for visualization
cv.rectangle(img_show, top_left, bottom_right, 255, 2) 

#Plotting result
plt.subplot(121), plt.imshow(templ_show)
plt.subplot(121), plt.imshow(img_scaled)
plt.title(f"Internal map" ), plt.xticks([]), plt.yticks([]) 
plt.subplot(122), plt.imshow(img_show)
plt.scatter(rover_position[0], rover_position[1], color="red")
plt.title('Global map'), plt.xticks([]), plt.yticks([])

#Saving position data and visualization in new directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#Insert filename
file_name = f'name{timestamp}.png'
#Insert path
save_folder = "pathname{}".format(timestamp)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
file_path = os.path.join(save_folder, file_name)
plt.savefig(file_path) 
#Insert filename
position_file = os.path.join(save_folder, "name.txt".format(timestamp))
with open(position_file, "a") as file:
    file.write("{} ".format(rover_position)) 
