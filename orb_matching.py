
#importing necessary libraries
import os 
import numpy as np
import cv2 as cv
import datetime 
from matplotlib import pyplot as plt 

#Saving the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#Defining path to a new folder. Insert path
save_folder = "pathname{}".format(timestamp)
#Creating folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#Selecting and reading reference map. Insert path to file
imgfile = 'pathname'
img = cv.imread(imgfile)
#Extracting internal map. Insert path
base_dir = "pathname"
#Sorting parent directory in ascending order based on timestamp
test_dir = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
test_dir.sort(reverse=True)

#Extracting internal map from newest directory
if test_dir:
    last_test = test_dir[0]
    templ_files = [f for f in os.listdir(last_test) if f.endswith(".png")]
    if templ_files:
        templfile = os.path.join(last_test, templ_files[0])

#Reading internal map
templ = cv.imread(templfile)
#Creating ORB-object 
orb = cv.ORB_create(nfeatures=400,scaleFactor=1.1, nlevels=10, edgeThreshold=20, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE)

#Detection features and computing descriptors in reference map
img_kp = orb.detect(img, None)
img_kp, img_des = orb.compute(img, img_kp)
#Drawing matching pixels for visualization
img_result = cv.drawKeypoints(img, img_kp, None, color=(0,255,0), flags=0)
#defining image variable
result = img
#Detection features and computing descriptors in internal map
templ_kp = orb.detect(templ, None)
templ_kp, templ_des = orb.compute(templ, templ_kp)
#Drawing matching pixels for visualization
templ_result = cv.drawKeypoints(templ, templ_kp, None, color=(0,255,0), flags=0)
#Making and saving plots displaying features in maps
plt.imshow(img_result)
plt.title('Image')
file_name = f'orb_plot_image_{timestamp}.png'
file_path = os.path.join(save_folder, file_name)
plt.savefig(file_path)
plt.close()

plt.imshow(templ_result)
plt.title('Template')
file_name = f'orb_plot_template_{timestamp}.png'
file_path = os.path.join(save_folder, file_name)
plt.savefig(file_path)
plt.close()

#Creating a Brute Force Matcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

#Matching features in maps based on descriptors
matches = bf.match(img_des, templ_des)

#Sorting matches based on distance
matches = sorted(matches, key = lambda x : x.distance)
#Selectiong matches with shortest distance
top_matches = matches[:40]

#Defining size of a window
window_size = 40
window_matches_list = []
i=0
#Sliding the window over the reference map 
step_size=10
for y in range(0, img.shape[0] - window_size, step_size):
    for x in range(0, img.shape[1] - window_size, step_size):
        #Checking current window
        window = img[y:y + window_size, x:x + window_size]

        #Counting number of matches in the current window
        window_matches = [match for match in top_matches if
                           y <= img_kp[match.queryIdx].pt[1] < y + window_size and
                           x <= img_kp[match.queryIdx].pt[0] < x + window_size]

        #Storing number of matches and coordinate of the window
        window_matches_list.append((len(window_matches), (x,y)))
       
#Sorting in descending order 
window_matches_list.sort(reverse=True)
#Extracting coordinates of window containing the most matches
top_window_position = window_matches_list[0][1]
top_x, top_y = top_window_position

#Extracting top matches within the window  
top_window_matches = [match for match in top_matches if
                      top_y <= img_kp[match.queryIdx].pt[1] < top_y + window_size and
                      top_x <= img_kp[match.queryIdx].pt[0] < top_x + window_size]

#Extracting keypoints and descriptors from the matches in the window with the most matches 
top_window_kp = [img_kp[match.queryIdx] for match in top_window_matches]
top_window_des = np.array([img_des[match.queryIdx] for match in top_window_matches])

template_kp = [templ_kp[match.trainIdx] for match in top_window_matches]
template_des = np.array([templ_des[match.trainIdx] for match in top_window_matches])

#Finding the mean coordinate of all keypoints in the area
mean_coords = np.mean(np.array([kp.pt for kp in top_window_kp]), axis=0)

template_mean_coordinate = np.mean(np.array([kp.pt for kp in template_kp]), axis=0)

#Plotting the image with a rectangle representing the area with the most matches
result_img = img.copy()
cv.rectangle(result_img, (top_window_position[0], top_window_position[1]),
             (top_x + window_size, top_y + window_size), (0, 255, 0), 2)

#Marking the mean coordinate with green dot
cv.circle(result_img, (int(mean_coords[0]), int(mean_coords[1])), 4, (0, 255, 0), -1)

#Plotting and saving result
plt.imshow(result_img)
plt.title('Area with the Most Matches and mean coordinate')
#insert filename
file_name = f'filename{timestamp}.png'
file_path = os.path.join(save_folder, file_name)
plt.savefig(file_path)
plt.savefig(file_path)
plt.close()

#Drawing matches for plot
result = cv.drawMatches(img, img_kp, templ, templ_kp, top_matches, result, flags = 2)

#Displaying the best matching points
plt.rcParams['figure.figsize'] = [14.0, 10.0]
plt.title('Best Matching Points')
plt.imshow(result)

#Insert filename
file_name = f'name{timestamp}.png'
#Saving plots and position data
save_folder = "pathnname{timestamp}"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
file_path = os.path.join(save_folder, file_name)
plt.savefig(file_path)
plt.close()
#Insert filename
position_file = os.path.join(save_folder, "name{}.txt".format(timestamp))
#Writing to file
with open(position_file, "a") as file:
    file.write("Mean coordinate {}, Mean coordiate in template: {}".format(mean_coords, template_mean_coordinate))
    file.close()

